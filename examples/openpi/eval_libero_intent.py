"""Evaluate a Pi0.5 checkpoint on the LIBERO goal suite, with optional slot-intent.

Direct (in-process) rollout — adapts external/openpi/examples/libero/main.py to load
the policy locally via create_trained_policy and (optionally) inject a 64D intent
sampled from the MIP flow map (mip.pi05_intent.IntentGenerator) at each replan.

Per-step obs follows openpi's LIBERO convention:
  state = [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]   (8D, matches our norm stats)
For the intent sidecar (MIP flow map) we additionally build the MIP obs window:
  state15 = [eef_pos, axisangle, gripper_qpos, joint_pos]  + agentview/eye_in_hand @128.

Outputs per-task + mean success rate to stdout and a JSON file.

Usage (baseline, no intent):
  python examples/openpi/eval_libero_intent.py --config-name pi05_libero \
    --checkpoint-dir /data/.../pi05_libero --out logs/eval_baseline.json
Usage (intent):
  python examples/openpi/eval_libero_intent.py --config-name pi05_libero_intent \
    --checkpoint-dir /data/.../slot_intent_goal_run1/18000 --intent \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt /data/.../slot_intent/models/model_best.pt \
    --out logs/eval_intent.json
"""

import argparse
import collections
import functools
import json
import math
import pathlib

import numpy as np
import torch

# Must run before any LIBERO env import: patches robosuite 1.5.x API back to what
# LIBERO 1.4 expects (load_controller_config, robot/gripper shims). Same fix MIP uses.
import mip.envs.libero._robosuite_compat  # noqa: F401, E402

# LIBERO's *.pruned_init init-state files predate torch's weights_only=True default;
# load them as trusted local pickles (the MIP flow-map loader already does this).
torch.load = functools.partial(torch.load, weights_only=False)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RES = 256
MAX_STEPS = 300  # libero_goal: longest demo ~270 steps
NUM_STEPS_WAIT = 10


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _resize(img_hwc, size):
    from PIL import Image

    return np.asarray(Image.fromarray(img_hwc).resize((size, size), Image.BILINEAR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--task-suite", default="libero_goal")
    ap.add_argument("--num-trials-per-task", type=int, default=20)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rotate-images", action="store_true",
                    help="rotate 180 (off-the-shelf pi05_libero/PI convention); omit for our finetune")
    ap.add_argument("--intent", action="store_true")
    ap.add_argument("--intent-task-config", default="libero_goal_suite_image_slot_intent")
    ap.add_argument("--intent-ckpt", default=None)
    ap.add_argument("--config-dir", default="examples/configs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import openpi.training.config as _config
    from openpi.policies import policy_config as _policy_config

    np.random.seed(args.seed)

    # ---- load policy ----
    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)

    # ---- optional intent generator (MIP flow map sidecar) ----
    gen = None
    if args.intent:
        from mip.pi05_intent import IntentGenerator
        assert args.intent_ckpt, "--intent requires --intent-ckpt"
        gen = IntentGenerator(args.intent_task_config, args.intent_ckpt, args.config_dir, device=args.device)
    obs_steps = gen.obs_steps if gen is not None else 1

    # ---- LIBERO suite ----
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.n_tasks

    def mip_obs(obs):
        """Build (state15, agentview128_chw, wrist128_chw) for the flow map, MIP convention."""
        st = np.concatenate([obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                             obs["robot0_gripper_qpos"], obs["robot0_joint_pos"]]).astype(np.float32)
        agv = _resize(obs["agentview_image"], 128).transpose(2, 0, 1)        # CHW uint8
        wr = _resize(obs["robot0_eye_in_hand_image"], 128).transpose(2, 0, 1)
        return st, agv, wr

    results = {}
    total_ep, total_succ = 0, 0
    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RES, camera_widths=ENV_RES)
        try:
            env.seed(args.seed)  # robosuite 1.5.x may not expose a callable seed(); init states fix the reset anyway
        except (TypeError, AttributeError):
            pass
        t_ep, t_succ = 0, 0
        for ep in range(args.num_trials_per_task):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            plan = collections.deque()
            win = collections.deque(maxlen=obs_steps)
            done = False
            for t in range(MAX_STEPS + NUM_STEPS_WAIT):
                if t < NUM_STEPS_WAIT:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    continue
                if not plan:
                    agv = obs["agentview_image"]
                    wr = obs["robot0_eye_in_hand_image"]
                    if args.rotate_images:
                        agv = np.ascontiguousarray(agv[::-1, ::-1])
                        wr = np.ascontiguousarray(wr[::-1, ::-1])
                    element = {
                        "observation/image": agv,
                        "observation/wrist_image": wr,
                        "observation/state": np.concatenate([
                            obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32),
                        "prompt": str(task.language),
                    }
                    if gen is not None:
                        st, a128, w128 = mip_obs(obs)
                        win.append((st, a128, w128))
                        while len(win) < obs_steps:
                            win.appendleft(win[0])
                        sw = np.stack([w[0] for w in win])[None]          # (1,To,15)
                        iw = {"agentview_rgb": np.stack([w[1] for w in win])[None],
                              "eye_in_hand_rgb": np.stack([w[2] for w in win])[None]}
                        element["intent"] = gen.intent_for_windows(sw, iw)[0].astype(np.float32)
                    action_chunk = np.asarray(policy.infer(element)["actions"])
                    plan.extend(action_chunk[: args.replan_steps])
                obs, _, done, _ = env.step(plan.popleft().tolist())
                if gen is not None:  # keep the obs window fresh between replans
                    win.append(mip_obs(obs))
                if done:
                    break
            t_ep += 1
            t_succ += int(done)
            total_ep += 1
            total_succ += int(done)
        env.close()
        sr = t_succ / max(t_ep, 1)
        results[str(task.language)] = sr
        print(f"[{args.task_suite}] task {task_id} SR={sr:.2f} ({t_succ}/{t_ep}) :: {task.language}", flush=True)

    mean_sr = total_succ / max(total_ep, 1)
    summary = {"config": args.config_name, "checkpoint": args.checkpoint_dir, "intent": args.intent,
               "task_suite": args.task_suite, "num_trials_per_task": args.num_trials_per_task,
               "mean_sr": mean_sr, "total": f"{total_succ}/{total_ep}", "per_task": results}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== {args.config_name} intent={args.intent} MEAN SR = {mean_sr:.3f} ({total_succ}/{total_ep}) -> {args.out}")


if __name__ == "__main__":
    main()
