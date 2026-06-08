"""Evaluate Pi0.5 (with/without slot-intent) under radial XY perturbation of the target object.

For each (task, δ) pair, shifts the target object's XY position by δ metres in
evenly-spaced directions before each episode and measures success rate.  Running
both configs side-by-side produces the SR-vs-δ degradation curves that test whether
intent conditioning helps generalise to OOD object positions.

Training-distribution note: across all libero_goal init states, object x-coordinates
have zero variance (fixed per-object column on the table); y-coordinates vary with
std ≈ 0.01 m.  A radial perturbation in XY therefore probes two distinct regimes:
  - y component alone: in-distribution direction, OOD at δ ≳ 0.03 m (3 σ)
  - x component:       immediately OOD at any δ > 0

Tasks 0 ("open middle drawer") and 7 ("turn on stove") are skipped: their targets
are a slide joint / hinge joint, not a free-body object with an XY position.

Qpos layout (identical across all 10 libero_goal tasks):
  state[9:12]   — akita_black_bowl  xyz
  state[16:19]  — cream_cheese      xyz
  state[23:26]  — wine_bottle       xyz
  state[30:33]  — plate             xyz

Usage (no-intent baseline):
  python examples/openpi/eval_libero_perturbed.py \\
    --config-name pi05_libero_nointent \\
    --checkpoint-dir /data/.../nointent_goal_run1/30000 \\
    --out logs/perturb_nointent.json

Usage (slot-intent conditioned):
  python examples/openpi/eval_libero_perturbed.py \\
    --config-name pi05_libero_intent \\
    --checkpoint-dir /data/.../slot_intent_goal_run1/30000 \\
    --intent \\
    --intent-task-config libero_goal_suite_image_slot_intent \\
    --intent-ckpt /data/.../slot_intent/models/model_best.pt \\
    --out logs/perturb_intent.json

Then plot the two JSON files together to get the SR-vs-δ comparison.
"""

import argparse
import collections
import functools
import json
import math
import pathlib
import sys

import numpy as np
import torch

import mip.envs.libero._robosuite_compat  # noqa: F401
torch.load = functools.partial(torch.load, weights_only=False)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RES = 256
MAX_STEPS = 300
NUM_STEPS_WAIT = 10

# qpos index of the xyz start for each free-body object (free joint = 7 qpos: xyz + quat)
OBJECT_QPOS_XYZ = {
    "bowl":         9,
    "cream_cheese": 16,
    "wine_bottle":  23,
    "plate":        30,
}

# task_id -> (object_name, qpos_xyz_start) to perturb; None = skip
TASK_TARGET = {
    0: None,                    # "open middle drawer"       — drawer is a slide joint
    1: ("bowl", 9),             # "put the bowl on the stove"
    2: ("wine_bottle", 23),     # "put the wine bottle on top of the cabinet"
    3: ("bowl", 9),             # "open top drawer and put bowl inside"
    4: ("bowl", 9),             # "put the bowl on top of the cabinet"
    5: ("plate", 30),           # "push the plate to the front of the stove"
    6: ("cream_cheese", 16),    # "put the cream cheese in the bowl"
    7: None,                    # "turn on the stove"        — button is a hinge joint
    8: ("bowl", 9),             # "put the bowl on the plate"
    9: ("wine_bottle", 23),     # "put the wine bottle on the rack"
}


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _resize(img_hwc, size):
    from PIL import Image
    return np.asarray(Image.fromarray(img_hwc).resize((size, size), Image.BILINEAR))


def _perturb(state, qpos_xyz_start, dx, dy):
    """Copy state and shift the target object's XY by (dx, dy) metres."""
    s = state.copy()
    s[qpos_xyz_start]     += dx
    s[qpos_xyz_start + 1] += dy
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--task-suite", default="libero_goal")
    ap.add_argument("--tasks", default=None,
                    help="comma-separated task IDs (0-9); default = all with a perturbable target")
    ap.add_argument("--task-target-qpos", type=int, default=None,
                    help="override TASK_TARGET for all tasks: qpos index of target object xyz "
                         "(e.g. 9 for bowl). Use when running a suite other than libero_goal.")
    ap.add_argument("--deltas", default="0.0,0.02,0.05,0.10,0.15,0.20",
                    help="comma-separated perturbation magnitudes in metres")
    ap.add_argument("--num-trials-per-delta", type=int, default=10,
                    help="episodes per (task, delta); directions are evenly spaced in [0, 2π)")
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rotate-images", action="store_true",
                    help="rotate 180° (needed for off-the-shelf pi05_libero; omit for our finetune)")
    ap.add_argument("--intent", action="store_true")
    ap.add_argument("--intent-override", default="flow", choices=["flow", "zero", "random"],
                    help="flow=normal sidecar; zero=all-zeros intent; random=Gaussian noise intent")
    ap.add_argument("--intent-task-config", default="libero_goal_suite_image_slot_intent")
    ap.add_argument("--intent-ckpt", default=None)
    ap.add_argument("--config-dir", default="examples/configs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="logs/eval_perturbed.json")
    args = ap.parse_args()

    deltas = [float(d) for d in args.deltas.split(",")]
    n = args.num_trials_per_delta

    sys.path.insert(0, "external/openpi/src")
    import openpi.training.config as _config
    from openpi.policies import policy_config as _policy_config

    np.random.seed(args.seed)

    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)

    gen = None
    if args.intent:
        from mip.pi05_intent import IntentGenerator
        assert args.intent_ckpt, "--intent requires --intent-ckpt"
        gen = IntentGenerator(args.intent_task_config, args.intent_ckpt, args.config_dir,
                              device=args.device)
    obs_steps = gen.obs_steps if gen is not None else 1

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()

    if args.tasks is not None:
        task_ids = [int(x) for x in args.tasks.split(",")]
    else:
        task_ids = [tid for tid in range(suite.n_tasks) if TASK_TARGET[tid] is not None]

    # Evenly-spaced directions: trial i uses the same angle at every δ, so curves are
    # directly comparable (same spatial direction probed at all magnitudes).
    angles = np.linspace(0.0, 2 * math.pi, n, endpoint=False)

    def _mip_obs(obs):
        st = np.concatenate([obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                             obs["robot0_gripper_qpos"], obs["robot0_joint_pos"]]).astype(np.float32)
        agv = _resize(obs["agentview_image"], 128).transpose(2, 0, 1)
        wr  = _resize(obs["robot0_eye_in_hand_image"], 128).transpose(2, 0, 1)
        return st, agv, wr

    # results[delta_str][task_id_str] = {sr, succ, ep, task, target_obj, delta}
    results: dict = {str(d): {} for d in deltas}

    for task_id in task_ids:
        task = suite.get_task(task_id)
        target = TASK_TARGET.get(task_id)
        if target is None and args.task_target_qpos is None:
            print(f"[task {task_id}] SKIP — no free-joint target :: {task.language}", flush=True)
            continue

        if args.task_target_qpos is not None:
            obj_name, qpos_start = "override", args.task_target_qpos
        else:
            obj_name, qpos_start = target
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RES, camera_widths=ENV_RES)
        try:
            env.seed(args.seed)
        except (TypeError, AttributeError):
            pass

        for delta in deltas:
            t_ep = t_succ = 0
            for trial_i in range(n):
                base = init_states[trial_i % len(init_states)]
                if delta > 0.0:
                    dx = delta * math.cos(angles[trial_i])
                    dy = delta * math.sin(angles[trial_i])
                    init_state = _perturb(base, qpos_start, dx, dy)
                else:
                    init_state = base

                env.reset()
                obs = env.set_init_state(init_state)
                plan = collections.deque()
                win  = collections.deque(maxlen=obs_steps)
                done = False

                for t in range(MAX_STEPS + NUM_STEPS_WAIT):
                    if t < NUM_STEPS_WAIT:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        continue
                    if not plan:
                        agv = obs["agentview_image"]
                        wr  = obs["robot0_eye_in_hand_image"]
                        if args.rotate_images:
                            agv = np.ascontiguousarray(agv[::-1, ::-1])
                            wr  = np.ascontiguousarray(wr[::-1, ::-1])
                        element = {
                            "observation/image":       agv,
                            "observation/wrist_image": wr,
                            "observation/state": np.concatenate([
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            ]).astype(np.float32),
                            "prompt": str(task.language),
                        }
                        if gen is not None:
                            st, a128, w128 = _mip_obs(obs)
                            win.append((st, a128, w128))
                            while len(win) < obs_steps:
                                win.appendleft(win[0])
                            if args.intent_override == "zero":
                                element["intent"] = np.zeros(gen.intent_dim, dtype=np.float32)
                            elif args.intent_override == "random":
                                element["intent"] = np.random.randn(gen.intent_dim).astype(np.float32)
                            else:
                                sw = np.stack([w[0] for w in win])[None]      # (1, To, 15)
                                iw = {
                                    "agentview_rgb":  np.stack([w[1] for w in win])[None],
                                    "eye_in_hand_rgb": np.stack([w[2] for w in win])[None],
                                }
                                element["intent"] = gen.intent_for_windows(sw, iw)[0].astype(np.float32)
                        action_chunk = np.asarray(policy.infer(element)["actions"])
                        plan.extend(action_chunk[: args.replan_steps])
                    obs, _, done, _ = env.step(plan.popleft().tolist())
                    if gen is not None:
                        win.append(_mip_obs(obs))
                    if done:
                        break

                t_ep   += 1
                t_succ += int(done)

            sr = t_succ / max(t_ep, 1)
            results[str(delta)][str(task_id)] = {
                "sr": sr, "succ": t_succ, "ep": t_ep,
                "task": task.language, "target_obj": obj_name, "delta": delta,
            }
            print(f"[task {task_id}] δ={delta:.2f}  SR={sr:.2f} ({t_succ}/{t_ep})  "
                  f"obj={obj_name} :: {task.language}", flush=True)

        env.close()

    # Aggregate: mean SR across tasks at each δ
    for d in deltas:
        task_srs = [v["sr"] for v in results[str(d)].values()]
        results[str(d)]["_mean_sr"] = float(np.mean(task_srs)) if task_srs else float("nan")

    out_data = {
        "config":               args.config_name,
        "checkpoint":           args.checkpoint_dir,
        "intent":               args.intent,
        "task_suite":           args.task_suite,
        "num_trials_per_delta": n,
        "deltas":               deltas,
        "results":              results,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"\n=== {args.config_name}  intent={args.intent} ===")
    print(f"{'delta':>8}  {'mean_SR':>8}  tasks")
    for d in deltas:
        mean_sr = results[str(d)].get("_mean_sr", float("nan"))
        per_task = "  ".join(
            f"{tid}:{v['sr']:.2f}"
            for tid, v in results[str(d)].items()
            if not tid.startswith("_")
        )
        print(f"{d:8.2f}  {mean_sr:8.3f}  {per_task}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
