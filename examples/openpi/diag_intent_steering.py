"""Diagnostic #2 — does the adaRMS intent channel have causal authority over the reach?

Every OOD failure is a no_reach (the gripper goes to the wrong place). For intent to ever fix
that, injecting a different intent must *move where the gripper goes*. This probe tests that
causal link directly, decoupled from whether the flow map is any good:

  - Object is held at its TRUE position P0 (no perturbation, vision is correct).
  - We inject a frozen DECOY intent = the flow map's intent for a scene where the object sits at
    P0 + m·û (shifted by m metres in direction û).  The policy still SEES the object at P0.
  - We measure how far the gripper's closest-approach point drifts along û.

Interpretation (reported alongside the injected-signal magnitude ‖z_decoy − z_true‖):
  - drift ≈ 0  while ‖Δz‖ large  ⇒ channel is INERT (policy ignores intent, rides vision).
  - drift > 0 scaling with û     ⇒ channel STEERS reach (intent has authority; flow map is the
                                   only thing to fix).
  - ‖Δz‖ ≈ 0                     ⇒ flow map collapsed (can't even build a distinct decoy) —
                                   cross-check with diag_flow_collapse.

A control condition injects z_true (Δz=0) to baseline any directional bias.

Usage:
  python examples/openpi/diag_intent_steering.py \
    --config-name pi05_base_intent \
    --checkpoint-dir /data/.../base_intent_goal_run1/29999 \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt /data/.../slot_intent/models/model_best.pt \
    --out logs/diag_intent_steering.json
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

ENV_RES = 256
MAX_STEPS = 300
NUM_STEPS_WAIT = 10
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
TASK_TARGET = {1: ("bowl", 9), 8: ("bowl", 9), 3: ("bowl", 9), 4: ("bowl", 9), 6: ("cream_cheese", 16)}


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


def _mip_obs(obs):
    st = np.concatenate([obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                         obs["robot0_gripper_qpos"], obs["robot0_joint_pos"]]).astype(np.float32)
    agv = _resize(obs["agentview_image"], 128).transpose(2, 0, 1)
    wr  = _resize(obs["robot0_eye_in_hand_image"], 128).transpose(2, 0, 1)
    return st, agv, wr


def _object_xyz(env, qpos_start):
    flat = np.asarray(env.get_sim_state(), dtype=np.float64)
    return flat[qpos_start:qpos_start + 3].copy()


def _intent_for_scene(gen, env, init_state):
    """Flow-map intent for the first-frame window of a settled scene."""
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    st, agv, wr = _mip_obs(obs)
    win = [(st, agv, wr)] * gen.obs_steps
    sw = np.stack([w[0] for w in win])[None]
    iw = {gen.image_keys[0]: np.stack([w[1] for w in win])[None],
          gen.image_keys[1]: np.stack([w[2] for w in win])[None]}
    return gen.intent_for_windows(sw, iw)[0].astype(np.float32)


def _rollout_with_intent(policy, env, init_state, qpos_start, intent_vec, task_language, replan_steps):
    """Run the object-at-P0 episode with a FROZEN injected intent; return (approach_xy, obj_xy, done)."""
    env.reset()
    obs = env.set_init_state(init_state)
    plan = collections.deque()
    done = False
    obj_xy = None
    min_d, approach_xy = float("inf"), None
    for t in range(MAX_STEPS + NUM_STEPS_WAIT):
        if t < NUM_STEPS_WAIT:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            continue
        if obj_xy is None:
            obj_xy = _object_xyz(env, qpos_start)[:2]
        if not plan:
            element = {
                "observation/image":       obs["agentview_image"],
                "observation/wrist_image": obs["robot0_eye_in_hand_image"],
                "observation/state": np.concatenate([
                    obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"]]).astype(np.float32),
                "prompt": str(task_language),
                "intent": intent_vec,                 # FROZEN injected intent
            }
            action_chunk = np.asarray(policy.infer(element)["actions"])
            plan.extend(action_chunk[:replan_steps])
        obs, _, done, _ = env.step(plan.popleft().tolist())
        eef_xy = np.asarray(obs["robot0_eef_pos"][:2], dtype=np.float64)
        d = float(np.linalg.norm(eef_xy - obj_xy))
        if d < min_d:
            min_d, approach_xy = d, eef_xy
        if done:
            break
    return approach_xy, obj_xy, bool(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="pi05_base_intent")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--intent-task-config", default="libero_goal_suite_image_slot_intent")
    ap.add_argument("--intent-ckpt", required=True)
    ap.add_argument("--config-dir", default="examples/configs")
    ap.add_argument("--task-suite", default="libero_goal")
    ap.add_argument("--tasks", default="1,8")
    ap.add_argument("--magnitude", type=float, default=0.10, help="decoy object shift (m)")
    ap.add_argument("--n-dirs", type=int, default=4, help="decoy directions, evenly spaced")
    ap.add_argument("--inits-per-dir", type=int, default=5)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="logs/diag_intent_steering.json")
    args = ap.parse_args()

    sys.path.insert(0, "external/openpi/src")
    import openpi.training.config as _config
    from openpi.policies import policy_config as _policy_config
    from mip.pi05_intent import IntentGenerator

    np.random.seed(args.seed)
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint_dir)
    gen = IntentGenerator(args.intent_task_config, args.intent_ckpt, args.config_dir, device=args.device)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    dirs = np.linspace(0.0, 2 * math.pi, args.n_dirs, endpoint=False)

    rows = []   # one per (task, dir, init): {dz, drift_decoy, drift_ctrl, succ_decoy, succ_ctrl}
    for task_id in [int(x) for x in args.tasks.split(",")]:
        task = suite.get_task(task_id)
        _, qpos_start = TASK_TARGET[task_id]
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RES, camera_widths=ENV_RES)
        try:
            env.seed(args.seed)
        except (TypeError, AttributeError):
            pass

        for di, ang in enumerate(dirs):
            u = np.array([math.cos(ang), math.sin(ang)])
            for i in range(min(args.inits_per_dir, len(init_states))):
                base = init_states[i]
                z_true = _intent_for_scene(gen, env, base)
                s_decoy = base.copy()
                s_decoy[qpos_start]     += args.magnitude * u[0]
                s_decoy[qpos_start + 1] += args.magnitude * u[1]
                z_decoy = _intent_for_scene(gen, env, s_decoy)
                dz = float(np.linalg.norm(z_decoy - z_true))

                ap_d, obj_d, sd = _rollout_with_intent(policy, env, base, qpos_start, z_decoy,
                                                       task.language, args.replan_steps)
                ap_c, obj_c, sc = _rollout_with_intent(policy, env, base, qpos_start, z_true,
                                                       task.language, args.replan_steps)
                drift_d = float(np.dot(ap_d - obj_d, u)) if ap_d is not None else float("nan")
                drift_c = float(np.dot(ap_c - obj_c, u)) if ap_c is not None else float("nan")
                rows.append(dict(task=task_id, dir=di, dz=dz, drift_decoy=drift_d, drift_ctrl=drift_c,
                                 succ_decoy=int(sd), succ_ctrl=int(sc)))
            print(f"[task {task_id}] dir {di} ({math.degrees(ang):.0f}°) done", flush=True)
        env.close()

    dz = np.array([r["dz"] for r in rows])
    dd = np.array([r["drift_decoy"] for r in rows])
    dc = np.array([r["drift_ctrl"] for r in rows])
    steer = dd - dc                                   # drift attributable to the decoy signal
    report = {
        "magnitude": args.magnitude, "n_rows": len(rows),
        "mean_injected_dz": float(np.nanmean(dz)),
        "mean_drift_decoy": float(np.nanmean(dd)),
        "mean_drift_control": float(np.nanmean(dc)),
        "mean_steering": float(np.nanmean(steer)),
        "std_steering": float(np.nanstd(steer)),
        "succ_decoy": float(np.mean([r["succ_decoy"] for r in rows])),
        "succ_control": float(np.mean([r["succ_ctrl"] for r in rows])),
        "rows": rows,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)

    print("\n================= INTENT-STEERING (CAUSAL AUTHORITY) =================")
    print(f"decoy shift = {args.magnitude} m, {len(rows)} probes\n")
    print(f"  mean injected signal  ‖z_decoy−z_true‖ : {report['mean_injected_dz']:.4f}")
    print(f"  mean drift along û  (decoy intent)     : {report['mean_drift_decoy']:+.4f} m")
    print(f"  mean drift along û  (control = z_true)  : {report['mean_drift_control']:+.4f} m")
    print(f"  STEERING  (decoy − control)            : {report['mean_steering']:+.4f} ± {report['std_steering']:.4f} m")
    print(f"  success rate  decoy={report['succ_decoy']:.2f}  control={report['succ_control']:.2f}")
    print("\n  read: steering≈0 with large injected signal ⇒ channel INERT;")
    print("        steering>0 (toward û) ⇒ channel STEERS reach; injected≈0 ⇒ flow map collapsed.")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
