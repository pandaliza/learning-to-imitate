"""Diagnostic #1 — is the slot-intent flow map collapsed?

The override experiment showed flow-intent == zero-intent at eval, implying the deployable
prior p(z|s) (the flow map) emits a near-constant vector regardless of the scene. This script
measures that directly, with NO policy rollouts — just flow-map forward passes on real LIBERO
observations rendered from many init states and object positions.

Two measurements:

  (A) DIVERSITY across distinct scenes (δ=0): if the flow map encodes anything scene-specific,
      its intent vectors should spread out. We report per-dim std, mean L2 norm, mean pairwise
      cosine similarity, and PCA effective rank. cosine≈1 + std≈0 ⇒ collapsed to a constant.

  (B) OBJECT-POSITION SENSITIVITY: for each scene we shift the TARGET object's XY by δ (so the
      rendered image actually shows the object move) and measure ‖z(δ) − z(0)‖. If the intent is
      supposed to localise the object, it must move with it. Δz≈0 ⇒ the prior carries no object-
      position information — which is exactly why intent can't fix the no_reach failures.

Usage:
  python examples/openpi/diag_flow_collapse.py \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt /data/.../slot_intent/models/model_best.pt \
    --out logs/diag_flow_collapse.json
"""

import argparse
import functools
import json
import math
import pathlib
import sys

import numpy as np
import torch

import mip.envs.libero._robosuite_compat  # noqa: F401
torch.load = functools.partial(torch.load, weights_only=False)  # ckpt carries numpy objects

ENV_RES = 256
NUM_STEPS_WAIT = 10
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# Perturbable free-body targets in libero_goal (qpos xyz start in the flat [time,qpos,qvel] state).
TASK_TARGET = {
    1: ("bowl", 9), 2: ("wine_bottle", 23), 3: ("bowl", 9), 4: ("bowl", 9),
    5: ("plate", 30), 6: ("cream_cheese", 16), 8: ("bowl", 9), 9: ("wine_bottle", 23),
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


def _mip_obs(obs):
    st = np.concatenate([obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                         obs["robot0_gripper_qpos"], obs["robot0_joint_pos"]]).astype(np.float32)
    agv = _resize(obs["agentview_image"], 128).transpose(2, 0, 1)
    wr  = _resize(obs["robot0_eye_in_hand_image"], 128).transpose(2, 0, 1)
    return st, agv, wr


def _intent_at(gen, env, init_state):
    """Settle the env at init_state, then return the flow-map intent for that first-frame window."""
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    st, agv, wr = _mip_obs(obs)
    win = [(st, agv, wr)] * gen.obs_steps          # first-frame window (edge-padded), as eval does
    sw = np.stack([w[0] for w in win])[None]         # (1, To, 15)
    iw = {gen.image_keys[0]: np.stack([w[1] for w in win])[None],
          gen.image_keys[1]: np.stack([w[2] for w in win])[None]}
    return gen.intent_for_windows(sw, iw)[0].astype(np.float64)


def _summ(Z):
    """Spread statistics for a stack of intent vectors Z (M, D)."""
    Z = np.asarray(Z, dtype=np.float64)
    mu = Z.mean(0)
    per_dim_std = Z.std(0)
    norms = np.linalg.norm(Z, axis=1)
    # mean pairwise cosine similarity
    Zn = Z / (norms[:, None] + 1e-9)
    cos = Zn @ Zn.T
    iu = np.triu_indices(len(Z), k=1)
    mean_cos = float(cos[iu].mean()) if len(iu[0]) else float("nan")
    # PCA effective rank (participation ratio of eigenvalues of the covariance)
    C = np.cov(Z.T)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    eff_rank = float((ev.sum() ** 2) / (np.square(ev).sum() + 1e-12)) if ev.sum() > 0 else 0.0
    return dict(mean_std=float(per_dim_std.mean()), mean_norm=float(norms.mean()),
                norm_of_mean=float(np.linalg.norm(mu)), mean_pairwise_cos=mean_cos,
                eff_rank=eff_rank, dim=int(Z.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent-task-config", default="libero_goal_suite_image_slot_intent")
    ap.add_argument("--intent-ckpt", required=True)
    ap.add_argument("--config-dir", default="examples/configs")
    ap.add_argument("--task-suite", default="libero_goal")
    ap.add_argument("--tasks", default=None, help="comma-separated task ids; default = perturbable goal tasks")
    ap.add_argument("--inits-per-task", type=int, default=5)
    ap.add_argument("--deltas", default="0.0,0.02,0.05,0.10,0.15")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="logs/diag_flow_collapse.json")
    args = ap.parse_args()

    deltas = [float(d) for d in args.deltas.split(",")]
    np.random.seed(args.seed)

    sys.path.insert(0, "external/openpi/src")
    from mip.pi05_intent import IntentGenerator
    gen = IntentGenerator(args.intent_task_config, args.intent_ckpt, args.config_dir, device=args.device)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[args.task_suite]()

    task_ids = ([int(x) for x in args.tasks.split(",")] if args.tasks
                else [t for t in range(suite.n_tasks) if t in TASK_TARGET])

    z0_all = []                                   # δ=0 intents across all scenes (diversity)
    z0_by_task = {}                               # task -> list of δ=0 intents (intra/inter task)
    sens = {f"{d:.2f}": [] for d in deltas if d > 0}   # δ -> list of ‖z(δ)-z(0)‖
    sens_rel = {f"{d:.2f}": [] for d in deltas if d > 0}

    for task_id in task_ids:
        task = suite.get_task(task_id)
        _, qpos_start = TASK_TARGET[task_id]
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RES, camera_widths=ENV_RES)
        try:
            env.seed(args.seed)
        except (TypeError, AttributeError):
            pass

        z0_by_task[task_id] = []
        for i in range(min(args.inits_per_task, len(init_states))):
            base = init_states[i]
            z0 = _intent_at(gen, env, base)
            z0_all.append(z0); z0_by_task[task_id].append(z0)
            n0 = np.linalg.norm(z0) + 1e-9
            for d in deltas:
                if d <= 0:
                    continue
                ang = 2 * math.pi * (i / max(args.inits_per_task, 1))   # vary direction per init
                s = base.copy()
                s[qpos_start]     += d * math.cos(ang)
                s[qpos_start + 1] += d * math.sin(ang)
                zd = _intent_at(gen, env, s)
                dz = float(np.linalg.norm(zd - z0))
                sens[f"{d:.2f}"].append(dz)
                sens_rel[f"{d:.2f}"].append(dz / n0)
        env.close()
        print(f"[task {task_id}] {len(z0_by_task[task_id])} scenes done :: {task.language}", flush=True)

    # (A) diversity across all δ=0 scenes
    div = _summ(z0_all)
    # intra-task vs inter-task spread: mean within-task pairwise dist vs across-task
    def _mean_pair_dist(vecs):
        vecs = np.asarray(vecs)
        if len(vecs) < 2:
            return float("nan")
        d = np.linalg.norm(vecs[:, None, :] - vecs[None, :, :], axis=-1)
        return float(d[np.triu_indices(len(vecs), 1)].mean())
    intra = float(np.nanmean([_mean_pair_dist(v) for v in z0_by_task.values()]))
    inter = _mean_pair_dist([np.mean(v, 0) for v in z0_by_task.values()])

    report = {
        "n_scenes": len(z0_all),
        "diversity_delta0": div,
        "intra_task_mean_pair_dist": intra,
        "inter_task_mean_pair_dist": inter,
        "object_sensitivity_abs": {k: float(np.mean(v)) for k, v in sens.items() if v},
        "object_sensitivity_rel": {k: float(np.mean(v)) for k, v in sens_rel.items() if v},
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)

    print("\n================= FLOW-MAP COLLAPSE DIAGNOSTIC =================")
    print(f"scenes: {div['dim']}-D intent over {report['n_scenes']} δ=0 scenes\n")
    print("(A) DIVERSITY across scenes (collapsed ⇒ std→0, cos→1, eff_rank→1):")
    print(f"    mean per-dim std   : {div['mean_std']:.4f}")
    print(f"    mean vector norm   : {div['mean_norm']:.4f}")
    print(f"    norm of mean vector: {div['norm_of_mean']:.4f}   (≈ mean_norm ⇒ all point the same way)")
    print(f"    mean pairwise cos  : {div['mean_pairwise_cos']:.4f}")
    print(f"    PCA effective rank : {div['eff_rank']:.2f} / {div['dim']}")
    print(f"    intra-task pair dist: {intra:.4f}   inter-task pair dist: {inter:.4f}")
    print("\n(B) OBJECT-POSITION SENSITIVITY  ‖z(δ)−z(0)‖  (carries object loc ⇒ grows with δ):")
    print(f"    {'δ':>6} {'abs Δz':>9} {'rel Δz':>9}")
    for d in deltas:
        if d <= 0:
            continue
        k = f"{d:.2f}"
        if sens.get(k):
            print(f"    {d:6.2f} {np.mean(sens[k]):9.4f} {np.mean(sens_rel[k]):9.4f}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
