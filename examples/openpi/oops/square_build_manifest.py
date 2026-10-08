"""Policy-free physical-case manifest + validity for robomimic NutAssemblySquare (square-mh-state).

Same design as build_case_manifest.py (LIBERO), adapted to robomimic:
  - reset_id k = the native robomimic reset drawn after np.random.seed(SEED0 + k) (SEED0 = 42, the native
    eval seed); its full flattened state is stored, so both arms start from the identical state
  - the moved body is the square nut (free joint SquareNut_joint0); world-XY translation of the stored state,
    z and orientation preserved; delta=0 once per reset; 10 fixed directions per nonzero delta
  - the native robomimic rollout has NO settle steps. Validity is therefore decided by a diagnostic probe that is
    never part of a rollout: the robot is held kinematically (its joint qpos/qvel re-imposed every substep) while
    the scene evolves for PROBE_SUBSTEPS, then for another PROBE_SUBSTEPS to test stability
  - rules = oops_common.VALIDITY_RULES thresholds (version tagged "<vv>-square-probe")
  - per-case covariate: distance of the requested nut XY outside the bounding box of the demonstrated start
    positions (square/mh low_dim.hdf5, all 300 demos) -- decided before any policy runs

  python examples/openpi/oops/square_build_manifest.py --out-dir oops_runs/square_manifest_pv1
"""

import argparse
import json
import math
import os
import pathlib
import platform
import sys
import time
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import oops_common as oc  # noqa: E402
from build_case_manifest import classify, max_penetration, subtree_bodies, table_footprint  # noqa: E402

TASK = "square-mh-state: place the square nut on the square peg"
MOVED_JOINT = "SquareNut_joint0"
SEED0 = 42
PROBE_SUBSTEPS = 250  # 0.5 s of physics at the default 2 ms timestep


def load_native(overrides):
    """Native config/env/dataset helpers from examples/collect/collect_diversity_rollouts.py.

    That module (and eval_success_rate.py) still compute ROOT as if they lived in examples/; patch it to the repo
    root, and stub the LIBERO env import (not installed here; unused by robomimic)."""
    sys.path.insert(0, str(REPO / "examples/collect"))
    sys.path.insert(0, str(REPO / "examples"))
    sys.path.insert(0, str(REPO))
    if "mip.envs.libero" not in sys.modules:
        try:
            import mip.envs.libero  # noqa: F401
        except Exception:
            stub = types.ModuleType("mip.envs.libero")
            stub.make_vec_env = None
            sys.modules["mip.envs.libero"] = stub
    cwd = os.getcwd()
    real_chdir = os.chdir
    os.chdir = lambda *_a, **_k: None
    try:
        import collect_diversity_rollouts as cdr
    finally:
        os.chdir = real_chdir
        os.chdir(cwd)
    cdr.ROOT = REPO
    return cdr, cdr.load_config(overrides)


def make_single_env(task_cfg):
    from mip.envs.robomimic.robomimic_env import make_robomimic_env
    import io
    out = sys.stdout
    sys.stdout = io.StringIO()
    try:
        env = make_robomimic_env(task_cfg, 0, False, seed=None)()
    finally:
        sys.stdout = out
    lowdim = env.env.env          # MultiStepWrapper -> VideoRecordingWrapper -> RobomimicLowdimWrapper
    return env, lowdim, lowdim.env  # robomimic EnvRobosuite


def native_reset_state(robo_env, reset_id):
    """Exactly RobomimicLowdimWrapper's seeded reset: np.random.seed(seed); env.reset(); get_state()."""
    np.random.seed(SEED0 + reset_id)
    robo_env.reset()
    return np.asarray(robo_env.get_state()["states"], dtype=np.float64).copy()


def demo_start_box(dataset_path, xyz_idx):
    import h5py
    with h5py.File(dataset_path) as h:
        s0 = np.stack([h["data"][k]["states"][0] for k in h["data"].keys()])
    x, y = s0[:, xyz_idx], s0[:, xyz_idx + 1]
    return {"x": [float(x.min()), float(x.max())], "y": [float(y.min()), float(y.max())],
            "x_std": float(x.std()), "y_std": float(y.std()), "n_demos": int(len(s0))}


def dist_outside_box(xy, box):
    dx = max(box["x"][0] - xy[0], 0.0, xy[0] - box["x"][1])
    dy = max(box["y"][0] - xy[1], 0.0, xy[1] - box["y"][1])
    return float(math.hypot(dx, dy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-resets", type=int, default=50)
    ap.add_argument("--deltas", default="0.0,0.02,0.05,0.10")
    ap.add_argument("--n-angles", type=int, default=oc.DEFAULT_N_ANGLES)
    args = ap.parse_args()

    cdr, cfg = load_native(["task=square_mh_state", "task.num_envs=1"])
    env, lowdim, robo = make_single_env(cfg.task)
    rs = robo.env  # robosuite NutAssemblySquare
    from huggingface_hub import hf_hub_download
    dataset_path = hf_hub_download(repo_id=cfg.task.dataset_repo, filename=cfg.task.dataset_filename,
                                   repo_type="dataset")
    deltas = [float(x) for x in args.deltas.split(",")]
    rules = dict(oc.VALIDITY_RULES)
    rules["version"] = oc.VALIDITY_VERSION + "-square-probe"
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def sim():
        return rs.sim  # robosuite may rebuild the sim on hard reset: never cache it

    xyz_idx = oc.flat_xyz_index(sim(), MOVED_JOINT)
    m0 = sim().model
    nut_body = int(m0.jnt_bodyid[m0.joint_name2id(MOVED_JOINT)])
    moved = subtree_bodies(m0, nut_body)
    robot_q = [int(m0.jnt_qposadr[j]) for j in range(m0.njnt)
               if (m0.joint_id2name(j) or "").startswith(("robot0_", "gripper0_"))]
    robot_v = [int(m0.jnt_dofadr[j]) for j in range(m0.njnt)
               if (m0.joint_id2name(j) or "").startswith(("robot0_", "gripper0_"))]
    box = demo_start_box(dataset_path, xyz_idx)
    print(f"moved={MOVED_JOINT} flat_idx={xyz_idx} body={m0.body_id2name(nut_body)} demo box={box}", flush=True)

    def place(state):
        robo.reset()                       # fresh controller/sim, then the exact stored state
        robo.reset_to({"states": state})
        return sim()

    def probe(state):
        s = place(state)
        pen, pen_body = max_penetration(s, moved)
        q0 = s.data.qpos[robot_q].copy()

        def run(n):
            for _ in range(n):
                s.data.qpos[robot_q] = q0
                s.data.qvel[robot_v] = 0.0
                s.step()
            s.forward()
            return np.asarray(s.get_state().flatten(), dtype=np.float64)
        f1 = run(PROBE_SUBSTEPS)
        settled = f1[xyz_idx:xyz_idx + 3].copy()
        pre = bool(rs._check_success())
        f2 = run(PROBE_SUBSTEPS)
        return {"pen": pen, "pen_body": pen_body, "settled": settled, "pre_sat": pre,
                "motion": float(np.linalg.norm(f2[xyz_idx:xyz_idx + 3] - settled)), "settled_hash": oc.state_hash(f1)}

    foot = table_footprint(place(native_reset_state(robo, 0)))
    robot_xy = np.asarray(sim().data.get_body_xpos("robot0_base")[:2], dtype=np.float64).copy()
    records, states = [], []
    t0 = time.time()
    ref, bases = {}, {}
    for (r, delta, a_idx, a_rad) in oc.physical_cases(range(args.n_resets), deltas, args.n_angles):
        if r not in bases:
            bases[r] = native_reset_state(robo, r)
        base = bases[r]
        dx, dy = oc.displacement(delta, a_rad)
        state = oc.perturb_state(base, xyz_idx, dx, dy) if delta > 0 else base.copy()
        h = oc.state_hash(state)
        rec = {"case_id": oc.case_id("square_mh_state", r, delta, a_idx, h), "task_name": "square_mh_state",
               "instruction": TASK, "reset_id": r, "native_env_seed": SEED0 + r, "delta_m": delta,
               "angle_idx": a_idx, "angle_rad": None if math.isnan(a_rad) else a_rad, "dx_m": dx, "dy_m": dy,
               "moved_joint": MOVED_JOINT, "moved_body": m0.body_id2name(nut_body), "flat_xyz_index": xyz_idx,
               "base_state_hash": oc.state_hash(base), "initialized_state_hash": h,
               "config_version": oc.CONFIG_VERSION, "validity_version": rules["version"],
               "original_xyz": base[xyz_idx:xyz_idx + 3].tolist(), "initialized_xyz": state[xyz_idx:xyz_idx + 3].tolist(),
               "dist_outside_demo_box_m": dist_outside_box(state[xyz_idx:xyz_idx + 2], box)}
        try:
            c = probe(state)
            if delta == 0.0:
                ref[r] = c
            rr = ref[r]
            req_xy = rr["settled"][:2] + np.array([dx, dy])
            metrics = {
                "placement_penetration_m": c["pen"], "placement_penetration_body": c["pen_body"],
                "settled_xyz": c["settled"].tolist(), "requested_xy": req_xy.tolist(),
                "xy_drift_m": float(np.linalg.norm(c["settled"][:2] - req_xy)),
                "z_change_m": float(c["settled"][2] - rr["settled"][2]),
                "other_disturb_max_m": 0.0,  # no other object in the scene (the round nut is parked off-table)
                "pre_satisfied": c["pre_sat"], "post_settle_motion_m": c["motion"],
                "on_table_footprint": True if foot is None else bool(foot[0] <= req_xy[0] <= foot[1]
                                                                    and foot[2] <= req_xy[1] <= foot[3]),
                "dist_from_robot_base_m": float(np.linalg.norm(req_xy - robot_xy)),
                "settled_state_hash": c["settled_hash"],
            }
        except Exception as e:
            metrics = {"sim_exception": repr(e)}
        status, reasons = classify(metrics, rules, delta == 0.0)
        rec.update(metrics)
        rec["validity_status"], rec["validity_reason"] = status, ";".join(reasons)
        rec["state_row"] = len(states)
        records.append(rec)
        states.append(state)
        if len(records) % 100 == 0:
            print(f"  {len(records)} cases ({time.time() - t0:.0f}s)", flush=True)

    with open(out / "cases.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, default=oc._json_default) + "\n")
    np.savez_compressed(out / "init_states.npz", states=np.stack(states),
                        case_ids=np.array([r["case_id"] for r in records]))
    import mujoco
    import robosuite
    meta = {"experiment": oc.EXPERIMENT + "/square", "config_version": oc.CONFIG_VERSION, "validity_rules": rules,
            "probe": f"robot held kinematically; {PROBE_SUBSTEPS} substeps settle + {PROBE_SUBSTEPS} stability",
            "native_settle_steps": 0, "seed0": SEED0, "n_resets": args.n_resets, "deltas": deltas,
            "primary_deltas": [0.0, 0.05, 0.10], "supplementary_deltas": [d for d in deltas if d not in (0.0, 0.05, 0.10)],
            "n_angles": args.n_angles, "angles_rad": oc.angles(args.n_angles).tolist(), "demo_start_box": box,
            "table_footprint": foot, "dataset": dataset_path, "task_config": "square_mh_state",
            "physics_stack": {"robosuite": robosuite.__version__, "mujoco": mujoco.__version__,
                              "python": platform.python_version(), "host": platform.node()},
            "created_unix": time.time(), "n_cases": len(records)}
    (out / "manifest_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"-> {out} ({len(records)} cases, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
