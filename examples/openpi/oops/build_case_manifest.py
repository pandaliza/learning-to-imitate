"""Policy-free physical-case manifest + validity pre-pass for the spatial-robustness study.

Builds every physical case (task x benchmark reset x delta x direction) BEFORE any policy runs, stores the
exact initialized simulator state and its hash, and assigns a validity status from simulator physics only
(rules in oops_common.VALIDITY_RULES, frozen as VALIDITY_VERSION). Both arms are later started from these
stored states, so validity can never depend on a policy outcome.

Per case it records:
  - requested shift (world XY, metres, free-joint translation; z and orientation preserved)
  - deepest contact penetration of the moved subtree at placement (before any physics step)
  - post-native-settle (NUM_STEPS_WAIT dummy steps, identical to the rollout) position, XY drift from the
    requested position, Z change vs the same reset's unperturbed settle, displacement of every OTHER free
    object, whether the env success predicate is already true
  - an extra-settle stability probe (+EXTRA_SETTLE_STEPS dummy steps; manifest-only)
  - distance from the robot base (recorded only; reachability is never used to declare invalidity)

Outputs (in --out-dir):
  cases.jsonl          one record per physical case (validity, metrics, hashes)
  init_states.npz      states[i] = exact flat init state of cases.jsonl line i (float64)
  task_manifest.csv    per-task row in the spec's schema
  manifest_meta.json   rules, versions, physics stack, args

Run (CPU is enough; no renderer needed):
  python examples/openpi/oops/build_case_manifest.py --out-dir oops_runs/manifest_pv1
"""

import argparse
import csv
import functools
import json
import math
import os
import pathlib
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import oops_common as oc  # noqa: E402


def import_libero():
    """robosuite>=1.5 needs the repo's LIBERO compat shim; robosuite 1.4 must NOT get it."""
    import robosuite
    if tuple(int(x) for x in robosuite.__version__.split(".")[:2]) >= (1, 5):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
        import mip.envs.libero._robosuite_compat  # noqa: F401
    import torch
    torch.load = functools.partial(torch.load, weights_only=False)  # LIBERO *.pruned_init are pickles
    from libero.libero import benchmark, get_libero_path
    return benchmark, get_libero_path


def physics_stack():
    import mujoco
    import robosuite
    import libero
    return {"robosuite": robosuite.__version__, "mujoco": mujoco.__version__,
            "libero_path": str(pathlib.Path(libero.__file__).resolve().parent),
            "python": platform.python_version(), "host": platform.node()}


def resolve_task_ids(suite):
    ids = {}
    for tid in range(suite.n_tasks):
        ids[str(suite.get_task(tid).language).strip().lower()] = tid
    return ids


def subtree_bodies(model, root_body):
    out = {root_body}
    changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if b not in out and model.body_parentid[b] in out:
                out.add(b)
                changed = True
    return out


def free_joints(model, exclude):
    out = []
    for j in range(model.njnt):
        name = model.joint_id2name(j)
        if model.jnt_type[j] == 0 and name != exclude:
            out.append(name)
    return out


def max_penetration(sim, moved_bodies):
    """Deepest penetration (positive metres) between the moved subtree and anything else, + that body."""
    m, d = sim.model, sim.data
    worst, other = 0.0, ""
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        in1, in2 = b1 in moved_bodies, b2 in moved_bodies
        if in1 == in2:
            continue
        pen = -float(c.dist)
        if pen > worst:
            worst = pen
            ob = b2 if in1 else b1
            other = m.body_id2name(int(m.body_rootid[ob])) or m.body_id2name(ob) or str(ob)
    return worst, other


def table_footprint(sim):
    """(xmin, xmax, ymin, ymax) of the largest collision box on a body named *table*."""
    m, d = sim.model, sim.data
    best = None
    for g in range(m.ngeom):
        b = m.body_id2name(int(m.geom_bodyid[g])) or ""
        if "table" in b and m.geom_type[g] == 6 and m.geom_contype[g] != 0:
            sx, sy = float(m.geom_size[g][0]), float(m.geom_size[g][1])
            if best is None or sx * sy > best[0]:
                cx, cy = float(d.geom_xpos[g][0]), float(d.geom_xpos[g][1])
                best = (sx * sy, (cx - sx, cx + sx, cy - sy, cy + sy))
    return None if best is None else best[1]


def classify(metrics, rules, is_zero):
    """Status + reason codes from pre-policy physics only. invalid beats unknown beats valid."""
    inv, unk = [], []
    if metrics.get("sim_exception"):
        return "unknown", ["sim_exception"]
    if metrics["pre_satisfied"]:
        inv.append("pre_satisfied")
    if not metrics["on_table_footprint"]:
        inv.append("off_table")
    pen = metrics["placement_penetration_m"]
    if pen > rules["penetration_invalid_m"]:
        inv.append("interpenetration")
    elif pen > rules["penetration_unknown_m"]:
        unk.append("penetration_borderline")
    if not is_zero:
        xy = metrics["xy_drift_m"]
        if xy > rules["xy_drift_invalid_m"]:
            inv.append("displaced_by_contact")
        elif xy > rules["xy_drift_unknown_m"]:
            unk.append("xy_drift_borderline")
        dz = metrics["z_change_m"]
        if dz < -rules["z_change_invalid_m"]:
            inv.append("fell")
        elif dz > rules["z_change_invalid_m"]:
            inv.append("support_changed_elevated")
        elif abs(dz) > rules["z_change_unknown_m"]:
            unk.append("z_change_borderline")
        od = metrics["other_disturb_max_m"]
        if od > rules["other_disturb_invalid_m"]:
            inv.append("scene_disturbed")
        elif od > rules["other_disturb_unknown_m"]:
            unk.append("scene_disturb_borderline")
    if metrics["post_settle_motion_m"] > rules["unstable_m"]:
        inv.append("unstable")
    if inv:
        return "invalid", inv + unk
    if unk:
        return "unknown", unk
    return "valid", ["ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", default=None, help="'|'-separated instructions (default: all eligible)")
    ap.add_argument("--reset-ids", default=",".join(map(str, oc.DEFAULT_RESET_IDS)))
    ap.add_argument("--deltas", default=",".join(map(str, oc.DEFAULT_DELTAS)))
    ap.add_argument("--n-angles", type=int, default=oc.DEFAULT_N_ANGLES)
    ap.add_argument("--render-env", action="store_true",
                    help="build OffScreenRenderEnv (exactly the rollout env class) instead of a renderer-free ControlEnv")
    args = ap.parse_args()

    benchmark, get_libero_path = import_libero()
    from libero.libero.envs.env_wrapper import ControlEnv
    suite = benchmark.get_benchmark_dict()[oc.SUITE]()
    tid_by_instr = resolve_task_ids(suite)
    instructions = args.tasks.split("|") if args.tasks else oc.ELIGIBLE_INSTRUCTIONS
    reset_ids = [int(x) for x in args.reset_ids.split(",")]
    deltas = [float(x) for x in args.deltas.split(",")]
    rules = oc.VALIDITY_RULES
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records, states, task_rows = [], [], []
    t0 = time.time()
    for instr in oc.TASKS:
        instruction, joint, elig_reason, constraints = instr
        tid = tid_by_instr.get(instruction)
        if tid is None:
            raise RuntimeError(f"instruction not found in {oc.SUITE}: {instruction!r}")
        task = suite.get_task(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        selected = instruction in instructions and joint is not None
        row = {"task_name": oc.slug(instruction), "instruction": instruction, "suite_task_id": tid,
               "manipulated_body": "", "moved_subtree": "", "source_region": "", "destination_region": "",
               "fixed_goal_geometry": "", "perturb_frame": "", "eligibility_reason": elig_reason,
               "known_scene_constraints": constraints, "bddl_file": task.bddl_file}
        if args.render_env:
            from libero.libero.envs import OffScreenRenderEnv
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=oc.ENV_RES, camera_widths=oc.ENV_RES)
        else:
            env = ControlEnv(bddl_file_name=str(bddl), has_renderer=False, has_offscreen_renderer=False,
                             use_camera_obs=False)
        env.reset()
        sim = env.env.sim
        m = sim.model
        pp = env.env.parsed_problem
        goal = pp.get("goal_state", [])
        row["destination_region"] = json.dumps(goal)
        row["fixed_goal_geometry"] = ";".join(str(g[-1]) for g in goal)
        if joint is None:
            task_rows.append(row)
            env.close()
            continue
        obj = joint.replace("_joint0", "")
        jid = m.joint_name2id(joint)
        body = int(m.jnt_bodyid[jid])
        moved = subtree_bodies(m, body)
        row["manipulated_body"] = m.body_id2name(body)
        row["moved_subtree"] = ";".join(sorted(m.body_id2name(b) for b in moved))
        row["source_region"] = ";".join(x[-1] for x in pp.get("initial_state", []) if len(x) == 3 and x[1] == obj)
        row["perturb_frame"] = ("world XY translation of the free joint (flattened sim state [time,qpos,qvel], "
                                "index = jnt_qposadr+1); z and quaternion unchanged; applied to the stored init "
                                "state before set_init_state; then the native 10-step dummy-action settle")
        # goal must not reference the moved object's ORIGINAL location (it is relational in libero_goal)
        row["goal_references_moved_object_origin"] = any(obj in str(g) and "main_table_" + obj in str(g) for g in goal)
        task_rows.append(row)
        if not selected:
            env.close()
            continue

        xyz_idx = oc.flat_xyz_index(sim, joint)
        others = {j: oc.flat_xyz_index(sim, j) for j in free_joints(m, joint)}
        robot_xy = np.asarray(sim.data.get_body_xpos("robot0_base")[:2], dtype=np.float64).copy()
        foot = table_footprint(sim)
        init_states = suite.get_task_init_states(tid)
        print(f"[{tid}] {instruction}  moved={joint} flat_idx={xyz_idx}  table={foot}", flush=True)

        def run_case(state):
            """set_init_state -> placement contacts -> native settle -> extra settle. Physics only."""
            env.reset()  # robosuite hard reset rebuilds MjSim: never reuse a sim handle across resets
            env.set_init_state(state)
            pen, pen_body = max_penetration(env.env.sim, moved)
            done = False
            for _ in range(oc.NUM_STEPS_WAIT):
                _, _, done, _ = env.step(oc.LIBERO_DUMMY_ACTION)
            flat = np.asarray(env.get_sim_state(), dtype=np.float64)
            settled = flat[xyz_idx:xyz_idx + 3].copy()
            other_xyz = {j: flat[i:i + 3].copy() for j, i in others.items()}
            pre_sat = bool(done) or bool(env.env._check_success())
            s_ = env.env.sim
            vel = float(np.linalg.norm(s_.data.qvel[s_.model.jnt_dofadr[jid]:s_.model.jnt_dofadr[jid] + 3]))
            settled_hash = oc.state_hash(flat)
            for _ in range(oc.EXTRA_SETTLE_STEPS):
                env.step(oc.LIBERO_DUMMY_ACTION)
            later = np.asarray(env.get_sim_state(), dtype=np.float64)[xyz_idx:xyz_idx + 3]
            return {"pen": pen, "pen_body": pen_body, "settled": settled, "other": other_xyz,
                    "pre_sat": pre_sat, "motion": float(np.linalg.norm(later - settled)), "vel": vel,
                    "settled_hash": settled_hash}

        ref = {}
        for (r, delta, a_idx, a_rad) in oc.physical_cases(reset_ids, deltas, args.n_angles):
            base = np.asarray(init_states[r], dtype=np.float64)
            dx, dy = oc.displacement(delta, a_rad)
            state = oc.perturb_state(base, xyz_idx, dx, dy) if delta > 0 else base.copy()
            h = oc.state_hash(state)
            rec = {"base_state_hash": oc.state_hash(base),"case_id": oc.case_id(instruction, r, delta, a_idx, h), "task_name": oc.slug(instruction),
                   "instruction": instruction, "suite_task_id": tid, "reset_id": r, "delta_m": delta,
                   "angle_idx": a_idx, "angle_rad": None if math.isnan(a_rad) else a_rad,
                   "dx_m": dx, "dy_m": dy, "moved_joint": joint, "moved_body": row["manipulated_body"],
                   "flat_xyz_index": xyz_idx, "initialized_state_hash": h, "config_version": oc.CONFIG_VERSION,
                   "validity_version": oc.VALIDITY_VERSION,
                   "original_xyz": base[xyz_idx:xyz_idx + 3].tolist(),
                   "initialized_xyz": state[xyz_idx:xyz_idx + 3].tolist()}
            metrics = {}
            try:
                c = run_case(state)
                if delta == 0.0:
                    ref[r] = c
                rr = ref.get(r)
                assert rr is not None, "delta=0 must be processed before delta>0 for each reset"
                req_xy = rr["settled"][:2] + np.array([dx, dy])
                other_disp = max([float(np.linalg.norm(c["other"][j][:2] - rr["other"][j][:2])) for j in others] or [0.0])
                metrics = {
                    "placement_penetration_m": c["pen"], "placement_penetration_body": c["pen_body"],
                    "settled_xyz": c["settled"].tolist(), "requested_xy": req_xy.tolist(),
                    "xy_drift_m": float(np.linalg.norm(c["settled"][:2] - req_xy)),
                    "z_change_m": float(c["settled"][2] - rr["settled"][2]),
                    "other_disturb_max_m": other_disp,
                    "pre_satisfied": c["pre_sat"], "post_settle_motion_m": c["motion"],
                    "settled_lin_speed_mps": c["vel"],
                    "on_table_footprint": True if foot is None else bool(
                        foot[0] + rules["table_margin_m"] <= req_xy[0] <= foot[1] - rules["table_margin_m"]
                        and foot[2] + rules["table_margin_m"] <= req_xy[1] <= foot[3] - rules["table_margin_m"]),
                    "dist_from_robot_base_m": float(np.linalg.norm(req_xy - robot_xy)),
                    "settled_state_hash": c["settled_hash"],
                }
            except Exception as e:  # simulator exception: separate from physical invalidity
                metrics = {"sim_exception": repr(e)}
            status, reasons = classify(metrics, rules, delta == 0.0)
            rec.update(metrics)
            rec["validity_status"] = status
            rec["validity_reason"] = ";".join(reasons)
            rec["state_row"] = len(states)
            records.append(rec)
            states.append(state)
        n = sum(1 for x in records if x["suite_task_id"] == tid)
        nv = sum(1 for x in records if x["suite_task_id"] == tid and x["validity_status"] == "valid")
        print(f"     {n} cases, {nv} valid  ({time.time() - t0:.0f}s)", flush=True)
        env.close()

    with open(out / "cases.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, default=oc._json_default) + "\n")
    np.savez_compressed(out / "init_states.npz", states=np.stack(states),
                        case_ids=np.array([r["case_id"] for r in records]))
    cols = ["task_name", "instruction", "suite_task_id", "manipulated_body", "moved_subtree", "source_region",
            "destination_region", "fixed_goal_geometry", "perturb_frame", "eligibility_reason",
            "known_scene_constraints", "goal_references_moved_object_origin", "bddl_file"]
    with open(out / "task_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(task_rows)
    meta = {"experiment": oc.EXPERIMENT, "config_version": oc.CONFIG_VERSION, "validity_rules": rules,
            "num_steps_wait": oc.NUM_STEPS_WAIT, "extra_settle_steps": oc.EXTRA_SETTLE_STEPS,
            "reset_ids": reset_ids, "deltas": deltas, "n_angles": args.n_angles,
            "angles_rad": oc.angles(args.n_angles).tolist(), "env_class": "OffScreenRenderEnv" if args.render_env else "ControlEnv",
            "physics_stack": physics_stack(), "created_unix": time.time(), "n_cases": len(records)}
    with open(out / "manifest_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"-> {out}  ({len(records)} cases)")


if __name__ == "__main__":
    main()
