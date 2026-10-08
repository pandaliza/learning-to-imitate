"""Restore-through-damage verification (spec §11), physics path (no renderer, CPU only).

Cases are damaging teleop demos replayed open loop (the open-loop replays in runs/replay/open_teleop reproduce their
damage). For each case: rebuild the demo episode (official playback setup: ep_meta, reset, reset_from_xml_string,
states[0]), step actions to t_b = first_damage_step - LEAD (pre-damage), capture S, continue through the damaging
contact to t_end and record the reference trace R. Then:

  T1 same-instance restore            restore S into the (now damaged) env, replay -> == R
  T2 restore twice                    restore S, run a perturbed branch, restore S again, replay -> == R
  T3 branch-order invariance          branches X (perturbed) and Y (original) in orders XY and YX -> X1==X2, Y1==Y2==R
  T4 fresh env instance               new env object, rebuild episode, restore S, replay -> == R
  T5 disk round trip, fresh process   S saved to disk, new python process rebuilds + loads + replays -> == R
  T6 post-chunk caching               capture S8 after 8 steps from S (via disk), continue -> == R[8:]; repeated twice
  T0 completeness                     whole-graph python digest right after restore == digest at capture
  N1 negative control                 physics-only restore (mj state only) into the damaged env: equal qpos, but
                                      health differs -> the checker must report FAIL
  N2 negative control                 full restore but prev_linear_velocities re-synced to current cvel (playback
                                      convention) -> impact-force trace differs at the first step(s)

Per-step trace: full MuJoCo integration state, every low-dim observation (health, proprio, ...), per-link
damage_info numbers (damage, impact, filtered/unfiltered qs and raw forces, contact count + summed magnitude),
reward, official success, wipe counts, env timestep / cur_time. Comparison is bit-exact (max |diff| reported).

  python test_restore.py --cases pick_egg:without:demo_1,... --out OUT.json
  python test_restore.py --child SNAP.pkl ACTIONS.npy CASE OUT.npz   (internal: T5 child process)
"""

import argparse
import json
import os
import subprocess
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, FULL)
from common import add_oopsie_paths, teleop_path  # noqa: E402
from path_remap import remap_asset_paths  # noqa: E402

add_oopsie_paths()
os.environ.setdefault("PYNPUT_BACKEND", "dummy")
import snapshot_damage_state as SDS  # noqa: E402

PY = sys.executable
REG = {"pick_egg": "pick_egg", "wipe_countertop": "wipe_counter"}
FILE_PATH_ATTRS = (".file",)  # asset file paths of a rebuilt model (temp copies), not episode state
LEAD = 12          # steps between the snapshot and the first damage step of the reference
TAIL = 40          # steps recorded after the first damage step


def make_env_cpu(task):
    from damagesim.robosuite.io_utils import patch_gripper_for_position_control
    patch_gripper_for_position_control()
    from robosuite.controllers import load_composite_controller_config
    from envs.registry import EnvironmentRegistry
    cfg = EnvironmentRegistry.get(REG[task])
    return cfg.damageable_class(
        robots=cfg.robot, controller_configs=load_composite_controller_config(robot=cfg.robot),
        translucent_robot=False, has_renderer=False, has_offscreen_renderer=False,
        render_camera=cfg.camera_name, ignore_done=True, use_camera_obs=False,
        render_segmentation=False, control_freq=cfg.control_freq)


def load_case(task, cond, demo):
    with h5py.File(teleop_path(task, cond), "r") as f:
        g = f["data"][demo]
        return {"ep_meta": remap_asset_paths(g.attrs.get("ep_meta", "{}")),
                "model_file": remap_asset_paths(g.attrs["model_file"]), "states": g["states"][:],
                "actions": g["actions"][:]}


def setup_demo(env, c):
    """Official playback setup (scripts/playback_robocasa.py:174-223 via examples/oopsieverse_full/replay.py)."""
    ep = json.loads(c["ep_meta"])
    env.set_ep_meta(ep) if ep else env.unset_ep_meta()
    env.reset()
    env.reset_from_xml_string(c["model_file"])
    env.sim.reset()
    for robot in env.robots:
        for arm in robot.arms:
            for sid in (robot.eef_site_id[arm], robot.eef_cylinder_id[arm]):
                if sid is not None:
                    env.sim.model.site_rgba[sid] = np.zeros(4)
    env.sim.set_state_from_flattened(c["states"][0])
    env.sim.forward()
    for obj in env.get_damageable_objects():
        for ev in obj.damage_evaluators:
            for p in ev._get_damageable_part_names():
                ev.prev_linear_velocities[p] = ev._get_part_linear_velocity(p)


def step_record(env, a):
    from damagesim.utils.misc_utils import robocasa_check_success
    obs, rew, done, info = env.step(a)
    row = {"mj": SDS.integration_state(env), "reward": np.float64(rew),
           "success": np.bool_(robocasa_check_success(env)), "timestep": np.int64(env.timestep),
           "cur_time": np.float64(env.cur_time),
           "total_wiped": np.int64(info.get("total_wiped", -1))}
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.dtype != object and v.size < 5000:
            row[f"obs/{k}"] = np.asarray(v).copy()
    di = []
    for n in env.health_list_link_names:
        o, l = n.split("@", 1)
        m = info["damage_info"].get(o, {}).get(l, {}).get("mechanical", {})
        cs = m.get("contacts", []) or []
        di.append([m.get(k, np.nan) if m.get(k) is not None else np.nan for k in
                   ("damage", "impact_forces", "filtered_qs_forces", "filtered_raw_sim_forces",
                    "unfiltered_qs_forces", "unfiltered_raw_sim_forces")] +
                  [len(cs), float(sum(c["magnitude"] for c in cs))])
    row["damage_info"] = np.asarray(di, np.float64)
    return row


def run(env, actions):
    rows = [step_record(env, a) for a in actions]
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def compare(A, B):
    """Bit-exact comparison; returns (all_equal, {component: max_abs_diff or 'shape'})."""
    out, eq = {}, True
    for k in sorted(set(A) | set(B)):
        if k not in A or k not in B or A[k].shape != B[k].shape:
            out[k] = "missing/shape"
            eq = False
            continue
        a, b = A[k].astype(np.float64), B[k].astype(np.float64)
        same = np.array_equal(a, b, equal_nan=True)
        d = np.nanmax(np.abs(a - b)) if a.size else 0.0
        out[k] = 0.0 if same else float(d)
        eq &= bool(same)
    return eq, out


def health_check(A, B):
    """Health-specific verdict: 'equal qpos with different health' must count as FAIL."""
    hA, hB = A["obs/health"], B["obs/health"]
    return {"health_equal": bool(np.array_equal(hA, hB)),
            "max_health_diff": float(np.max(np.abs(hA.astype(np.float64) - hB))),
            "damage_events_ref": int((A["damage_info"][:, :, 0] > 0).sum()),
            "damage_events_test": int((B["damage_info"][:, :, 0] > 0).sum()),
            "health_delta_ref": float(100 * hA.shape[1] - hA[-1].astype(np.float64).sum()),
            "health_delta_test": float(100 * hB.shape[1] - hB[-1].astype(np.float64).sum()),
            "success_ref": bool(A["success"][-1]), "success_test": bool(B["success"][-1])}


def first_damage_open_loop(task, cond, demo):
    f = f"/edrive1/lizad/oopsieverse_full_study/runs/replay/open_teleop/{task}__{cond}__{demo}.npz"
    H = np.load(f)["rp_health"]
    idx = np.where((H < 100 - 1e-6).any(1))[0]
    return int(idx[0]) + 1  # replay step i -> after i+1 actions (actions[1..i+1])


def child(snap_path, actions_path, case, out_path):
    task, cond, demo = case.split(":")
    os.chdir("/edrive1/lizad/oopsieverse_full_study/src/oopsieverse")
    env = make_env_cpu(task)
    setup_demo(env, load_case(task, cond, demo))
    snap = SDS.load(snap_path)
    SDS.restore(env, snap)
    np.savez(out_path, **run(env, np.load(actions_path)))


def test_case(task, cond, demo, workdir):
    t0 = time.time()
    res = {"case": f"{task}:{cond}:{demo}"}
    c = load_case(task, cond, demo)
    acts = c["actions"][1:]  # stored actions[t] precede states[t]; replay steps actions[1:]
    fd = first_damage_open_loop(task, cond, demo)
    tb = max(8, fd - LEAD)
    t_end = len(acts)  # through the damaging contact to the end of the demo (success included)
    res.update({"first_damage_step_open_loop": fd, "t_b": tb, "t_end": t_end})
    env = make_env_cpu(task)
    setup_demo(env, c)
    for a in acts[:tb]:
        env.step(a)
    h_at_tb = np.asarray(env.get_observations()[0]["health"], np.float64)
    res["pre_damage_at_t_b"] = bool(np.all(h_at_tb >= 100.0))
    S = SDS.capture(env, episode={"t": tb}, meta={"case": res["case"]})
    G0 = SDS.graph_digest(env)
    seg = acts[tb:t_end]
    R = run(env, seg)
    res["reference"] = health_check(R, R)
    rng = np.random.default_rng(0)
    pert = np.clip(seg + rng.normal(0, 0.3, seg.shape), -1, 1)
    pert[:, -1] = seg[:, -1]  # keep base-mode bit

    def check(name, B, ref=R):
        eq, comp = compare(ref, B)
        hc = health_check(ref, B)
        res[name] = {"pass": bool(eq and hc["health_equal"]), "bit_exact": eq, "max_abs_diff": comp, **hc}

    # T1
    SDS.restore(env, S)
    G1 = SDS.graph_digest(env)
    diffs = SDS.digest_diff(G0, G1)
    res["T0_graph_digest"] = {"pass": not diffs, "n_paths": len(G0), "differing_paths": diffs[:50]}
    check("T1_same_instance", run(env, seg))
    # T2
    SDS.restore(env, S)
    run(env, pert[:10])
    SDS.restore(env, S)
    check("T2_restore_twice", run(env, seg))
    # T3
    SDS.restore(env, S); X1 = run(env, pert)
    SDS.restore(env, S); Y1 = run(env, seg)
    SDS.restore(env, S); Y2 = run(env, seg)
    SDS.restore(env, S); X2 = run(env, pert)
    eqx, cx = compare(X1, X2)
    eqy1, _ = compare(R, Y1)
    eqy2, _ = compare(R, Y2)
    hx = health_check(X1, X2)
    res["T3_branch_order"] = {"pass": bool(eqx and eqy1 and eqy2 and hx["health_equal"]), "X_order_invariant": eqx,
                              "Y1_eq_R": eqy1, "Y2_eq_R": eqy2, "X_max_abs_diff": cx,
                              "X_differs_from_Y": not compare(X1, Y1)[0],
                              "X_health_delta": hx["health_delta_ref"], "X_damage_events": hx["damage_events_ref"]}
    # T6 post-chunk caching (via disk), twice
    SDS.restore(env, S)
    run(env, seg[:8])
    S8 = SDS.capture(env, episode={"t": tb + 8}, meta={"case": res["case"]})
    p8 = os.path.join(workdir, f"{demo}_{cond}_{task}_S8.pkl")
    SDS.save(S8, p8)
    S8l = SDS.load(p8)
    Rtail = {k: v[8:] for k, v in R.items()}
    SDS.restore(env, S8l); C1 = run(env, seg[8:])
    SDS.restore(env, S8l); C2 = run(env, seg[8:])
    e1, c1 = compare(Rtail, C1)
    e2, _ = compare(C1, C2)
    res["T6_post_chunk_cache"] = {"pass": bool(e1 and e2), "cached_eq_uncached": e1, "repeat_identical": e2,
                                  "max_abs_diff": c1, "snapshot_digest_roundtrip": SDS.snapshot_digest(S8) == SDS.snapshot_digest(S8l)}
    # N1 negative control: physics-only restore into the damaged env (after running through damage)
    SDS.restore(env, S); run(env, seg)
    import mujoco
    m, d = SDS._mj(env)
    mujoco.mj_setState(m, d, S["physics"]["state"], S["physics"]["spec"]); mujoco.mj_forward(m, d)
    q_equal = bool(np.array_equal(env.sim.data.qpos, S["physics"]["qpos"]))
    N1 = run(env, seg)
    hn = health_check(R, N1)
    eqn, _ = compare(R, N1)
    res["N1_physics_only_restore"] = {"qpos_equal_at_restore": q_equal, "checker_verdict": "PASS" if (eqn and hn["health_equal"]) else "FAIL",
                                      "expected": "FAIL", "ok": not (eqn and hn["health_equal"]), **hn}
    # N2 negative control: velocities re-synced (playback convention) instead of restored
    SDS.restore(env, S)
    for obj in env.get_damageable_objects():
        for ev in obj.damage_evaluators:
            for p in ev._get_damageable_part_names():
                ev.prev_linear_velocities[p] = ev._get_part_linear_velocity(p)
    N2 = run(env, seg)
    eq2, c2 = compare(R, N2)
    imp = np.abs(R["damage_info"][:, :, 1] - N2["damage_info"][:, :, 1])
    res["N2_velocity_resync"] = {"checker_verdict": "PASS" if eq2 else "FAIL", "expected": "FAIL", "ok": not eq2,
                                 "first_step_impact_maxdiff": float(imp[0].max()),
                                 "steps_with_impact_diff": int((imp.max(1) > 0).sum()),
                                 "health_equal": bool(np.array_equal(R["obs/health"], N2["obs/health"]))}
    # snapshot to disk for T4/T5
    sp = os.path.join(workdir, f"{demo}_{cond}_{task}_S.pkl")
    SDS.save(S, sp)
    ap = os.path.join(workdir, f"{demo}_{cond}_{task}_actions.npy")
    np.save(ap, seg)
    res["snapshot_bytes"] = os.path.getsize(sp)
    del env
    # T4 fresh env instance in this process
    env2 = make_env_cpu(task)
    setup_demo(env2, c)
    run(env2, acts[:3])  # make the fresh env "used" (different state) before restoring
    SDS.restore(env2, SDS.load(sp))
    G4 = SDS.graph_digest(env2)
    d4 = SDS.digest_diff(G0, G4, ignore_suffixes=FILE_PATH_ATTRS)
    res["T4_fresh_instance_ignored_paths"] = {k: [G0.get(k), G4.get(k)] for k in SDS.digest_diff(G0, G4) if k not in d4}
    check("T4_fresh_instance", run(env2, seg))
    res["T4_fresh_instance"]["graph_digest_diff_vs_capture"] = d4[:50]
    res["T4_fresh_instance"]["graph_digest_pass"] = not d4
    res["T4_fresh_instance"]["pass"] = res["T4_fresh_instance"]["pass"] and not d4
    del env2
    # T5 fresh process
    op = os.path.join(workdir, f"{demo}_{cond}_{task}_child.npz")
    rc = subprocess.call([PY, os.path.abspath(__file__), "--child", sp, ap, res["case"], op],
                         stdout=subprocess.DEVNULL, stderr=open(op + ".log", "w"))
    if rc == 0:
        Z = dict(np.load(op))
        check("T5_disk_fresh_process", Z)
    else:
        res["T5_disk_fresh_process"] = {"pass": False, "error": f"child rc={rc}, see {op}.log"}
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases")
    ap.add_argument("--out")
    ap.add_argument("--workdir", default="/edrive1/lizad/oopsieverse_full_study/runs/selection/restore_tests")
    ap.add_argument("--child", nargs=4)
    a = ap.parse_args()
    if a.child:
        child(*a.child)
        return
    os.makedirs(a.workdir, exist_ok=True)
    os.chdir("/edrive1/lizad/oopsieverse_full_study/src/oopsieverse")
    results = []
    for case in a.cases.split(","):
        r = test_case(*case.split(":"), a.workdir)
        results.append(r)
        print(json.dumps({k: (v.get("pass", v.get("ok")) if isinstance(v, dict) else v) for k, v in r.items()}), flush=True)
        json.dump(results, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
