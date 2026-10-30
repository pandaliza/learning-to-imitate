"""Per-step wiped-marker identities for wipe_countertop demos (W3/W4 dirt-targeted intent), from the SIM STATE.

For each demo this re-runs the official playback loop (playback_robocasa_portable.playback_episode: set_ep_meta ->
reset -> reset_from_xml_string(model_file) -> sim.reset -> set_state(states[0]); then for each i: set_state(states[i]),
forward, sync damage-evaluator velocities, env.step(actions[1:][i])) in a render-free CPU env (no offscreen renderer,
no camera obs; physics, controller, damage evaluation and the wipe check are unchanged). The env's own
`_check_wiped_markers` is wrapped (read-only) to record WHICH marker indices are wiped at each step.

Recorded per demo (npz under runs/wipe_v2/markers/<cond>__<demo>.npz):
  marker_pos_world [50, 3]   marker body world positions after reset (xy == ep_meta["dirt_marker_positions"],
                             z = counter top + 0.46; asserted)
  wiped_step       [50] int  playback info index i at which marker m was wiped (-1 = never). Playback info[i] is
                             the step from state i with playback action[i], reaching state i+1.
  newly_wiped_sim  [T] int   per-step count from the sim; asserted == the playback part's info/newly_wiped
  sponge_pos       [T, 3]    sponge body world position after each step (state i+1)
  eef_pos_world    [T, 3]    EEF site world position after each step (state i+1)
Usage: python extract_markers.py <cond>/<demo> [...]     (cond in with/without)
"""
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.dirname(HERE)
for p in (os.path.join(FULL, "baseline"), FULL):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYNPUT_BACKEND", "dummy")
STORAGE = "/edrive1/lizad/oopsieverse_full_study"
TELEOP = os.path.join(STORAGE, "data/oopsieverse-demos/paper_experiments/teleop_data/wipe_countertop/{cond}_live_feedback.hdf5")
PART = os.path.join(STORAGE, "data/playback/parts/wipe_countertop__{cond}__{demo}.hdf5")
OUT = os.path.join(STORAGE, "runs/wipe_v2/markers")


def make_env_cpu():
    from damagesim.robosuite.io_utils import patch_gripper_for_position_control
    patch_gripper_for_position_control()  # collection-time gripper semantics, as the official playback wrapper
    from robosuite.controllers import load_composite_controller_config
    from envs.registry import EnvironmentRegistry
    cfg = EnvironmentRegistry.get("wipe_counter")
    return cfg.damageable_class(robots=cfg.robot, controller_configs=load_composite_controller_config(robot=cfg.robot),
                                translucent_robot=False, has_renderer=False, has_offscreen_renderer=False,
                                render_camera=cfg.camera_name, ignore_done=True, use_camera_obs=False,
                                render_segmentation=False, control_freq=cfg.control_freq)


def run_demo(env, cond, demo):
    import playback_robocasa_portable as pb
    from path_remap import remap_asset_paths
    with h5py.File(TELEOP.format(cond=cond), "r") as f:
        g = f["data"][demo]
        states = g["states"][:]
        actions = g["actions"][1:]
        ep_meta = json.loads(remap_asset_paths(g.attrs["ep_meta"]))
        xml = remap_asset_paths(g.attrs["model_file"])
    env.set_ep_meta(ep_meta)
    env.reset()
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    pb.sync_damage_evaluator_velocities(env)

    rec = {"cur": None}
    orig = env._check_wiped_markers

    def hooked():
        out = orig()
        rec["cur"] = [env.markers.index(m) for m in out]
        return out
    env._check_wiped_markers = hooked
    mpos = np.array([env.sim.data.body_xpos[env.sim.model.body_name2id(m.root_body)] for m in env.markers], np.float64)
    em_xy = np.asarray(ep_meta["dirt_marker_positions"], np.float64)
    assert np.abs(mpos[:, :2] - em_xy).max() < 1e-9, "marker xy != ep_meta"
    sponge_id = env.sim.model.body_name2id(env.objects["sponge"].root_body)
    arm = env.robots[0].arms[0]
    site = env.robots[0].eef_site_id[arm]
    T = len(actions)
    wiped_step = -np.ones(len(env.markers), np.int64)
    nw = np.zeros(T, np.int64)
    sp = np.zeros((T, 3))
    ee = np.zeros((T, 3))
    for i in range(T):
        env.sim.set_state_from_flattened(states[i])
        env.sim.forward()
        pb.sync_damage_evaluator_velocities(env)
        rec["cur"] = None
        _obs, _r, _d, info = env.step(actions[i])
        ids = rec["cur"] or []
        assert len(ids) == int(info["newly_wiped"])
        for m in ids:
            assert wiped_step[m] == -1
            wiped_step[m] = i
        nw[i] = len(ids)
        sp[i] = env.sim.data.body_xpos[sponge_id]
        ee[i] = env.sim.data.site_xpos[site]
    env._check_wiped_markers = orig
    with h5py.File(PART.format(cond=cond, demo=demo), "r") as f:
        g = f["data"][demo]
        nw_pb = np.array([int(x) for x in g["info/newly_wiped"][:]])
        prop = g["obs/proprio"][:]
    res = {"marker_pos_world": mpos, "wiped_step": wiped_step, "newly_wiped_sim": nw, "newly_wiped_playback": nw_pb,
           "sponge_pos": sp, "eef_pos_world": ee}
    # checks vs the playback part used for training
    res["counts_equal"] = np.array(len(nw_pb) == T and np.array_equal(nw_pb, nw))
    res["eef_vs_proprio_maxabs"] = np.array(np.abs(ee - prop[1:T + 1, 14:17]).max() if len(prop) == T + 1 else np.nan)
    return res


def main():
    os.chdir(os.path.join(STORAGE, "src/oopsieverse"))
    import playback_robocasa_portable  # noqa: F401  (inserts the oopsieverse/oopsiebench paths)
    env = make_env_cpu()
    os.makedirs(OUT, exist_ok=True)
    for d in sys.argv[1:]:
        cond, demo = d.split("/")
        out = os.path.join(OUT, f"{cond}__{demo}.npz")
        if os.path.exists(out):
            print("exists", out)
            continue
        r = run_demo(env, cond, demo)
        tmp = out + f".tmp{os.getpid()}.npz"
        np.savez(tmp, **r)
        os.replace(tmp, out)
        print(json.dumps({"demo": d, "T": int(len(r["newly_wiped_sim"])), "wiped": int((r["wiped_step"] >= 0).sum()),
                          "counts_equal": bool(r["counts_equal"]), "eef_vs_proprio": float(r["eef_vs_proprio_maxabs"])}),
              flush=True)


if __name__ == "__main__":
    main()
