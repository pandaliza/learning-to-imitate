"""Replay OopsieVerse teleop episodes in the pinned backend (spec section 6).

Gripper semantics (--gripper):
  released: the gripper exactly as the released playback script builds it (robosuite PandaGripper,
            binary sign-of-action with speed 0.2 ramp; robosuite/models/grippers/panda_gripper.py:43-58).
  teleop:   the semantics the demos were COLLECTED with: teleop_robocasa.py:1047 defaults to
            continuous_gripper=True, which calls damagesim.robosuite.io_utils.patch_gripper_for_position_control()
            (io_utils.py:33-50; gripper command = absolute finger position). We call that official function
            unchanged; no physics/evaluation code is edited.

Two modes, both starting from the official playback setup
(scripts/playback_robocasa.py:174-223, with the path-only remap from path_remap.py):
  force: official state-forcing playback. For each i: set states[i], forward, sync evaluator
         velocities, step(actions[i+1]) and re-score (identical to playback_episode).
  open : open-loop action replay. Set states[0] once, then step actions[1:] with no restore.

Alignment (teleop_robocasa.py:116-140): stored states[t], obs[t], damage_info[t] are all
recorded AFTER env.step(actions[t]). Replay step i (action index i+1) is therefore compared
with stored index i+1.

Per-episode arrays go to <storage>/runs/replay/<mode>/<task>__<cond>__<demo>.npz; a summary row is
returned. The stored HDF5s are opened read-only and never modified.
"""

import argparse
import json
import os
import sys
import time
import traceback

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import STORAGE, TASKS, add_oopsie_paths, teleop_path  # noqa: E402
from path_remap import remap_asset_paths  # noqa: E402

add_oopsie_paths()

RUN_DIR = os.path.join(STORAGE, "runs", "replay")
_ENVS = {}


GRIPPER = os.environ.get("OV_GRIPPER", "teleop")
os.environ.setdefault("PYNPUT_BACKEND", "dummy")  # io_utils imports pynput (teleop devices); headless host has no X


def get_env(env_name):
    if env_name not in _ENVS:
        if GRIPPER == "teleop":
            from damagesim.robosuite.io_utils import patch_gripper_for_position_control
            patch_gripper_for_position_control()
        else:
            assert GRIPPER == "released", GRIPPER
        from robosuite.controllers import load_composite_controller_config
        from envs.registry import EnvironmentRegistry
        cfg = EnvironmentRegistry.get(env_name)
        # Same kwargs as playback_robocasa.py:354-369 / teleop_robocasa.py:998-1009 except rendering
        # (no cameras, no segmentation, no onscreen viewer); rendering does not enter physics or damage.
        _ENVS[env_name] = cfg.damageable_class(
            robots=cfg.robot, controller_configs=load_composite_controller_config(robot=cfg.robot),
            translucent_robot=False, has_renderer=False, has_offscreen_renderer=False,
            render_camera=cfg.camera_name, ignore_done=True, use_camera_obs=False,
            render_segmentation=False, control_freq=cfg.control_freq)
    return _ENVS[env_name]


def sync_damage_evaluator_velocities(env):
    # verbatim logic of playback_robocasa.py:142-151
    for obj in env.get_damageable_objects():
        for evaluator in obj.damage_evaluators:
            if hasattr(evaluator, "prev_linear_velocities") and hasattr(evaluator, "_get_part_linear_velocity"):
                for part_name in evaluator._get_damageable_part_names():
                    evaluator.prev_linear_velocities[part_name] = evaluator._get_part_linear_velocity(part_name)


def setup_episode(env, g):
    ep_meta = json.loads(remap_asset_paths(g.attrs.get("ep_meta", "{}")))
    if ep_meta:
        env.set_ep_meta(ep_meta)
    else:
        env.unset_ep_meta()
    env.reset()
    env.reset_from_xml_string(remap_asset_paths(g.attrs["model_file"]))
    env.sim.reset()
    for robot in env.robots:
        for arm_name in robot.arms:
            if robot.eef_site_id[arm_name] is not None:
                env.sim.model.site_rgba[robot.eef_site_id[arm_name]] = np.zeros(4)
            if robot.eef_cylinder_id[arm_name] is not None:
                env.sim.model.site_rgba[robot.eef_cylinder_id[arm_name]] = np.zeros(4)
    health_after_reset = np.array(env.get_observations()[0]["health"], dtype=np.float64)
    states = g["states"][:]
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    sync_damage_evaluator_velocities(env)
    return states, health_after_reset


def link_table(link_names, damage_info):
    """Per-link [damage, impact, qs, raw_contact] in health_list order."""
    out = np.zeros((len(link_names), 4))
    for k, n in enumerate(link_names):
        o, l = n.split("@", 1)
        m = damage_info.get(o, {}).get(l, {}).get("mechanical", {})
        out[k] = [m.get("damage", np.nan), m.get("impact_forces", np.nan),
                  m.get("filtered_qs_forces", np.nan), m.get("filtered_raw_sim_forces", np.nan)]
    return out


OBS_KEYS = ["robot0_eef_pos", "robot0_joint_pos", "robot0_gripper_qpos", "object-state"]


def replay_episode(task, cond, demo, mode, save=True):
    from damagesim.utils.misc_utils import robocasa_check_success
    env_name = TASKS[task]
    env = get_env(env_name)
    t0 = time.time()
    with h5py.File(teleop_path(task, cond), "r") as f:
        g = f["data"][demo]
        link_names = [n.decode() if isinstance(n, bytes) else str(n) for n in g.attrs["health_list_link_names"]]
        actions = g["actions"][:]
        st_health = g["obs/health"][:].astype(np.float64)
        st_dinfo = [json.loads(x) for x in g["info/damage_info"][:]]
        st_obs = {k: g["obs"][k][:].astype(np.float64) for k in OBS_KEYS if k in g["obs"]}
        st_rewards = g["rewards"][:]
        states, health_after_reset = setup_episode(env, g)
    assert list(env.health_list_link_names) == link_names, (env.health_list_link_names, link_names)
    assert states.shape[1] == env.sim.get_state().flatten().shape[0]
    T = len(states)
    n = T - 1
    rp_health = np.zeros((n, len(link_names)))
    rp_tab = np.zeros((n, len(link_names), 4))
    st_tab = np.stack([link_table(link_names, d) for d in st_dinfo])
    rp_state_err = np.zeros(n)        # ||replayed post-step state - stored states[i+1]|| (qpos+qvel)
    rp_qpos_err = np.zeros(n)
    rp_obs_err = {k: np.zeros(n) for k in st_obs}
    rp_success = np.zeros(n, dtype=bool)
    rp_reward = np.zeros(n)
    rp_wiped = np.full(n, -1)
    nq = env.sim.model.nq
    initial_success = robocasa_check_success(env)
    for i in range(n):
        if mode == "force":
            env.sim.set_state_from_flattened(states[i])
            env.sim.forward()
            sync_damage_evaluator_velocities(env)
        obs, reward, done, info = env.step(actions[i + 1])
        rp_health[i] = obs["health"]
        rp_tab[i] = link_table(link_names, info["damage_info"])
        s = env.sim.get_state().flatten()
        d = s - states[i + 1]
        rp_state_err[i] = np.linalg.norm(d[1:])
        rp_qpos_err[i] = np.linalg.norm(d[1:1 + nq])
        for k in st_obs:
            rp_obs_err[k][i] = np.linalg.norm(np.asarray(obs[k], dtype=np.float64) - st_obs[k][i + 1])
        rp_success[i] = robocasa_check_success(env)
        rp_reward[i] = reward
        rp_wiped[i] = info.get("total_wiped", -1)
    elapsed = time.time() - t0
    arrays = dict(link_names=np.array(link_names), rp_health=rp_health, st_health=st_health,
                  rp_tab=rp_tab, st_tab=st_tab, rp_state_err=rp_state_err, rp_qpos_err=rp_qpos_err,
                  rp_success=rp_success, rp_reward=rp_reward, st_rewards=st_rewards, rp_wiped=rp_wiped,
                  health_after_reset=health_after_reset,
                  **{f"obs_err__{k}": v for k, v in rp_obs_err.items()})
    if save:
        d = os.path.join(RUN_DIR, f"{mode}_{GRIPPER}")
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, f"{task}__{cond}__{demo}.npz"), **arrays)
    return {"task": task, "cond": cond, "demo": demo, "mode": mode, "gripper": GRIPPER, "T": T, "elapsed_s": round(elapsed, 1),
            "initial_success": bool(initial_success), "final_success": bool(rp_success[-1]),
            "ever_success": bool(rp_success.any()),
            "first_success_step": int(np.argmax(rp_success)) + 1 if rp_success.any() else None}


def _job(args):
    task, cond, demo, mode = args
    try:
        return replay_episode(task, cond, demo, mode)
    except Exception as e:
        return {"task": task, "cond": cond, "demo": demo, "mode": mode, "error": repr(e), "tb": traceback.format_exc()[-2000:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="JSON list of [task, cond, demo, mode]")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True, help="JSONL summary output")
    a = ap.parse_args()
    jobs = [tuple(j) for j in json.load(open(a.jobs))]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            if "error" not in r and r.get("gripper", "released") == GRIPPER:
                done.add((r["task"], r["cond"], r["demo"], r["mode"]))
    jobs = [j for j in jobs if j not in done]
    # group by env so each worker process builds one env and reuses it
    jobs.sort(key=lambda j: (j[0], j[3], j[1], j[2]))
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(a.workers, max(1, len(jobs))), maxtasksperchild=None) as pool, open(a.out, "a") as fo:
        for r in pool.imap_unordered(_job, jobs):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            print(r.get("task"), r.get("cond"), r.get("demo"), r.get("mode"), r.get("elapsed_s"), r.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
