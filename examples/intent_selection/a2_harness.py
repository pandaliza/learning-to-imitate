"""A2 restoration harness: env construction, fixed action sequences, trajectory recording.

Also a CLI worker used by the fresh-process / multi-worker test:

    .venv/bin/python examples/intent_selection/a2_harness.py worker \
        --jobs jobs.pkl --order 2,0,1 --out results.pkl [--construction-seed 7]
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import os
import pickle
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from mip.intent_selection.snapshots import find_layers, restore_full_state  # noqa: E402

DATASET_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/snapshots/*/robomimic/square/mh/low_dim.hdf5"
)

_CFG = None


def task_config():
    """TaskConfig for square_mh_state, composed exactly like train_robomimic.py."""
    global _CFG
    if _CFG is None:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=f"{REPO}/examples/configs", version_base=None):
            cfg = compose(config_name="main", overrides=["task=square_mh_state"])
        _CFG = cfg.task
    return _CFG


def build_env(construction_seed=0, reset_seed=None):
    """make_env(task, idx=0, render=False, seed=construction_seed)(); optionally reset."""
    from mip.envs.robomimic.robomimic_env import make_env

    with contextlib.redirect_stdout(io.StringIO()):
        env = make_env(task_config(), 0, render=False, seed=construction_seed)()
    if reset_seed is not None:
        env.reset(seed=reset_seed)
    return env


def robosuite_env(env):
    return find_layers(env)["robosuite"]


def synthetic_sequence(seed: int, n: int, toggle_every: int = 9) -> np.ndarray:
    """Legal 7-D OSC_POSE actions for the env's actual (delta) controller mode.

    Translation in [-0.4, 0.4], rotation in [-0.15, 0.15] (normalized units, env clips to
    [-1, 1]); gripper toggles between -1 (open) and +1 (close) every ``toggle_every`` steps.
    """
    rng = np.random.default_rng(seed)
    a = np.zeros((n, 7))
    a[:, :3] = rng.uniform(-0.4, 0.4, size=(n, 3))
    a[:, 3:6] = rng.uniform(-0.15, 0.15, size=(n, 3))
    phase = int(rng.integers(0, 2))
    a[:, 6] = np.where(((np.arange(n) // toggle_every) + phase) % 2 == 0, -1.0, 1.0)
    return a


def demo(demo_key="demo_1"):
    import h5py

    path = sorted(glob.glob(DATASET_GLOB))[0]
    with h5py.File(path, "r") as f:
        return {"states": f[f"data/{demo_key}/states"][:], "actions": f[f"data/{demo_key}/actions"][:]}


def reset_to_demo(env, states0):
    """Reset through the normal stack, then set the demo's initial sim state (robomimic reset_to)."""
    env.reset(seed=0)
    layers = find_layers(env)
    layers["robomimic"].reset_to({"states": states0})
    # re-sync controller goals/observables to the new sim state exactly like a reset would
    rs = layers["robosuite"]
    for r in rs.robots:
        r.composite_controller.update_state()
        for c in r.composite_controller.part_controllers.values():
            c.update(force=True)
        r.composite_controller.reset()
    rs._obs_cache = {}
    rs._update_observables(force=True)
    ms = layers["multistep"]
    from collections import deque

    obs = layers["lowdim"].get_observation()
    ms.obs = deque([obs], maxlen=ms.n_obs_steps + 1)
    ms.reward, ms.done = [], []


def run_sequence(env, actions, chunk: int = 1) -> dict:
    """Step the full wrapper stack and record per-env-step quantities.

    ``chunk`` actions are sent per MultiStepWrapper.step call (1 = per-step recording via the
    wrapper; 8 = native replan chunks). Records are per env step either way.
    """
    rs = robosuite_env(env)
    ms = find_layers(env)["multistep"]
    robot = rs.robots[0]
    osc = robot.composite_controller.part_controllers["right"]
    grip = robot.composite_controller.part_controllers["right_gripper"]
    nut_bid = rs.obj_body_id[rs.nuts[rs.nut_id].name]
    eef_sid = robot.eef_site_id["right"]
    rec = {k: [] for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "eef_pos", "eef_quat_obs",
                           "object_obs", "nut_pos", "nut_quat", "reward", "success", "done",
                           "env_done", "obs", "osc_goal_pos", "osc_goal_ori", "osc_torques",
                           "gripper_torques", "gripper_current_action", "time")}
    actions = np.asarray(actions, dtype=np.float64)
    rs_obs_dim = int(np.asarray(ms.obs[-1]).shape[0])
    i = 0
    while i < len(actions):
        if ms.done and ms.done[-1]:
            break
        block = actions[i:i + chunk]
        n_before = len(ms.reward)
        obs, _r, _term, _trunc, _info = env.step(block)
        n_new = len(ms.reward) - n_before
        # per-step obs come from the wrapper deque (last n_new entries)
        # (the deque holds only n_obs_steps + 1 frames; older steps of a long chunk are NaN)
        kept = list(ms.obs)[-min(n_new, len(ms.obs)):] if n_new > 0 else []
        for j in range(n_new):
            jj = j - (n_new - len(kept))
            o = np.asarray(kept[jj], dtype=np.float64) if jj >= 0 else np.full(rs_obs_dim, np.nan)
            rec["obs"].append(o.copy())
            rec["object_obs"].append(o[:14].copy())
            rec["eef_pos"].append(o[14:17].copy())
            rec["eef_quat_obs"].append(o[17:21].copy())
            rec["reward"].append(float(ms.reward[n_before + j]))
            rec["done"].append(bool(ms.done[n_before + j]))
        # physics / controller quantities are only observable at chunk end; record them for
        # the last step of the chunk, and pad earlier steps with NaN when chunk > 1
        d = rs.sim.data
        snap = {
            "qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "qacc_warmstart": d.qacc_warmstart.copy(),
            "ctrl": d.ctrl.copy(), "nut_pos": d.xpos[nut_bid].copy(), "nut_quat": d.xquat[nut_bid].copy(),
            "success": bool(np.sum(rs.objects_on_pegs) > 0), "env_done": bool(rs.done),
            "osc_goal_pos": np.array(osc.goal_pos, dtype=np.float64).copy(),
            "osc_goal_ori": np.array(osc.goal_ori, dtype=np.float64).copy(),
            "osc_torques": np.array(osc.torques, dtype=np.float64).copy(),
            "gripper_torques": np.array(grip.torques, dtype=np.float64).copy(),
            "gripper_current_action": np.array(robot.gripper["right"].current_action).copy(),
            "time": float(d.time),
        }
        for j in range(n_new):
            for k, v in snap.items():
                if j == n_new - 1:
                    rec[k].append(v)
                else:
                    rec[k].append(np.full_like(np.asarray(v, dtype=np.float64), np.nan))
        i += chunk
    _ = eef_sid
    return {k: np.asarray(v) for k, v in rec.items()}


def max_dev(a: dict, b: dict) -> dict:
    """Max abs deviation per key (NaN-aware); inf if lengths differ."""
    out = {}
    for k in a:
        x, y = np.asarray(a[k], dtype=np.float64), np.asarray(b[k], dtype=np.float64)
        if x.shape != y.shape:
            out[k] = float("inf")
            continue
        if x.size == 0:
            out[k] = 0.0
            continue
        both_nan = np.isnan(x) & np.isnan(y)
        diff = np.where(both_nan, 0.0, np.abs(x - y))
        out[k] = float(np.nanmax(np.where(np.isnan(diff), np.inf, diff)))
    return out


def bitwise_equal(a: dict, b: dict) -> bool:
    return all(np.asarray(a[k]).shape == np.asarray(b[k]).shape
               and np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes() for k in a)


# --------------------------------------------------------------------------------------
# worker CLI
# --------------------------------------------------------------------------------------


def _worker(args):
    with open(args.jobs, "rb") as f:
        jobs = pickle.load(f)  # {"snapshots": {sid: bytes}, "seqs": {qid: arr}, "keys": [(sid,qid)]}
    from mip.intent_selection.snapshots import Snapshot

    env = build_env(construction_seed=args.construction_seed, reset_seed=args.construction_seed)
    keys = jobs["keys"]
    order = [int(x) for x in args.order.split(",")] if args.order else list(range(len(keys)))
    out = {"pid": os.getpid(), "order": order, "results": {}}
    for idx in order:
        sid, qid = keys[idx]
        snap = Snapshot.from_bytes(jobs["snapshots"][sid])
        restore_full_state(snap, env)
        out["results"][(sid, qid)] = run_sequence(env, jobs["seqs"][qid])
    with open(args.out, "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--jobs", required=True)
    w.add_argument("--order", default="")
    w.add_argument("--out", required=True)
    w.add_argument("--construction-seed", type=int, default=0)
    a = p.parse_args()
    if a.cmd == "worker":
        _worker(a)
