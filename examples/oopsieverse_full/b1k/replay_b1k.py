"""Instrumented wrapper around the official OopsieVerse B1K playback.

Runs ``scripts/playback_b1k.py::run_playback`` from the pinned oopsieverse source
unchanged, except for:
  * portability: cwd + sys.path point at the pinned source (relative INIT_STATE_PATH,
    task modules), output paths are explicit CLI args;
  * instrumentation (read-only): the per-task ``playback_step_fn`` hook is wrapped so that
    after every playback env.step we record wall time and the post-step serialized
    sim state. After each episode, that state is compared to the *next* stored state
    dump (state[i+2]) to quantify one-step divergence of the official state-forced replay.

Usage:
  python replay_b1k.py --task_name shelve_item --source_hdf5_path X.hdf5 \
      --playback_hdf5_path out.hdf5 --demo_ids 0 1 --metrics_json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

OOPSIE_ROOT = os.environ.get(
    "OOPSIE_ROOT", "/edrive1/lizad/oopsieverse_full_study/src/oopsieverse"
)
os.chdir(OOPSIE_ROOT)
sys.path.insert(0, OOPSIE_ROOT)
sys.path.insert(0, os.path.join(OOPSIE_ROOT, "scripts"))

T_PROC0 = time.time()
import playback_b1k as pb  # noqa: E402  (official script, imported as a module)
import torch as th  # noqa: E402
import omnigibson as og  # noqa: E402

T_IMPORT = time.time() - T_PROC0

REC = {"episodes": {}, "t_import": T_IMPORT}
_cur = {"ep": None, "t_last": None, "post_states": [], "step_dt": []}


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}/{k}" if prefix else str(k)))
    elif th.is_tensor(d):
        out[prefix] = d.detach().float().cpu().reshape(-1)
    return out


# ── Layout shim (portability): shelve_item recordings lack the robot Temperature state ──
SHIM = {"enabled": False, "n_applied": 0, "n_skipped_same_size": 0, "robot_uuid": None,
        "insert_value": None, "checks": {}}


def _shim_fix(state, saved_size=None):
    """Insert the robot's missing Temperature value (OG DEFAULT_TEMPERATURE) before its AG block.

    Applies only when ALL hold (otherwise raises): stored length == runtime length - 1,
    the robot is the last uuid-tagged registry entry of the stored vector, and the runtime robot's
    last stateful non_kin state is Temperature (size 1). Returns (state, size).
    """
    st = th.as_tensor(state, dtype=th.float32).flatten()
    n = int(saved_size) if saved_size is not None else int(st.numel())
    st = st[:n]
    if not SHIM["enabled"]:
        return st, n
    rt = SHIM["runtime_size"]
    if n == rt:
        SHIM["n_skipped_same_size"] += 1
        return st, n
    assert n == rt - 1, f"shim: unexpected stored size {n} vs runtime {rt}"
    u = float(SHIM["robot_uuid"])
    idx = th.nonzero((st - u).abs() < 0.5).flatten()
    assert idx.numel() >= 1, "shim: robot uuid not found in stored state"
    i = int(idx[-1])
    blk = n - (i + 1)
    assert blk == SHIM["robot_block_runtime"] - 1, f"shim: robot is not the last entry or block size {blk} unexpected"
    ag = SHIM["ag_size"]
    ins = n - ag  # position right before the AG block
    out = th.cat([st[:ins], th.tensor([SHIM["insert_value"]], dtype=st.dtype), st[ins:]])
    SHIM["n_applied"] += 1
    return out, n + 1


def _enable_shim(env):
    """Called after env creation: derive the runtime facts the shim checks against."""
    from omnigibson.object_states import Temperature
    from omnigibson.object_states.temperature import m as temp_m

    robot = env.robots[0]
    rstate = robot.dump_state(serialized=False)
    non_kin = list(rstate["non_kin"].keys())
    SHIM["checks"]["robot_non_kin_runtime"] = non_kin
    assert non_kin and non_kin[-1] == "Temperature", f"shim: runtime robot non_kin {non_kin}"
    assert int(robot.states[Temperature].serialize(rstate["non_kin"]["Temperature"]).numel()) == 1
    SHIM["robot_uuid"] = int(robot.uuid)
    SHIM["robot_block_runtime"] = int(robot.serialize(rstate).numel())
    SHIM["ag_size"] = int(sum(robot._ag_serialized_state_size(a) for a in robot.arm_names))
    SHIM["runtime_size"] = None  # measured at the first load of each episode (after scene.restore)
    SHIM["insert_value"] = float(temp_m.DEFAULT_TEMPERATURE)
    SHIM["checks"]["robot_block_runtime"] = SHIM["robot_block_runtime"]
    SHIM["checks"]["ag_size"] = SHIM["ag_size"]
    SHIM["enabled"] = True

    from damagesim.omnigibson.damageable_env import OGDamageableDataPlaybackWrapper as W

    def _load(self, state, saved_size):
        if SHIM["runtime_size"] is None:
            SHIM["runtime_size"] = int(og.sim.dump_state(serialized=True).numel())
            SHIM["checks"].setdefault("runtime_size_per_episode", []).append(SHIM["runtime_size"])
        st, n = _shim_fix(state, saved_size)
        og.sim.load_state(st, serialized=True)  # exact size; no truncation fallback
        return n

    W._load_state_with_size_fallback = _load


def _divergence(post_states, stored_states, stored_sizes):
    """Compare post-step state i with stored state[i+2] (next recorded state).

    Returns per-step max position error over objects (m), the worst object, and a
    whole-vector max-abs diff. Uses og.sim.deserialize so no state is loaded.
    """
    per_step = []
    worst_obj = {}
    for i, ps in enumerate(post_states):
        j = i + 2
        if j >= len(stored_states):
            break
        ref, _ = _shim_fix(stored_states[j], int(stored_sizes[j]))
        if ps.numel() != ref.numel():
            per_step.append({"i": i, "size_mismatch": [int(ps.numel()), int(ref.numel())]})
            continue
        a = _flatten(og.sim.deserialize(ps)[0])
        b = _flatten(og.sim.deserialize(ref)[0])
        pos_err = {}
        for k in a:
            if k in b and k.endswith("/pos") and a[k].numel() == 3:
                pos_err[k] = float(th.linalg.norm(a[k] - b[k]))
        jp_err = max(
            [float((a[k] - b[k]).abs().max()) for k in a if k in b and "joint_pos" in k and a[k].numel()] or [0.0]
        )
        top = max(pos_err.items(), key=lambda kv: kv[1]) if pos_err else ("", 0.0)
        worst_obj[top[0]] = max(worst_obj.get(top[0], 0.0), top[1])
        per_step.append({"i": i, "max_pos_err_m": top[1], "argmax": top[0], "max_joint_pos_err": jp_err})
    errs = [p["max_pos_err_m"] for p in per_step if "max_pos_err_m" in p]
    jerrs = [p["max_joint_pos_err"] for p in per_step if "max_joint_pos_err" in p]
    return {
        "n_compared": len(errs),
        "max_pos_err_m": max(errs) if errs else None,
        "median_pos_err_m": float(sorted(errs)[len(errs) // 2]) if errs else None,
        "max_joint_pos_err": max(jerrs) if jerrs else None,
        "median_joint_pos_err": float(sorted(jerrs)[len(jerrs) // 2]) if jerrs else None,
        "worst_objects": dict(sorted(worst_obj.items(), key=lambda kv: -kv[1])[:5]),
        "n_size_mismatch": sum(1 for p in per_step if "size_mismatch" in p),
        "n_steps_pos_err_gt_2cm": sum(1 for e in errs if e > 0.02),
        "n_steps_pos_err_gt_5cm": sum(1 for e in errs if e > 0.05),
        "per_step": [[p["i"], round(p["max_pos_err_m"], 5), p["argmax"].split("/")[3] if p["argmax"].count("/") >= 3 else p["argmax"],
                      round(p["max_joint_pos_err"], 5)] for p in per_step if "max_pos_err_m" in p],
    }


_orig_attach = pb.attach_task_playback_hooks


def _attach_instrumented(env, task_cfg, task_mod):
    _orig_attach(env, task_cfg, task_mod)
    if ARGS is not None and ARGS.robot_temperature_shim:
        _enable_shim(env)
        REC["shim"] = {k: v for k, v in SHIM.items()}
    orig_step_fn = env.playback_step_fn

    def step_fn(e):
        if orig_step_fn is not None:
            orig_step_fn(e)
        now = time.time()
        if _cur["t_last"] is not None:
            _cur["step_dt"].append(now - _cur["t_last"])
        _cur["t_last"] = now
        _cur["post_states"].append(og.sim.dump_state(serialized=True).clone())

    env.playback_step_fn = step_fn

    orig_episode = env.playback_episode

    def playback_episode(episode_id, **kw):
        _cur.update(ep=episode_id, t_last=None, post_states=[], step_dt=[])
        if SHIM["enabled"]:
            SHIM["runtime_size"] = None
        t0 = time.time()
        res = orig_episode(episode_id=episode_id, **kw)
        wall = time.time() - t0
        grp = env.input_hdf5["data"][f"demo_{episode_id}"]
        div = _divergence(_cur["post_states"], grp["state"][()], grp["state_size"][()])
        dts = _cur["step_dt"]
        REC["shim"] = {k: v for k, v in SHIM.items()}
        REC["episodes"][str(episode_id)] = {
            "wall_s": wall,
            "n_actions": int(grp["action"].shape[0]),
            "n_steps_timed": len(dts),
            "median_step_s": float(sorted(dts)[len(dts) // 2]) if dts else None,
            "steps_per_s": (len(dts) / sum(dts)) if dts else None,
            "divergence_vs_next_stored_state": div,
        }
        print(f"[replay_b1k] demo_{episode_id}: {json.dumps(REC['episodes'][str(episode_id)])}", flush=True)
        _flush()
        return res

    env.playback_episode = playback_episode


pb.attach_task_playback_hooks = _attach_instrumented


def _disable_external_cameras():
    """Cost-only change: drop the recorded config's external VisionSensors (2x720p rgb+seg),
    which official --skip_save_images keeps. Physics/damage/success are untouched."""
    from damagesim.omnigibson.damageable_env import OGDamageableDataPlaybackWrapper as W

    orig = W.create_from_hdf5.__func__

    def create_from_hdf5(cls, input_path, output_path, **kw):
        if kw.get("external_sensors_config") is None:
            kw["external_sensors_config"] = []
        return orig(cls, input_path, output_path, **kw)

    W.create_from_hdf5 = classmethod(create_from_hdf5)

ARGS = None


def _flush():
    if ARGS is not None and ARGS.metrics_json:
        with open(ARGS.metrics_json, "w") as f:
            json.dump(REC, f, indent=1)


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", required=True)
    p.add_argument("--source_hdf5_path", required=True)
    p.add_argument("--playback_hdf5_path", required=True)
    p.add_argument("--demo_ids", nargs="*", type=int, default=None)
    p.add_argument("--metrics_json", default=None)
    p.add_argument("--with_images", action="store_true", help="save 256px images (default: skip)")
    p.add_argument("--no_external_cameras", action="store_true")
    p.add_argument("--robot_temperature_shim", action="store_true",
                   help="shelve_item: insert missing robot Temperature (23.0) into stored states")
    ARGS = p.parse_args()
    if ARGS.no_external_cameras:
        _disable_external_cameras()

    task_mod = pb.load_task_module(ARGS.task_name)
    task_cfg = task_mod.get_task_config()
    pb_args = argparse.Namespace(
        task_name=ARGS.task_name,
        source_hdf5_path=ARGS.source_hdf5_path,
        playback_hdf5_path=ARGS.playback_hdf5_path,
        demo_ids=ARGS.demo_ids,
        low_resolution=True,
        skip_save_images=not ARGS.with_images,
        camera_name="external_sensor0",
        video_dir=None,
    )
    t0 = time.time()
    REC.update(task=ARGS.task_name, source=ARGS.source_hdf5_path, demo_ids=ARGS.demo_ids, no_external_cameras=ARGS.no_external_cameras)
    try:
        pb.run_playback(pb_args, task_cfg, task_mod)
        REC["status"] = "ok"
    except Exception:
        import traceback

        REC["status"] = "error"
        REC["error"] = traceback.format_exc()
        print(REC["error"], flush=True)
    REC["t_run_playback_total_s"] = time.time() - t0
    REC["t_process_s"] = time.time() - T_PROC0
    _flush()
    og.shutdown()


if __name__ == "__main__":
    main()
