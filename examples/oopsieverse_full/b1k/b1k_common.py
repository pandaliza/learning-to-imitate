"""Shared B1K (OmniGibson / Isaac Sim 4.5) helpers for the DIAGNOSTIC arm (protocol_v1 Amendment 1).

Evaluation environment = the official teleop environment (scripts/teleop_b1k.py build_env_config from the pinned
TaskConfig: 30 Hz action/physics/rendering, DummyTask) with the observation sensors of the official playback that
produced the training data (scripts/playback_b1k.py, --low_resolution): robot obs_modalities [proprio, rgb,
seg_instance] with 256x256 VisionSensor, 23-D playback proprio keys, and the task's two external cameras at 256x256
(build_external_sensors_config of playback_b1k.py), flatten_obs_space=True (as create_from_hdf5 sets).
Resets = official teleop reset_env (task reset(): init-state pkl + base/arm noise (+ object pose/scale noise for
shelve_item), re-rolled until every tracked entity is at full health), seeded per reset with
np.random / torch / random seeds = reset seed.

Must be imported AFTER the Kit args are on sys.argv (see kit_args()).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

OOPSIE_ROOT = "/edrive1/lizad/oopsieverse_full_study/src/oopsieverse"
STORE = "/edrive1/lizad/oopsieverse_full_study"
B1K_RUNS = os.path.join(STORE, "runs/b1k_diag")
REPO = "/scr/lizad/personal/learning-to-imitate"
B1K_OUT = "/edrive1/lizad/oopsieverse_full_study/reports/b1k"  # /scr full since 2026-10-23; copied back later

TIMEOUT = {"add_firewood": 550, "shelve_item": 1000}  # metric_contract_b1k_amendment_draft.json episode_protocol
POST_SUCCESS_STEPS = 20  # teleop_b1k.py:918-925 (20 extra actions after success), amendment draft
SHELVE_RANDOMIZED = ["book", "wineglass", "bottle_of_wine", "bottle_of_beer"]


def kit_args_from_env():
    g = os.environ.get("OMNIGIBSON_GPU_ID")
    assert g is not None and g != "0", "set OMNIGIBSON_GPU_ID to a probed working Isaac GPU (never 0)"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:  # verified 2026-10-23: CUDA index must equal the Vulkan/nvidia-smi index of the active GPU
        assert cvd.split(",")[int(g)] == g and "0" not in cvd.split(","), f"bad CVD {cvd} for GPU {g}"
    args = [f"--/renderer/activeGpu={g}", "--/renderer/multiGpu/maxGpuCount=1", "--/renderer/multiGpu/enabled=false",
            f"--/physics/cudaDevice={g}"]
    for a in args:
        if a not in sys.argv:
            sys.argv.append(a)
    return int(g)


def setup_paths():
    os.chdir(OOPSIE_ROOT)
    for p in (OOPSIE_ROOT, os.path.join(OOPSIE_ROOT, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_env(task, cameras=True):
    """Returns (env, task_cfg, task_mod, teleop_module)."""
    setup_paths()
    import torch as th  # noqa: F401
    import omnigibson as og  # noqa: F401
    from omnigibson.macros import gm
    import playback_b1k as pb
    import teleop_b1k as tb
    from damagesim.omnigibson.damageable_env import OGDamageableEnvironment
    from damagesim.utils.playback_schema import playback_proprio_keys

    task_mod = pb.load_task_module(task)
    cfg = task_mod.get_task_config()
    gm.USE_GPU_DYNAMICS = cfg.use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = cfg.enable_transition_rules
    env_config = tb.build_env_config(cfg)
    env_config["env"]["flatten_obs_space"] = True
    if os.environ.get("B1K_PHYSICS_HZ"):  # DIAGNOSTIC ONLY (competence diagnosis); primary = pinned 30 Hz
        env_config["env"]["physics_frequency"] = float(os.environ["B1K_PHYSICS_HZ"])
    rc = env_config["robots"][0]
    if cameras:
        rc["obs_modalities"] = ["proprio", "rgb", "seg_instance"]
        rc["sensor_config"] = {"VisionSensor": {"modalities": ["rgb", "seg_instance"],
                                                "sensor_kwargs": {"image_height": 256, "image_width": 256}}}
        rc["exclude_sensor_names"] = cfg.exclude_sensor_names
        env_config["env"]["external_sensors"] = pb.build_external_sensors_config(
            cfg, cfg.robot_name, cfg.robot_type, 256, 256)
    else:
        rc["obs_modalities"] = ["proprio"]
    rc["proprio_obs"] = playback_proprio_keys(arm="0")
    env = OGDamageableEnvironment(configs=env_config)
    return env, cfg, task_mod, tb


def seeded_reset(env, cfg, task_mod, tb, seed):
    import numpy as np
    import torch as th
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    tb.reset_env(env, cfg, task_mod)


def reset_snapshot(env, task):
    """Serialized sim state + non-state reset attributes (object scales) + readable poses."""
    import numpy as np
    import omnigibson as og
    st = og.sim.dump_state(serialized=True).detach().cpu().numpy().astype(np.float32)
    rec = {"state_size": int(st.size), "state_sha256": sha256_bytes(st.tobytes())}
    robot = env.robots[0]
    p, q = robot.get_position_orientation()
    rec["robot_base_pos"] = [round(float(x), 6) for x in p]
    rec["robot_base_quat"] = [round(float(x), 6) for x in q]
    rec["robot_qpos"] = [round(float(x), 6) for x in robot.get_joint_positions()]
    objs = {}
    for o in env.scene.objects:
        if o.name in ("fireplace", "log_center", "log_left", "target_object", "box_of_crackers", "book",
                      "bottle_of_wine", "wineglass", "bottle_of_beer", "stand"):
            op, oq = o.get_position_orientation()
            objs[o.name] = {"pos": [round(float(x), 6) for x in op], "quat": [round(float(x), 6) for x in oq],
                            "scale": [round(float(x), 6) for x in o.scale]}
    rec["objects"] = objs
    return rec, st


def entity_health(env):
    """{entity: min link health} for every tracked damageable entity (same rule as build_b1k_labels.py)."""
    out = {}
    for o in list(env.scene.objects) + list(env.robots):
        if getattr(o, "track_damage", False):
            lh = getattr(o, "link_healths", None)
            if lh:
                out[o.name] = float(min(float(v) for v in lh.values()))
    return out


def link_health_vector(env):
    names, vals = [], []
    for o in sorted(list(env.scene.objects) + list(env.robots), key=lambda x: x.name):
        if getattr(o, "track_damage", False):
            for ln, h in (getattr(o, "link_healths", None) or {}).items():
                names.append(f"{o.name}@{ln}")
                vals.append(float(h))
    return names, vals


def dumps(o):
    return json.dumps(o, sort_keys=True, default=str)
