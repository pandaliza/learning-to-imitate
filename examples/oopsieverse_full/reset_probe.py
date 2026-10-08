"""Seeded env reset + initial-state fingerprint for the OopsieVerse reset banks (spec sections 7, 10, 11).

Runs inside the isolated env /edrive1/lizad/oopsieverse_full_study/env/oopsie (CPU only, no renderer).

Reset mechanics (verified-source, pinned clones):
  * robosuite MujocoEnv.__init__ creates ``self.rng = np.random.default_rng(seed)`` once
    (robosuite/environments/base.py:144-145). Every reset draw (robot base deviation
    robocasa env_utils.py:1531/1602, robot joint init noise robot_env.py:531 -> robot.py:251,
    object placement samplers env_utils.py:916/1005, wipe dirt layout wipe_counter.py:_sample_start_pos)
    consumes this generator. A second reset on the same env therefore draws NEW values; it does
    not reproduce the first reset.
  * ``self._ep_meta`` persists across resets until ``unset_ep_meta()`` is called
    (base.py:404-416). Kitchen._reset_internal reuses ``init_robot_base_pos`` from it
    (kitchen.py:1118-1121), Kitchen._setup_model reuses layout/style, and WipeCounter reuses
    ``dirt_marker_positions`` (wipe_counter.py:_reset_dirt_markers). The teleop collection wrapper
    calls ``env.set_ep_meta(env.get_ep_meta())`` on every reset (teleop_robocasa.py:92-107), and
    so do replay/playback, which is why demos within a collection session share one base pose /
    dirt layout. A used env that ever had ep_meta set is NOT a fresh reset.
  * The only global-numpy draw in the reset path is a visual-only counter rgba
    (robocasa/models/fixtures/counter.py:369,474).

Canonical study reset for seed s (``seeded_reset``): ``env.unset_ep_meta(); env.rng =
np.random.default_rng(s); env.reset()``. ``build_reset_banks.py tests`` checks that this equals a
fresh env constructed with ``seed=s`` and reset once.
"""

import hashlib
import json
import re
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import add_oopsie_paths  # noqa: E402

add_oopsie_paths()
os.environ.setdefault("PYNPUT_BACKEND", "dummy")

# study task name -> oopsiebench registry name / tracked object
REGISTRY = {"pick_egg": "pick_egg", "wipe_countertop": "wipe_counter"}
OBJ = {"pick_egg": "egg", "wipe_countertop": "sponge"}


def make_env(task, seed=None):
    """Same kwargs as replay.py / teleop_robocasa.py:998-1009 minus rendering; collection-time gripper."""
    from damagesim.robosuite.io_utils import patch_gripper_for_position_control
    patch_gripper_for_position_control()
    from robosuite.controllers import load_composite_controller_config
    from envs.registry import EnvironmentRegistry
    cfg = EnvironmentRegistry.get(REGISTRY[task])
    return cfg.damageable_class(
        robots=cfg.robot, controller_configs=load_composite_controller_config(robot=cfg.robot),
        translucent_robot=False, has_renderer=False, has_offscreen_renderer=False,
        render_camera=cfg.camera_name, ignore_done=True, use_camera_obs=False,
        render_segmentation=False, control_freq=cfg.control_freq, seed=seed)


def fresh_obs_after_reset(env):
    """Observation of the actual post-reset state.

    DamageableWipeCounter.reset (wipe_counter.py) returns the observation from BEFORE its IK
    placement and 5 zero-action steps (it discards reset_robot_above_sponge's return value), so
    the returned robot proprio/health are stale. Always re-read observations after reset.
    """
    return env.get_observations()[0]


def seeded_reset(env, seed):
    """Canonical study reset: clear sticky ep_meta, reseed env.rng, reset, re-read obs."""
    env.unset_ep_meta()
    env.rng = np.random.default_rng(seed)
    env.reset()
    return fresh_obs_after_reset(env)


def fresh_env_reset(task, seed):
    env = make_env(task, seed=seed)
    env.reset()
    return env, fresh_obs_after_reset(env)


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


def _r(x, n=6):
    return [round(float(v), n) for v in np.asarray(x).ravel()]


def fingerprint(env, task, obs):
    """Initial-state summary. World frame; base yaw is fixed (robot_spawn_deviation_rot=0)."""
    from damagesim.utils.misc_utils import robocasa_check_success
    sim = env.sim
    state = sim.get_state().flatten()
    obj = OBJ[task]
    bid = env.obj_body_id[obj]
    obj_pos = np.array(sim.data.body_xpos[bid])
    obj_quat = np.array(sim.data.body_xquat[bid])  # wxyz
    yaw = float(np.arctan2(2 * (obj_quat[0] * obj_quat[3] + obj_quat[1] * obj_quat[2]),
                           1 - 2 * (obj_quat[2] ** 2 + obj_quat[3] ** 2)))
    base_pos = np.array(obs["robot0_base_pos"])
    counter_top = float(env.counter.pos[2] + env.counter.height / 2)
    health = np.asarray(obs["health"], dtype=np.float64)
    fp = dict(
        state_sha=_h(state[1:]),  # drop sim time
        qpos_sha=_h(sim.data.qpos), qvel_sha=_h(sim.data.qvel),
        state_dim=int(state.shape[0]),
        model_sha=hashlib.sha256(sim.model.get_xml().encode()).hexdigest()[:16],
        # counter.py:369,474 draws a geom rgba from the GLOBAL np.random (visual only), and the first model
        # load in a process serializes mesh content_type attributes that later loads omit; the normalized
        # hash ignores both (verified: no other XML difference between a fresh and a reused env)
        model_sha_norm=hashlib.sha256(re.sub(r' (rgba|content_type)="[^"]*"', "", sim.model.get_xml()).encode()).hexdigest()[:16],
        base_pos=_r(base_pos), base_quat=_r(obs["robot0_base_quat"]),
        init_robot_base_pos=_r(env.init_robot_base_pos),
        eef_pos=_r(obs["robot0_eef_pos"]), joint_pos=_r(obs["robot0_joint_pos"]),
        gripper_qpos=_r(obs["robot0_gripper_qpos"]),
        obj_pos=_r(obj_pos), obj_yaw=round(yaw, 6), obj_quat=_r(obj_quat),
        obj_minus_base=_r(obj_pos - base_pos),
        obj_height_above_counter=round(float(obj_pos[2] - counter_top), 6),
        health_min=round(float(health.min()), 6), health_sha=_h(health),
        initial_damage=bool(health.min() < 100.0 - 1e-6),
        success_at_reset=bool(robocasa_check_success(env)),
        layout_style=[int(env.layout_id), int(env.style_id)],
    )
    if task == "wipe_countertop":
        mp = np.array(env._marker_positions, dtype=np.float64)
        fp.update(dirt_sha=_h(mp), dirt_start=_r(mp[0]), dirt_centroid=_r(mp.mean(0)),
                  dirt_direction=round(float(env._marker_direction), 6),
                  n_markers=len(env.markers), n_wiped=len(env.wiped_markers))
    return fp


def health_info(env, obs):
    h = np.asarray(obs["health"], dtype=np.float64)
    return dict(health_min=float(h.min()), names_below_100=[n for n, v in zip(env.health_list_link_names, h) if v < 100 - 1e-6])


if __name__ == "__main__":
    import time
    task = sys.argv[1]
    seed = int(sys.argv[2])
    t0 = time.time()
    env = make_env(task, seed=seed)
    t1 = time.time()
    env.reset()
    obs = fresh_obs_after_reset(env)
    t2 = time.time()
    fp = fingerprint(env, task, obs)
    fp["construct_s"], fp["reset_s"] = round(t1 - t0, 2), round(t2 - t1, 2)
    t3 = time.time()
    obs = seeded_reset(env, seed + 1)
    fp["second_reset_s"] = round(time.time() - t3, 2)
    print(json.dumps(fp))
