"""Full environment state capture / restore for deterministic branching (spec §4 A2).

Verified against the real stack returned by
``mip.envs.robomimic.robomimic_env.make_env`` for robomimic square (robosuite 1.5.1,
mujoco 3.2.3)::

    MultiStepWrapper -> VideoRecordingWrapper -> RobomimicLowdimWrapper
      -> robomimic EnvRobosuite -> robosuite NutAssemblySquare (MujocoEnv)
           -> FixedBaseRobot -> CompositeController{right: OSC_POSE, right_gripper: GRIP}
                            -> PandaGripper model (stateful ``current_action``)

State captured (every layer, nothing derived from re-stepping physics):

* MuJoCo: ``mj_getState(mjSTATE_INTEGRATION)`` (time, qpos, qvel, act, qacc_warmstart,
  ctrl, qfrc_applied, xfrc_applied, eq_active, mocap_pos/quat, userdata, plugin state),
  plus named copies for diagnostics and a model fingerprint checked on restore.
* robosuite env: ``timestep``, ``cur_time``, ``done``, task latches (``objects_on_pegs``
  and any other mutable ndarray/scalar attribute on the env), ``_ep_meta``, ``_obs_cache``,
  every observable's sampling state (``_time_since_last_sample``, ``_current_delay``,
  ``_current_observed_value``, ``_sampled``, ``_enabled``, ``_active``), env ``rng``.
* Robot: ``torques``, proprioceptive Delta/Ring buffers, composite controller
  ``_applied_action_dict``, all mutable attributes of every part controller (OSC goal_pos /
  goal_ori / ref_* / J_* / mass_matrix / joint_pos / new_update / kp / kd / interpolators /
  ori_ref / relative_ori; gripper goal_qvel / joint_pos / ...), and each gripper model's
  ``current_action`` (Panda gripper integrates the commanded action across steps).
* Wrappers: MultiStepWrapper obs deque, reward list, done list, info deques, pending
  ``_seed``; VideoRecordingWrapper ``step_count``; RobomimicLowdimWrapper ``_seed``,
  ``init_state``.  MultiStepWrapper has no action queue: it executes the whole chunk in
  ``step``.  Any policy-side queue is owned by the caller and must be empty at a replan
  boundary (``pending_actions`` is recorded and returned).
* Process-global RNGs (numpy legacy + python ``random``) are captured in a separate
  namespace and restored only on request: stepping this env does not consume them; only
  ``reset()`` does.

Restore never steps physics.  It calls ``mj_setState``, ``mj_forward`` (derived quantities), then
``mj_setState`` again, because in mujoco 3.2.3 ``mj_forward`` overwrites ``qacc_warmstart``.
The integration state after restore is asserted bit-equal to the snapshot.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

SNAPSHOT_SCHEMA_VERSION = "2.0"

# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------


@dataclass
class Snapshot:
    """Versioned, picklable full-state snapshot (plain numpy / python containers only)."""

    schema_version: str
    physics: dict  # {"spec", "state", named arrays, "model_fingerprint"}
    task_env: dict  # robosuite env bookkeeping, observables, obs cache, env rng
    robots: list  # per-robot controller / gripper / buffer state
    wrappers: dict  # multistep / video / lowdim wrapper state
    global_rng: dict  # {"numpy": np.random.get_state(), "python": random.getstate()}
    pending_actions: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---- integrity -------------------------------------------------------------------
    def digest(self) -> str:
        """Stable sha256 over the restorable content (excludes meta)."""
        h = hashlib.sha256()
        _hash_into(h, self.schema_version)
        _hash_into(h, self.physics)
        _hash_into(h, self.task_env)
        _hash_into(h, self.robots)
        _hash_into(h, self.wrappers)
        _hash_into(h, self.pending_actions)
        return h.hexdigest()

    # ---- serialization ---------------------------------------------------------------
    def to_bytes(self) -> bytes:
        payload = {"schema_version": self.schema_version, "digest": self.digest(), "snapshot": self}
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_bytes(data: bytes) -> "Snapshot":
        payload = pickle.loads(data)
        snap = payload["snapshot"]
        if payload["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"snapshot schema {payload['schema_version']} != {SNAPSHOT_SCHEMA_VERSION}"
            )
        if snap.digest() != payload["digest"]:
            raise ValueError("snapshot digest mismatch (corrupt or tampered file)")
        return snap

    def serialize(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(self.to_bytes())
        tmp.replace(path)

    @staticmethod
    def deserialize(path: str | Path) -> "Snapshot":
        return Snapshot.from_bytes(Path(path).read_bytes())


@dataclass
class ObservationHistory:
    """What the policy needs after a restore (no physics stepping involved)."""

    obs: Any  # stacked (n_obs_steps, obs_dim) exactly as MultiStepWrapper returns
    n_real_frames: int  # number of real (non-padded) frames in the deque
    elapsed_steps: int  # env steps taken in the episode so far
    pending_actions: list  # caller-owned action queue (empty at a replan boundary)
    done: bool


# --------------------------------------------------------------------------------------
# Layer discovery
# --------------------------------------------------------------------------------------


def _iter_layers(obj):
    seen = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        yield obj
        obj = obj.__dict__.get("env") if hasattr(obj, "__dict__") else None


def find_layers(env) -> dict:
    """Locate each layer of the wrapper stack without relying on __getattr__ forwarding."""
    from robosuite.environments.base import MujocoEnv

    layers = {"multistep": None, "video": None, "lowdim": None, "robomimic": None, "robosuite": None}
    for layer in _iter_layers(env):
        name = type(layer).__name__
        if name == "MultiStepWrapper" and layers["multistep"] is None:
            layers["multistep"] = layer
        elif name == "VideoRecordingWrapper" and layers["video"] is None:
            layers["video"] = layer
        elif name in ("RobomimicLowdimWrapper", "RobomimicImageWrapper") and layers["lowdim"] is None:
            layers["lowdim"] = layer
        elif name == "EnvRobosuite" and layers["robomimic"] is None:
            layers["robomimic"] = layer
        elif isinstance(layer, MujocoEnv) and layers["robosuite"] is None:
            layers["robosuite"] = layer
    if layers["robosuite"] is None:
        raise ValueError(f"no robosuite MujocoEnv found under {type(env).__name__}")
    return layers


# --------------------------------------------------------------------------------------
# Generic attribute capture helpers
# --------------------------------------------------------------------------------------

_SCALAR_TYPES = (bool, int, float, str, type(None), np.bool_, np.integer, np.floating)


def _is_mujoco_view(arr: np.ndarray) -> bool:
    """True if arr aliases memory owned by a mujoco object (not a plain numpy array)."""
    base = arr
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    return not isinstance(base, (np.ndarray, bytes, bytearray, memoryview, type(None)))


def _plain(value, depth=0):
    """Return a deep copy if value is plain data (numpy / scalars / containers of them).

    Returns the sentinel _SKIP for anything else (sim handles, models, callables).
    """
    if depth > 6:
        return _SKIP
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            # e.g. robosuite pushes robot.torques (None) into recent_torques -> array(None)
            if all(isinstance(x, _SCALAR_TYPES) for x in value.flat):
                return copy.deepcopy(value)
            return _SKIP
        if _is_mujoco_view(value):
            return _SKIP
        return np.array(value, copy=True)
    if isinstance(value, _SCALAR_TYPES):
        return value
    if isinstance(value, (list, tuple)):
        out = [_plain(v, depth + 1) for v in value]
        if any(v is _SKIP for v in out):
            return _SKIP
        return type(value)(out) if isinstance(value, tuple) else out
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            pv = _plain(v, depth + 1)
            if pv is _SKIP:
                return _SKIP
            out[k] = pv
        return out
    return _SKIP


class _Skip:
    def __repr__(self):
        return "<SKIP>"


_SKIP = _Skip()

# Attributes that must never be captured / overwritten (handles into the live sim).
_NEVER = {"sim", "model", "data", "robot_model", "grippers", "viewer", "_sensor", "_corrupter",
          "_filter", "_delayer", "joint_action_policy"}


def _capture_attrs(obj, extra_objects=("interpolator", "interpolator_pos", "interpolator_ori")) -> dict:
    out = {}
    for k, v in obj.__dict__.items():
        if k in _NEVER:
            continue
        if k in extra_objects and v is not None:
            out[k] = ("__deepcopy__", copy.deepcopy(v))
            continue
        pv = _plain(v)
        if pv is not _SKIP:
            out[k] = pv
    return out


def _restore_attrs(obj, state: dict) -> None:
    for k, v in state.items():
        if isinstance(v, tuple) and len(v) == 2 and v[0] == "__deepcopy__":
            setattr(obj, k, copy.deepcopy(v[1]))
        else:
            setattr(obj, k, copy.deepcopy(v))


def _hash_into(h, obj) -> None:
    if isinstance(obj, np.ndarray):
        h.update(str(obj.dtype).encode())
        h.update(str(obj.shape).encode())
        if obj.dtype == object:
            for x in obj.flat:
                _hash_into(h, x)
        else:
            h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            h.update(str(k).encode())
            _hash_into(h, obj[k])
    elif isinstance(obj, (list, tuple, deque)):
        h.update(f"{type(obj).__name__}{len(obj)}".encode())
        for v in obj:
            _hash_into(h, v)
    elif isinstance(obj, (float, np.floating)):
        h.update(np.float64(obj).tobytes())
    elif isinstance(obj, (bool, int, str, type(None), np.bool_, np.integer)):
        h.update(f"{type(obj).__name__}:{obj!r}".encode())
    elif hasattr(obj, "__dict__"):
        h.update(type(obj).__name__.encode())
        _hash_into(h, {k: v for k, v in vars(obj).items() if k not in _NEVER})
    else:
        h.update(repr(obj).encode())


# --------------------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------------------


def _mj(rs):
    return rs.sim.model._model, rs.sim.data._data


def model_fingerprint(rs) -> str:
    """Hash of model quantities that could be mutated at runtime (mass, friction, gains...)."""
    m, _ = _mj(rs)
    h = hashlib.sha256()
    for name in ("nq", "nv", "na", "nu", "nmocap", "nbody", "ngeom"):
        h.update(f"{name}={getattr(m, name)}".encode())
    for name in ("body_mass", "body_inertia", "body_pos", "body_quat", "geom_friction",
                 "geom_size", "geom_pos", "dof_damping", "dof_armature", "dof_frictionloss",
                 "jnt_range", "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange"):
        _hash_into(h, np.asarray(getattr(m, name)))
    _hash_into(h, np.array([m.opt.timestep, m.opt.integrator, m.opt.cone, m.opt.solver,
                            m.opt.iterations, m.opt.tolerance], dtype=np.float64))
    return h.hexdigest()


def _capture_physics(rs) -> dict:
    import mujoco

    m, d = _mj(rs)
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(m, spec), dtype=np.float64)
    mujoco.mj_getState(m, d, state, spec)
    return {
        "spec": spec,
        "state": state,
        # named copies (diagnostics only; restore uses "state")
        "time": float(d.time),
        "qpos": d.qpos.copy(),
        "qvel": d.qvel.copy(),
        "act": d.act.copy(),
        "qacc_warmstart": d.qacc_warmstart.copy(),
        "ctrl": d.ctrl.copy(),
        "qfrc_applied": d.qfrc_applied.copy(),
        "xfrc_applied": d.xfrc_applied.copy(),
        "mocap_pos": d.mocap_pos.copy(),
        "mocap_quat": d.mocap_quat.copy(),
        "model_fingerprint": model_fingerprint(rs),
    }


def get_integration_state(rs) -> np.ndarray:
    import mujoco

    m, d = _mj(rs)
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    out = np.empty(mujoco.mj_stateSize(m, spec), dtype=np.float64)
    mujoco.mj_getState(m, d, out, spec)
    return out


def _restore_physics(state: dict, rs, check_model: bool = True) -> None:
    import mujoco

    if check_model:
        fp = model_fingerprint(rs)
        if fp != state["model_fingerprint"]:
            raise ValueError("model fingerprint mismatch: target env has a different MjModel")
    m, d = _mj(rs)
    spec = state["spec"]
    if mujoco.mj_stateSize(m, spec) != state["state"].size:
        raise ValueError("mjState size mismatch between snapshot and target model")
    mujoco.mj_setState(m, d, state["state"], spec)
    # mj_forward refreshes derived quantities (xpos, site poses, qM, ...) but in mujoco 3.2.3 it
    # also overwrites qacc_warmstart with the freshly solved qacc (measured: up to 1.2e-2).
    # The next env.step starts with sim.forward(), whose solver result depends on the incoming
    # warmstart, so re-apply the saved integration state after the forward.
    mujoco.mj_forward(m, d)
    mujoco.mj_setState(m, d, state["state"], spec)
    after = np.empty_like(state["state"])
    mujoco.mj_getState(m, d, after, spec)
    if not np.array_equal(after, state["state"]):
        raise RuntimeError("integration state differs from snapshot after restore")


# --------------------------------------------------------------------------------------
# robosuite env / robot / controllers
# --------------------------------------------------------------------------------------

# Env attributes that are configuration rather than episode state; never overwritten.
_ENV_CONFIG_KEYS = {"horizon", "ignore_done", "hard_reset", "control_freq", "control_timestep",
                    "model_timestep", "seed", "lite_physics", "has_renderer",
                    "has_offscreen_renderer", "renderer", "render_camera", "use_camera_obs",
                    "camera_names", "camera_heights", "camera_widths", "camera_depths",
                    "camera_segmentations", "render_gpu_device_id", "reward_scale",
                    "reward_shaping", "use_object_obs", "single_object_mode", "robot_names",
                    "robot_configs", "env_configuration", "num_robots", "num_cameras",
                    "renderer_config", "render_collision_mesh", "render_visual_mesh",
                    "deterministic_reset", "table_full_size", "table_friction", "table_offset"}

_OBSERVABLE_KEYS = ("_time_since_last_sample", "_current_delay", "_current_observed_value",
                    "_sampled", "_enabled", "_active")


def _capture_task_env(rs) -> dict:
    attrs = {}
    for k, v in rs.__dict__.items():
        if k in _ENV_CONFIG_KEYS or k in _NEVER or k in ("_obs_cache", "_observables", "rng"):
            continue
        pv = _plain(v)
        if pv is not _SKIP:
            attrs[k] = pv
    observables = {
        name: {k: copy.deepcopy(getattr(o, k)) for k in _OBSERVABLE_KEYS}
        for name, o in rs._observables.items()
    }
    return {
        "attrs": attrs,
        "obs_cache": copy.deepcopy(rs._obs_cache),
        "observables": observables,
        "rng": copy.deepcopy(rs.rng.bit_generator.state) if hasattr(rs, "rng") else None,
    }


def _restore_task_env(state: dict, rs) -> None:
    for k, v in state["attrs"].items():
        setattr(rs, k, copy.deepcopy(v))
    rs._obs_cache = copy.deepcopy(state["obs_cache"])
    missing = set(state["observables"]) ^ set(rs._observables)
    if missing:
        raise ValueError(f"observable set differs between snapshot and env: {sorted(missing)}")
    for name, ostate in state["observables"].items():
        o = rs._observables[name]
        for k, v in ostate.items():
            setattr(o, k, copy.deepcopy(v))
    if state["rng"] is not None:
        rs.rng.bit_generator.state = copy.deepcopy(state["rng"])


_ROBOT_BUFFERS = ("recent_qpos", "recent_actions", "recent_torques", "recent_ee_forcetorques",
                  "recent_ee_pose", "recent_ee_vel", "recent_ee_vel_buffer", "recent_ee_acc")


def _buffer_state(buf):
    """robosuite Delta/RingBuffer (or dict arm -> buffer) as plain data."""
    if isinstance(buf, dict):
        return {arm: _buffer_state(b) for arm, b in buf.items()}
    if buf is None:
        return None
    st = _plain(buf.__dict__)
    if st is _SKIP:
        raise ValueError(f"unsupported buffer state in {type(buf).__name__}")
    return {"__buffer__": type(buf).__name__, "state": st}


def _restore_buffer(buf, state) -> None:
    if state is None:
        return
    if "__buffer__" not in state:
        for arm, s in state.items():
            _restore_buffer(buf[arm], s)
        return
    if type(buf).__name__ != state["__buffer__"]:
        raise ValueError(f"buffer type mismatch {type(buf).__name__} vs {state['__buffer__']}")
    for k, v in state["state"].items():
        setattr(buf, k, copy.deepcopy(v))


def _capture_robot(robot) -> dict:
    cc = robot.composite_controller
    return {
        "torques": _plain(robot.torques),
        "buffers": {k: _buffer_state(getattr(robot, k)) for k in _ROBOT_BUFFERS},
        "enabled_parts": dict(robot._enabled_parts),
        "composite": {"_applied_action_dict": _plain(dict(cc._applied_action_dict))},
        "parts": {name: _capture_attrs(c) for name, c in cc.part_controllers.items()},
        "grippers": {arm: _plain(g.current_action) for arm, g in robot.gripper.items()
                     if hasattr(g, "current_action")},
    }


def _restore_robot(state: dict, robot) -> None:
    cc = robot.composite_controller
    robot.torques = copy.deepcopy(state["torques"])
    for k, v in state["buffers"].items():
        _restore_buffer(getattr(robot, k), v)
    robot._enabled_parts = dict(state["enabled_parts"])
    cc._applied_action_dict.clear()
    cc._applied_action_dict.update(copy.deepcopy(state["composite"]["_applied_action_dict"]))
    if set(state["parts"]) != set(cc.part_controllers):
        raise ValueError("part controller set differs between snapshot and env")
    for name, cstate in state["parts"].items():
        _restore_attrs(cc.part_controllers[name], cstate)
    for arm, ca in state["grippers"].items():
        robot.gripper[arm].current_action = np.array(ca, copy=True)


# --------------------------------------------------------------------------------------
# Wrappers
# --------------------------------------------------------------------------------------


def _capture_wrappers(layers) -> dict:
    out = {}
    ms = layers["multistep"]
    if ms is not None:
        out["multistep"] = {
            "obs": [copy.deepcopy(o) for o in ms.obs],
            "obs_maxlen": ms.obs.maxlen,
            "reward": copy.deepcopy(list(ms.reward)),
            "done": copy.deepcopy(list(ms.done)),
            "info": {k: copy.deepcopy(list(v)) for k, v in ms.info.items()},
            "info_maxlen": ms.n_obs_steps + 1,
            "_seed": ms.__dict__.get("_seed", None),
            "has_seed_attr": "_seed" in ms.__dict__,
            "n_obs_steps": ms.n_obs_steps,
            "n_action_steps": ms.n_action_steps,
            "max_episode_steps": ms.max_episode_steps,
        }
    if layers["video"] is not None:
        out["video"] = {"step_count": layers["video"].step_count}
    lw = layers["lowdim"]
    if lw is not None:
        out["lowdim"] = {"_seed": lw.__dict__.get("_seed"),
                         "init_state": _plain(lw.__dict__.get("init_state"))}
    return out


def _restore_wrappers(state: dict, layers) -> None:
    ms = layers["multistep"]
    if "multistep" in state:
        if ms is None:
            raise ValueError("snapshot has MultiStepWrapper state but target has no MultiStepWrapper")
        s = state["multistep"]
        if (s["n_obs_steps"], s["n_action_steps"], s["max_episode_steps"]) != (
                ms.n_obs_steps, ms.n_action_steps, ms.max_episode_steps):
            raise ValueError("MultiStepWrapper configuration differs between snapshot and target")
        ms.obs = deque(copy.deepcopy(s["obs"]), maxlen=s["obs_maxlen"])
        ms.reward = copy.deepcopy(s["reward"])
        ms.done = copy.deepcopy(s["done"])
        maxlen = s["info_maxlen"]
        ms.info = defaultdict(lambda: deque(maxlen=maxlen))
        for k, v in s["info"].items():
            ms.info[k] = deque(copy.deepcopy(v), maxlen=maxlen)
        if s["has_seed_attr"]:
            ms._seed = s["_seed"]
        elif "_seed" in ms.__dict__:
            del ms.__dict__["_seed"]
    if "video" in state and layers["video"] is not None:
        layers["video"].step_count = state["video"]["step_count"]
    if "lowdim" in state and layers["lowdim"] is not None:
        layers["lowdim"]._seed = state["lowdim"]["_seed"]
        layers["lowdim"].init_state = copy.deepcopy(state["lowdim"]["init_state"])


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def capture_full_state(env, wrapper=None, history=None, pending_actions=None) -> Snapshot:
    """Capture everything needed to continue ``env`` bit-exactly.

    Args:
        env: any layer of the make_env stack (typically the MultiStepWrapper it returns).
        wrapper: optional; layer to search for wrappers from (defaults to ``env``).
        history: optional stacked observation (n_obs_steps, obs_dim) the caller holds; if given
            it is checked against the wrapper's own history and a mismatch raises.
        pending_actions: caller-owned action queue. Must be empty/None at a replan boundary;
            it is stored and returned verbatim by restore.
    """
    layers = find_layers(wrapper if wrapper is not None else env)
    rs = layers["robosuite"]
    snap = Snapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        physics=_capture_physics(rs),
        task_env=_capture_task_env(rs),
        robots=[_capture_robot(r) for r in rs.robots],
        wrappers=_capture_wrappers(layers),
        global_rng={"numpy": np.random.get_state(), "python": random.getstate()},
        pending_actions=[np.array(a, copy=True) for a in (pending_actions or [])],
        meta={"env_class": type(rs).__name__,
              "elapsed_steps": len(layers["multistep"].reward) if layers["multistep"] else None},
    )
    if history is not None and layers["multistep"] is not None:
        own = layers["multistep"]._get_obs(layers["multistep"].n_obs_steps)
        if not _tree_equal(own, history):
            raise ValueError("provided history does not match MultiStepWrapper observation history")
    return snap


def restore_full_state(snapshot: Snapshot, env, wrapper=None, restore_global_rng: bool = False,
                       check_model: bool = True) -> ObservationHistory:
    """Restore ``snapshot`` into ``env`` (same or different instance) without stepping physics.

    Returns the policy-facing observation history, exactly what MultiStepWrapper would return.
    Process-global RNGs are only restored when ``restore_global_rng`` is True (they are a
    separate namespace and stepping does not consume them).
    """
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"snapshot schema {snapshot.schema_version} unsupported")
    layers = find_layers(wrapper if wrapper is not None else env)
    rs = layers["robosuite"]
    if type(rs).__name__ != snapshot.meta.get("env_class", type(rs).__name__):
        raise ValueError(f"env class mismatch: {type(rs).__name__} vs {snapshot.meta['env_class']}")
    if len(rs.robots) != len(snapshot.robots):
        raise ValueError("robot count mismatch")
    _restore_physics(snapshot.physics, rs, check_model=check_model)
    _restore_task_env(snapshot.task_env, rs)
    for rstate, robot in zip(snapshot.robots, rs.robots):
        _restore_robot(rstate, robot)
    _restore_wrappers(snapshot.wrappers, layers)
    if restore_global_rng:
        np.random.set_state(snapshot.global_rng["numpy"])
        random.setstate(snapshot.global_rng["python"])

    ms = layers["multistep"]
    if ms is not None:
        obs = ms._get_obs(ms.n_obs_steps)
        return ObservationHistory(obs=obs, n_real_frames=len(ms.obs), elapsed_steps=len(ms.reward),
                                  pending_actions=[np.array(a, copy=True) for a in snapshot.pending_actions],
                                  done=bool(ms.done[-1]) if ms.done else False)
    return ObservationHistory(obs=None, n_real_frames=0, elapsed_steps=int(rs.timestep),
                              pending_actions=list(snapshot.pending_actions), done=bool(rs.done))


def python_state_digest(env) -> dict:
    """Per-component sha256 of every plain-data attribute in the stack (for audits/tests).

    Covers env attrs, observables, obs cache, robot buffers, every part controller, gripper
    models and the wrappers. Used to prove that no mutable state is left uncaptured.
    """
    layers = find_layers(env)
    rs = layers["robosuite"]
    comps = {}

    def put(name, obj):
        h = hashlib.sha256()
        _hash_into(h, obj)
        comps[name] = h.hexdigest()

    put("mj_integration", get_integration_state(rs))
    put("env_attrs", {k: v for k, v in ((k, _plain(v)) for k, v in rs.__dict__.items()
                                         if k not in _NEVER) if v is not _SKIP})
    put("obs_cache", rs._obs_cache)
    put("observables", {n: {k: getattr(o, k) for k in _OBSERVABLE_KEYS} for n, o in rs._observables.items()})
    put("env_rng", rs.rng.bit_generator.state)
    for i, r in enumerate(rs.robots):
        put(f"robot{i}_attrs", {k: v for k, v in ((k, _plain(v)) for k, v in r.__dict__.items()
                                                   if k not in _NEVER) if v is not _SKIP})
        put(f"robot{i}_buffers", {k: _buffer_state(getattr(r, k)) for k in _ROBOT_BUFFERS})
        for name, c in r.composite_controller.part_controllers.items():
            put(f"robot{i}_ctrl_{name}", {k: v for k, v in ((k, _plain(v)) for k, v in c.__dict__.items()
                                                            if k not in _NEVER) if v is not _SKIP})
        for arm, g in r.gripper.items():
            put(f"robot{i}_gripper_{arm}", getattr(g, "current_action", None))
    ms = layers["multistep"]
    if ms is not None:
        put("multistep", {"obs": list(ms.obs), "reward": ms.reward, "done": ms.done,
                          "info": {k: list(v) for k, v in ms.info.items()}})
    if layers["video"] is not None:
        put("video", layers["video"].step_count)
    return comps


def _tree_equal(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_tree_equal(a[k], b[k]) for k in a)
    return np.array_equal(np.asarray(a), np.asarray(b))


def snapshot_to_bytes(snapshot: Snapshot) -> bytes:
    return snapshot.to_bytes()


def snapshot_from_bytes(data: bytes) -> Snapshot:
    return Snapshot.from_bytes(data)
