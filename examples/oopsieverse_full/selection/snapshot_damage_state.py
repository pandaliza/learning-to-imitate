"""Full snapshot / restore of an OopsieVerse RoboCasa + DamageSim episode (spec §11, §15 `snapshot_damage_state.py`).

Stack (pinned): robosuite 1.5.2 (aaa8b9b), robocasa 97a4060, mujoco 3.3.1, oopsieverse a739f99.
    DamagePickEgg / DamageableWipeCounter (RSDamageableEnvironment + Kitchen + ManipulationEnv + MujocoEnv)
      robots[0] = DamageableWheeledRobot (PandaOmron) -> HybridMobileBase composite controller
            {right: OSC_POSE, right_gripper: SimpleGrip (+ continuous-gripper patch), base: legacy mobile-base
             joint velocity, torso: joint position}; base-mode bit = arm controller `_goal_update_mode`
      DamageSim: per tracked entity (egg|sponge, PandaOmron) link_healths / damage_info / track_damage /
                 damageable_links and one RSMechanicalDamageEvaluator each (prev_linear_velocities, logs ...).

Reference design: mip/intent_selection/snapshots.py (robosuite 1.5.1 fixed-base; verified there). Differences:
mobile base + torso part controllers, DamageSim state, model edits (wipe markers), Kitchen/wipe bookkeeping, the
policy-side observation history and the episode-protocol bookkeeping (supplied by the caller as `episode`).

Nothing here edits official files. Restore never steps physics.

Rebuild order for a *different* env instance (fresh env or fresh process), see `rebuild_and_restore`:
    1. canonical reset of the PARENT reset (reset_banks_v1: unset_ep_meta; env.rng=default_rng(seed); reset())
       -> same compiled model (reset_banks_v1 determinism T1/T2) and a freshly initialised DamageSim
       (evaluators + health); hide teleop eef markers (as the evaluator does);
    2. model check: structural fingerprint must equal the snapshot's;
    3. model edits (geom_rgba, body_pos, site_rgba, geom_size?no) written back in full;
    4. MuJoCo integration state: mj_setState -> mj_forward -> mj_setState (mj_forward overwrites
       qacc_warmstart), asserted bit-equal;
    5. env python state, observables + obs cache, env.rng;
    6. robots: buffers, torques, composite + every part controller, gripper current_action;
    7. DamageSim: per entity health/info/flags, per evaluator full state (prev velocities, logs, caches,
       alias contacts_by_body -> contacts_by_part), env lock/init flags, health_list_link_names;
    8. wipe: wiped marker indices (objects re-bound), ee_force_bias, marker positions;
    9. process-global numpy / python RNG (stepping does not consume them; restored anyway).
Same-instance restore (branching inside one worker) runs steps 2-9 without the reset.
"""

from __future__ import annotations

import copy
import hashlib
import io
import os
import pickle
import random
from collections import deque

import numpy as np

SCHEMA = "ovsnap-1.0"

# ───────────────────────────── plain-data helpers ─────────────────────────────

_SCALARS = (bool, int, float, str, bytes, type(None), np.bool_, np.integer, np.floating)


class _Skip:
    def __repr__(self):
        return "<SKIP>"


SKIP = _Skip()


def _is_mj_view(a):
    b = a
    while isinstance(b, np.ndarray) and b.base is not None:
        b = b.base
    return not isinstance(b, (np.ndarray, bytes, bytearray, memoryview, type(None)))


def plain(v, depth=0):
    """Deep copy of plain data (numpy / scalars / containers); SKIP for anything else."""
    if depth > 12:
        return SKIP
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            if all(isinstance(x, _SCALARS) for x in v.flat):
                return copy.deepcopy(v)
            return SKIP
        if _is_mj_view(v):
            return SKIP
        return np.array(v, copy=True)
    if isinstance(v, _SCALARS):
        return v
    if isinstance(v, (list, tuple)):
        out = [plain(x, depth + 1) for x in v]
        if any(x is SKIP for x in out):
            return SKIP
        return tuple(out) if isinstance(v, tuple) else out
    if isinstance(v, deque):
        out = [plain(x, depth + 1) for x in v]
        if any(x is SKIP for x in out):
            return SKIP
        return deque(out, maxlen=v.maxlen)
    if isinstance(v, dict):
        out = type(v)() if type(v) in (dict,) else {}
        for k, x in v.items():
            px = plain(x, depth + 1)
            if px is SKIP:
                return SKIP
            out[k] = px
        return out
    return SKIP


def hash_into(h, o):
    if isinstance(o, np.ndarray):
        h.update(f"nd{o.dtype}{o.shape}".encode())
        if o.dtype == object:
            for x in o.flat:
                hash_into(h, x)
        else:
            h.update(np.ascontiguousarray(o).tobytes())
    elif isinstance(o, dict):
        h.update(f"d{len(o)}".encode())
        for k in sorted(o, key=str):
            h.update(str(k).encode())
            hash_into(h, o[k])
    elif isinstance(o, (list, tuple, deque)):
        h.update(f"{type(o).__name__}{len(o)}".encode())
        for x in o:
            hash_into(h, x)
    elif isinstance(o, (float, np.floating)):
        h.update(np.float64(o).tobytes())
    elif isinstance(o, (bool, int, str, bytes, type(None), np.bool_, np.integer)):
        h.update(f"{type(o).__name__}:{o!r}".encode())
    else:
        h.update(repr(o).encode())


def sha(o, n=16):
    h = hashlib.sha256()
    hash_into(h, o)
    return h.hexdigest()[:n]


# Attributes that are handles into the live sim / model objects, never captured or overwritten.
_NEVER = {"sim", "model", "data", "robot_model", "grippers", "gripper", "viewer", "_sensor", "_corrupter",
          "_filter", "_delayer", "entity", "damage_evaluators", "composite_controller", "part_controllers",
          "mujoco_arena", "objects", "fixtures", "markers", "wiped_markers", "placement_initializer",
          "_observables", "_obs_cache", "rng", "counter", "sink", "init_robot_base_ref", "robots",
          "fixture_refs", "object_placements", "fxtr_placements", "_xml_processors", "obj_registries",
          "_health_fig", "_health_ax", "_health_bars_dict", "_damage_color_manager", "damageable_robots",
          "sim_state_initial"}
# env configuration (constructor arguments) — not episode state
_ENV_CONFIG = {"horizon", "ignore_done", "hard_reset", "control_freq", "control_timestep", "model_timestep", "seed",
               "lite_physics", "has_renderer", "has_offscreen_renderer", "renderer", "render_camera", "use_camera_obs",
               "camera_names", "camera_heights", "camera_widths", "camera_depths", "camera_segmentations",
               "render_gpu_device_id", "reward_scale", "reward_shaping", "use_object_obs", "robot_names",
               "robot_configs", "env_configuration", "num_robots", "num_cameras", "renderer_config",
               "render_collision_mesh", "render_visual_mesh", "deterministic_reset", "render_segmentation",
               "low_dim", "use_external_camera", "translucent_robot", "load_model_on_init", "_cam_configs"}
_INTERP = ("interpolator", "interpolator_pos", "interpolator_ori")
_OBSERVABLE_KEYS = ("_time_since_last_sample", "_current_delay", "_current_observed_value", "_sampled",
                    "_enabled", "_active")
_ROBOT_BUFFERS = ("recent_qpos", "recent_actions", "recent_torques", "recent_ee_forcetorques", "recent_ee_pose",
                  "recent_ee_vel", "recent_ee_vel_buffer", "recent_ee_acc")


def _capture_attrs(obj, never=_NEVER, extra_skip=()):
    out, skipped = {}, []
    for k, v in obj.__dict__.items():
        if k in never or k in extra_skip:
            continue
        if k in _INTERP and v is not None:
            out[k] = ("__deepcopy__", copy.deepcopy(v))
            continue
        pv = plain(v)
        if pv is SKIP:
            skipped.append(k)
        else:
            out[k] = pv
    return out, skipped


def _restore_attrs(obj, st):
    """Write back into obj.__dict__ (where capture read from); bypasses read-only properties that shadow
    instance attributes (e.g. RoboCasa fixture `pos`)."""
    d = obj.__dict__
    for k, v in st.items():
        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str) and v[0] == "__deepcopy__":
            d[k] = copy.deepcopy(v[1])
        else:
            d[k] = copy.deepcopy(v)


# ───────────────────────────── physics + model ─────────────────────────────

_MODEL_EDIT_FIELDS = ("geom_rgba", "body_pos", "body_quat", "site_rgba", "geom_pos", "geom_quat", "geom_size",
                      "geom_contype", "geom_conaffinity", "body_mass", "geom_friction")
_MODEL_STRUCT_FIELDS = ("body_inertia", "dof_damping", "dof_armature", "dof_frictionloss", "jnt_range",
                        "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "geom_bodyid", "geom_type",
                        "body_parentid", "jnt_type", "geom_solref", "geom_solimp", "geom_margin", "eq_data")


def _mj(env):
    return env.sim.model._model, env.sim.data._data


def model_structure_fp(env):
    m, _ = _mj(env)
    h = hashlib.sha256()
    for n in ("nq", "nv", "na", "nu", "nmocap", "nbody", "ngeom", "nsite", "neq", "nsensordata"):
        h.update(f"{n}={getattr(m, n)}".encode())
    for n in _MODEL_STRUCT_FIELDS:
        hash_into(h, np.asarray(getattr(m, n)))
    hash_into(h, np.array([m.opt.timestep, m.opt.integrator, m.opt.cone, m.opt.solver, m.opt.iterations,
                           m.opt.tolerance, m.opt.noslip_iterations], dtype=np.float64))
    return h.hexdigest()[:24]


def model_edits(env):
    m, _ = _mj(env)
    return {n: np.array(getattr(m, n), copy=True) for n in _MODEL_EDIT_FIELDS}


def integration_state(env):
    import mujoco
    m, d = _mj(env)
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    s = np.empty(mujoco.mj_stateSize(m, spec), np.float64)
    mujoco.mj_getState(m, d, s, spec)
    return s


def _restore_physics(env, ph):
    import mujoco
    m, d = _mj(env)
    spec = ph["spec"]
    if mujoco.mj_stateSize(m, spec) != ph["state"].size:
        raise ValueError("mjState size mismatch")
    mujoco.mj_setState(m, d, ph["state"], spec)
    mujoco.mj_forward(m, d)
    mujoco.mj_setState(m, d, ph["state"], spec)
    if not np.array_equal(integration_state(env), ph["state"]):
        raise RuntimeError("integration state differs after restore")


# ───────────────────────────── DamageSim ─────────────────────────────

def damage_entities(env):
    """All DamageableMixin objects reachable through env._get_all_objects(), keyed by name (tracked or not)."""
    from damagesim.core.damageable_mixin import DamageableMixin
    out = {}
    for o in env._get_all_objects():
        if isinstance(o, DamageableMixin):
            if o.name in out and out[o.name] is not o:
                raise ValueError(f"duplicate damageable entity name {o.name}")
            out[o.name] = o
    return out


_ENTITY_KEYS = ("track_damage", "damageable_links", "link_healths", "damage_info", "damage_params", "control_freq")


def _capture_damage(env):
    ents = {}
    for name, o in damage_entities(env).items():
        e = {k: plain(getattr(o, k)) for k in _ENTITY_KEYS if hasattr(o, k)}
        for k, v in e.items():
            if v is SKIP:
                raise ValueError(f"DamageSim entity {name}.{k} is not plain data")
        evs = []
        for ev in o.damage_evaluators:
            st, skipped = _capture_attrs(ev, never={"entity", "contacts_by_body"})
            if skipped:
                raise ValueError(f"evaluator {name}/{type(ev).__name__} has non-plain state {skipped}")
            evs.append({"class": type(ev).__name__, "state": st,
                        "body_alias": getattr(ev, "contacts_by_body", None) is getattr(ev, "contacts_by_part", 0)})
        e["evaluators"] = evs
        ents[name] = e
    return {"entities": ents, "lock_health": bool(env.lock_health),
            "damage_evaluators_initialized": bool(env.damage_evaluators_initialized),
            "health_list_link_names": list(env.health_list_link_names),
            "_damage_config": copy.deepcopy(env._damage_config)}


def _restore_damage(env, st):
    ents = damage_entities(env)
    if set(ents) != set(st["entities"]):
        raise ValueError(f"damage entity set differs: {sorted(ents)} vs {sorted(st['entities'])}")
    for name, e in st["entities"].items():
        o = ents[name]
        for k in _ENTITY_KEYS:
            if k in e:
                setattr(o, k, copy.deepcopy(e[k]))
        if len(o.damage_evaluators) != len(e["evaluators"]):
            raise ValueError(f"{name}: evaluator count differs")
        for ev, es in zip(o.damage_evaluators, e["evaluators"]):
            if type(ev).__name__ != es["class"]:
                raise ValueError(f"{name}: evaluator class differs")
            _restore_attrs(ev, es["state"])
            if es["body_alias"]:
                ev.contacts_by_body = ev.contacts_by_part  # keep the alias the evaluator relies on
    env.lock_health = st["lock_health"]
    env.damage_evaluators_initialized = st["damage_evaluators_initialized"]
    env.health_list_link_names = list(st["health_list_link_names"])
    env._damage_config = copy.deepcopy(st["_damage_config"])


# ───────────────────────────── env / robots ─────────────────────────────

def _capture_env(env):
    attrs, skipped = _capture_attrs(env, never=_NEVER | _ENV_CONFIG)
    # DamageSim fields live on env too; they are captured (and restored) by _capture_damage
    for k in ("lock_health", "damage_evaluators_initialized", "health_list_link_names", "_damage_config"):
        attrs.pop(k, None)
    obsv = {n: {k: copy.deepcopy(getattr(o, k)) for k in _OBSERVABLE_KEYS} for n, o in env._observables.items()}
    wipe = None
    if hasattr(env, "markers") and hasattr(env, "wiped_markers"):
        idx = {id(m): i for i, m in enumerate(env.markers)}
        wipe = {"wiped_idx": [idx[id(m)] for m in env.wiped_markers], "n_markers": len(env.markers)}
    fixtures = {}
    for name, fx in _fixtures(env).items():
        st, _sk = _capture_attrs(fx)
        fixtures[name] = st
    return {"attrs": attrs, "skipped": skipped, "obs_cache": copy.deepcopy(env._obs_cache), "observables": obsv,
            "rng": copy.deepcopy(env.rng.bit_generator.state), "wipe": wipe, "fixtures": fixtures}


def _fixtures(env):
    """RoboCasa fixture objects (env.fixtures and mujoco_arena.fixtures; usually the same instances). Their python
    state (door / timer / temperature / toaster state ...) is updated every step by Kitchen.update_state."""
    out = {}
    for src in (getattr(env, "fixtures", None), getattr(getattr(env, "mujoco_arena", None), "fixtures", None)):
        if isinstance(src, dict):
            for k, fx in src.items():
                if k in out and out[k] is not fx:
                    out[f"arena::{k}"] = fx
                else:
                    out[k] = fx
    return out


def _restore_env(env, st):
    _restore_attrs(env, st["attrs"])
    env._obs_cache = copy.deepcopy(st["obs_cache"])
    if set(st["observables"]) != set(env._observables):
        raise ValueError("observable set differs")
    for n, os_ in st["observables"].items():
        for k, v in os_.items():
            setattr(env._observables[n], k, copy.deepcopy(v))
    env.rng.bit_generator.state = copy.deepcopy(st["rng"])
    fxs = _fixtures(env)
    if set(fxs) != set(st["fixtures"]):
        raise ValueError("fixture set differs")
    for name, fst in st["fixtures"].items():
        _restore_attrs(fxs[name], fst)
    if st["wipe"] is not None:
        if len(env.markers) != st["wipe"]["n_markers"]:
            raise ValueError("marker count differs")
        env.wiped_markers = [env.markers[i] for i in st["wipe"]["wiped_idx"]]


def _buf(b):
    if isinstance(b, dict):
        return {k: _buf(x) for k, x in b.items()}
    if b is None:
        return None
    s = plain(b.__dict__)
    if s is SKIP:
        raise ValueError(f"unsupported buffer {type(b).__name__}")
    return {"__buffer__": type(b).__name__, "state": s}


def _rbuf(b, s):
    if s is None:
        return
    if "__buffer__" not in s:
        for k, x in s.items():
            _rbuf(b[k], x)
        return
    if type(b).__name__ != s["__buffer__"]:
        raise ValueError("buffer type differs")
    for k, v in s["state"].items():
        setattr(b, k, copy.deepcopy(v))


_ROBOT_DAMAGE_KEYS = set(_ENTITY_KEYS)


def _capture_robot(r):
    cc = r.composite_controller
    attrs, skipped = _capture_attrs(r, never=_NEVER | set(_ROBOT_BUFFERS) | _ROBOT_DAMAGE_KEYS)
    parts = {}
    for n, c in cc.part_controllers.items():
        st, sk = _capture_attrs(c)
        parts[n] = {"state": st, "skipped": sk}
    cc_st, cc_sk = _capture_attrs(cc)
    return {"attrs": attrs, "skipped": skipped, "buffers": {k: _buf(getattr(r, k)) for k in _ROBOT_BUFFERS},
            "composite": cc_st, "composite_skipped": cc_sk, "parts": parts,
            "grippers": {a: plain(g.current_action) for a, g in r.gripper.items() if hasattr(g, "current_action")}}


def _restore_robot(r, st):
    cc = r.composite_controller
    _restore_attrs(r, st["attrs"])
    for k, v in st["buffers"].items():
        _rbuf(getattr(r, k), v)
    _restore_attrs(cc, st["composite"])
    if set(st["parts"]) != set(cc.part_controllers):
        raise ValueError("part controller set differs")
    for n, ps in st["parts"].items():
        _restore_attrs(cc.part_controllers[n], ps["state"])
    for a, ca in st["grippers"].items():
        r.gripper[a].current_action = None if ca is None else np.array(ca, copy=True)


# ───────────────────────────── public API ─────────────────────────────

def capture(env, episode=None, meta=None):
    """Snapshot dict (plain data only). `episode` = caller-owned policy/protocol state (obs history frames,
    step counter, success latch, running maxima ...), stored verbatim and returned by restore."""
    import mujoco
    snap = {
        "schema": SCHEMA,
        "env_class": type(env).__name__,
        "model_fp": model_structure_fp(env),
        "model_edits": model_edits(env),
        "physics": {"spec": int(mujoco.mjtState.mjSTATE_INTEGRATION), "state": integration_state(env),
                    "qpos": env.sim.data.qpos.copy(), "time": float(env.sim.data.time)},
        "env": _capture_env(env),
        "robots": [_capture_robot(r) for r in env.robots],
        "damage": _capture_damage(env),
        "global_rng": {"numpy": np.random.get_state(), "python": random.getstate()},
        "episode": copy.deepcopy(episode),
        "meta": dict(meta or {}),
    }
    return snap


def restore(env, snap, restore_global_rng=True):
    """In-place restore into an env that holds the SAME compiled model (same instance, or rebuilt by
    rebuild_and_restore). Returns the caller's `episode` state (deep copy)."""
    if snap["schema"] != SCHEMA:
        raise ValueError(f"schema {snap['schema']} != {SCHEMA}")
    if type(env).__name__ != snap["env_class"]:
        raise ValueError("env class differs")
    if model_structure_fp(env) != snap["model_fp"]:
        raise ValueError("model structure fingerprint differs: env holds a different compiled model")
    m, _ = _mj(env)
    for n, arr in snap["model_edits"].items():
        getattr(m, n)[...] = arr
    _restore_physics(env, snap["physics"])
    _restore_env(env, snap["env"])
    if len(env.robots) != len(snap["robots"]):
        raise ValueError("robot count differs")
    for r, rs in zip(env.robots, snap["robots"]):
        _restore_robot(r, rs)
    _restore_damage(env, snap["damage"])
    if restore_global_rng:
        np.random.set_state(snap["global_rng"]["numpy"])
        random.setstate(snap["global_rng"]["python"])
    return copy.deepcopy(snap["episode"])


def canonical_reset(env, seed):
    """reset_banks_v1 canonical reset + evaluator marker hiding (baseline/rollout_eval.apply_reset)."""
    env.unset_ep_meta()
    env.rng = np.random.default_rng(int(seed))
    env.reset()
    for robot in env.robots:
        for arm in robot.arms:
            if robot.eef_site_id[arm] is not None:
                env.sim.model.site_rgba[robot.eef_site_id[arm]] = np.zeros(4)
            if robot.eef_cylinder_id[arm] is not None:
                env.sim.model.site_rgba[robot.eef_cylinder_id[arm]] = np.zeros(4)


def rebuild_and_restore(env, snap, restore_global_rng=True):
    """Restore into ANY env instance of the task: rebuild the parent's compiled model by the canonical reset of
    the parent seed (snap['meta']['parent_seed']), then restore. Never call env.reset() after this."""
    canonical_reset(env, snap["meta"]["parent_seed"])
    return restore(env, snap, restore_global_rng=restore_global_rng)


def to_bytes(snap):
    payload = pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL)
    return pickle.dumps({"schema": SCHEMA, "sha256": hashlib.sha256(payload).hexdigest(), "payload": payload},
                        protocol=pickle.HIGHEST_PROTOCOL)


def from_bytes(b):
    outer = pickle.loads(b)
    if outer["schema"] != SCHEMA:
        raise ValueError("schema differs")
    if hashlib.sha256(outer["payload"]).hexdigest() != outer["sha256"]:
        raise ValueError("snapshot payload hash mismatch (corrupt file)")
    return pickle.loads(outer["payload"])


def save(snap, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(to_bytes(snap))
    os.replace(tmp, path)


def load(path):
    with open(path, "rb") as f:
        return from_bytes(f.read())


def snapshot_digest(snap):
    """Digest of restorable content (episode included, meta excluded)."""
    return sha({k: v for k, v in snap.items() if k not in ("meta", "global_rng")}, 32)


# ───────────────────────────── audit: whole-graph digest ─────────────────────────────

_WALK_MODULES = ("robosuite", "robocasa", "damagesim", "envs", "oopsiebench")
_WALK_SKIP_TYPES = ("MjSim", "MjModel", "MjData", "MjRenderContext", "MjRenderContextOffscreen", "MjRenderer",
                    "_MjModelBodyViews", "_Element", "Element", "ElementTree", "Observable")
# Paths whose content is static scene definition (MJCF objects, fixture XML), rendering, or derived caches that
# are rebuilt on demand. They are listed in restore_through_damage.md.
_WALK_SKIP_ATTRS = {"sim", "_model", "_data", "viewer", "worldbody", "asset", "root", "tree", "_obj", "_elements",
                    "_xml_processors", "_render_context_offscreen", "model", "_observables"}


def graph_digest(env, max_depth=7):
    """Walk every python object reachable from env whose class lives in robosuite/robocasa/damagesim/oopsiebench and
    hash each plain-data leaf by attribute path. Used to prove that nothing mutable is left uncaptured: the digest
    right after restore must equal the digest at capture time."""
    out = {}
    seen = {}

    def ours(o):
        mod = type(o).__module__ or ""
        return mod.split(".")[0] in _WALK_MODULES and type(o).__name__ not in _WALK_SKIP_TYPES

    def walk(o, path, depth):
        if depth > max_depth:
            return
        if id(o) in seen:
            out[path] = f"alias:{seen[id(o)]}"
            return
        seen[id(o)] = path
        for k, v in list(vars(o).items()):
            if k in _WALK_SKIP_ATTRS:
                continue
            p = f"{path}.{k}"
            visit(v, p, depth + 1)

    def visit(v, p, depth):
        if ours(v) and hasattr(v, "__dict__"):
            walk(v, p, depth)
            return
        if isinstance(v, (list, tuple, deque)) and v and any(ours(x) for x in v):
            for i, x in enumerate(v):
                visit(x, f"{p}[{i}]", depth + 1)
            return
        if isinstance(v, dict) and v and any(ours(x) for x in v.values()):
            for k, x in v.items():
                visit(x, f"{p}[{k!r}]", depth + 1)
            return
        pv = plain(v)
        if pv is SKIP:
            return
        if isinstance(v, np.ndarray) and _is_mj_view(v):
            return
        out[p] = sha(pv) if not (isinstance(pv, str) and p.endswith(".file")) else f"str:{pv}"

    walk(env, "env", 0)
    return out


def digest_diff(a, b, ignore_prefixes=(), ignore_suffixes=()):
    keys = set(a) | set(b)
    diffs = sorted(k for k in keys if a.get(k) != b.get(k) and not any(k.startswith(p) for p in ignore_prefixes)
                   and not any(k.endswith(s) for s in ignore_suffixes))
    return diffs
