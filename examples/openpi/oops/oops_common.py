"""Shared definitions for the "Oops, I Erred" spatial-robustness study (docs/oops_workshop_robustness_spec.md).

Everything that must be identical between the policy-free validity pre-pass (build_case_manifest.py),
the rollout harness (eval_spatial_shifts.py), the native-evaluator parity hook (eval_libero_intent.py
--episode-seeding) and the analysis (analyze_robustness.py) lives here: task manifest, perturbation
geometry, case identity, state hashing and per-episode seeding.

Pure numpy/hashlib -- importable without LIBERO, robosuite, openpi or a GPU.
"""

import hashlib
import json
import math

import numpy as np

EXPERIMENT = "oops_workshop_spatial_robustness_v1"
# Bump whenever perturbation geometry, settle procedure or case identity changes. Part of every
# case_id, so records from an incompatible configuration can never be matched or resumed.
CONFIG_VERSION = "pv1"
# Bump whenever a validity threshold/rule changes. Rules are frozen BEFORE any policy outcome is seen.
VALIDITY_VERSION = "vv1"

SUITE = "libero_goal"
ENV_RES = 256
MAX_STEPS = 300            # native libero_goal cap (eval_libero_intent.MAX_STEPS_BY_SUITE)
NUM_STEPS_WAIT = 10        # native settle: dummy actions before the policy acts
EXTRA_SETTLE_STEPS = 40    # manifest-only stability probe (never part of a policy rollout)
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

DEFAULT_DELTAS = (0.0, 0.05, 0.10)
DEFAULT_N_ANGLES = 10
DEFAULT_RESET_IDS = (0, 1, 2)

# Task manifest, keyed by the INSTRUCTION (integer task ids are resolved at runtime from the suite and
# never trusted across datasets). moved joint = the free joint of the object named in the instruction.
# Ineligible tasks have no free-body manipulated object (articulated fixture targets).
TASKS = [
    # instruction, moved free joint (or None), eligibility reason, known scene constraints
    ("open the middle drawer of the cabinet", None,
     "ineligible: target is a slide joint of a fixed fixture; no free-body object to translate", ""),
    ("put the bowl on the stove", "akita_black_bowl_1_joint0",
     "eligible: bowl is a free body resting on the table", "shift may land bowl on/over the stove (goal) or the plate"),
    ("put the wine bottle on top of the cabinet", "wine_bottle_1_joint0",
     "eligible: wine bottle is a free body resting on the table", "tall object: shift may topple it or hit the cabinet/rack"),
    ("open the top drawer and put the bowl inside", "akita_black_bowl_1_joint0",
     "eligible: bowl is a free body resting on the table (drawer stage unchanged)", "two-stage task; only the bowl is shifted"),
    ("put the bowl on top of the cabinet", "akita_black_bowl_1_joint0",
     "eligible: bowl is a free body resting on the table", "shift may land the bowl on the plate or stove"),
    ("push the plate to the front of the stove", "plate_1_joint0",
     "eligible: plate is a free body resting on the table", "goal is a fixed table region; shift may already satisfy it"),
    ("put the cream cheese in the bowl", "cream_cheese_1_joint0",
     "eligible: cream cheese is a free body resting on the table", "shift may land cream cheese in/against the bowl (goal)"),
    ("turn on the stove", None,
     "ineligible: target is a hinge joint of a fixed fixture; no free-body object to translate", ""),
    ("put the bowl on the plate", "akita_black_bowl_1_joint0",
     "eligible: bowl is a free body resting on the table", "shift may land the bowl on the plate (goal)"),
    ("put the wine bottle on the rack", "wine_bottle_1_joint0",
     "eligible: wine bottle is a free body resting on the table", "tall object: shift may topple it or hit the rack/cabinet"),
]
TASK_BY_INSTRUCTION = {t[0]: t for t in TASKS}
ELIGIBLE_INSTRUCTIONS = [t[0] for t in TASKS if t[1] is not None]

# Validity rules v1. Engineering thresholds; the xy/z ones are inherited from the existing harness defaults
# (eval_libero_perturbed --xy-eject-tol 0.04 / --z-fall-tol 0.03). Frozen with VALIDITY_VERSION.
VALIDITY_RULES = {
    "version": VALIDITY_VERSION,
    "xy_drift_invalid_m": 0.04,      # settled XY vs requested XY (contact resolution shoved the object)
    "xy_drift_unknown_m": 0.02,
    "z_change_invalid_m": 0.03,      # settled Z vs the same reset's unperturbed settled Z (fell / now on another support)
    "z_change_unknown_m": 0.015,
    "penetration_invalid_m": 0.005,  # deepest contact penetration of the moved subtree at placement (before any step)
    "penetration_unknown_m": 0.002,
    "other_disturb_invalid_m": 0.02, # any OTHER free object displaced vs the reset's unperturbed settle
    "other_disturb_unknown_m": 0.01,
    "unstable_m": 0.01,              # moved object still travelling between native settle end and +EXTRA_SETTLE_STEPS
    "table_margin_m": 0.0,           # requested XY must lie on the table-top footprint
    "pre_satisfied": "env success predicate already true after the native settle => invalid",
    "reachability": "recorded only (distance from robot base); never used to declare invalidity",
}


def slug(instruction):
    return instruction.strip().lower().replace(" ", "_")


def angles(n_angles=DEFAULT_N_ANGLES):
    """Fixed, evenly spaced directions in the world XY plane, decided before evaluation."""
    return np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False)


def physical_cases(reset_ids, deltas, n_angles):
    """(reset_id, delta, angle_idx, angle_rad) list. delta=0 appears ONCE per reset (angle_idx=-1)."""
    out = []
    ang = angles(n_angles)
    for r in reset_ids:
        for d in deltas:
            if d == 0.0:
                out.append((int(r), 0.0, -1, float("nan")))
            else:
                for i, a in enumerate(ang):
                    out.append((int(r), float(d), i, float(a)))
    return out


def displacement(delta, angle_rad):
    if delta == 0.0:
        return 0.0, 0.0
    return delta * math.cos(angle_rad), delta * math.sin(angle_rad)


def flat_xyz_index(sim, joint_name):
    """Index of a free joint's x in robosuite's flattened state [time, qpos, qvel] (= qpos addr + 1).

    The pre-2026-08-03 harness used hardcoded qpos addresses (bowl=9, ...) as FLAT indices; that shifted
    a gripper-finger joint / a neighbour's quaternion by dx and the object's world-x by dy. Never hardcode."""
    jid = sim.model.joint_name2id(joint_name)
    assert sim.model.jnt_type[jid] == 0, f"{joint_name} is not a free joint"
    return int(sim.model.jnt_qposadr[jid]) + 1


def perturb_state(state, xyz_idx, dx, dy):
    """Copy of the flat state with the free body's XY translated; z and orientation preserved."""
    s = np.array(state, dtype=np.float64, copy=True)
    s[xyz_idx] += dx
    s[xyz_idx + 1] += dy
    return s


def state_hash(state):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(state, dtype=np.float64)).tobytes()).hexdigest()


def case_id(instruction, reset_id, delta, angle_idx, init_state_hash):
    return (f"{slug(instruction)}|r{int(reset_id)}|d{delta:.3f}|a{int(angle_idx)}"
            f"|{CONFIG_VERSION}|{init_state_hash[:12]}")


def episode_seed(instruction, reset_id, delta, angle_idx, policy_seed):
    """Deterministic per-episode seed; identical integer for both arms on the same physical case.

    Order-independent (so resume/sharding does not change randomness). Matching integers give
    reproducibility, not identical semantic randomness across different models."""
    key = f"{slug(instruction)}|{int(reset_id)}|{delta:.3f}|{int(angle_idx)}|{int(policy_seed)}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


def seed_everything(seed, policy=None):
    """Seed numpy + torch (PyTorch openpi sampling noise / MIP flow-map intent noise) and, for a JAX
    openpi Policy, its internal PRNG key (Policy.infer splits self._rng)."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    if policy is not None and hasattr(policy, "_rng"):
        import jax
        policy._rng = jax.random.key(seed)


def file_fingerprint(path, full=False, chunk=1 << 24):
    """sha256 of a file. full=False hashes size + first/last 16 MiB (fast for multi-GB checkpoints)."""
    import os
    h = hashlib.sha256()
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as f:
        if full or size <= 2 * chunk:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        else:
            h.update(f.read(chunk))
            f.seek(size - chunk)
            h.update(f.read(chunk))
    return ("sha256:" if full or size <= 2 * chunk else "sha256-headtail:") + h.hexdigest()


def dir_fingerprint(path, full=False):
    """Fingerprint of a checkpoint directory (every non-empty file; JAX/orbax or PyTorch layout)."""
    import os
    parts = []
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            if os.path.getsize(p) > 0:
                parts.append(f"{os.path.relpath(p, path)}={file_fingerprint(p, full=full)}")
    return "dir-sha256:" + hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest(), parts


def append_jsonl(path, record):
    """Append one record durably (flush + fsync). A torn last line is ignored by read_jsonl."""
    import os
    line = json.dumps(record, default=_json_default) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    import os
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn final write from a killed job; the case is simply re-run on resume
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
