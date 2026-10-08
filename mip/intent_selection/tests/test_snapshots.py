"""API-level checks for mip.intent_selection.snapshots on the real square env.

The A2 trajectory tests (1-7 + ablation) live in test_a2_measurements.py.
"""

from __future__ import annotations

import pickle
import pickletools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples" / "intent_selection"))

pytest.importorskip("robosuite")

import a2_harness as H  # noqa: E402
from mip.intent_selection.snapshots import (  # noqa: E402
    SNAPSHOT_SCHEMA_VERSION,
    Snapshot,
    capture_full_state,
    find_layers,
    get_integration_state,
    python_state_digest,
    restore_full_state,
)


@pytest.fixture(scope="module")
def env_and_snap():
    env = H.build_env(construction_seed=0, reset_seed=5)
    H.run_sequence(env, H.synthetic_sequence(10, 17))
    return env, capture_full_state(env)


def test_layers_found(env_and_snap):
    env, _ = env_and_snap
    layers = find_layers(env)
    assert {type(v).__name__ for v in layers.values()} == {
        "MultiStepWrapper", "VideoRecordingWrapper", "RobomimicLowdimWrapper", "EnvRobosuite",
        "NutAssemblySquare"}


def test_snapshot_is_plain_data(env_and_snap):
    """Pickled snapshot references no robosuite/mujoco/robomimic classes (portable)."""
    _, snap = env_and_snap
    globals_used = {arg for op, arg, _ in pickletools.genops(pickle.dumps(snap))
                    if op.name in ("GLOBAL", "STACK_GLOBAL") or (op.name.endswith("UNICODE") and
                                                                isinstance(arg, str) and "." in arg)}
    assert not any(str(g).startswith(("robosuite", "mujoco", "robomimic")) for g in globals_used)


def test_serialization_roundtrip_and_tamper(env_and_snap, tmp_path):
    _, snap = env_and_snap
    p = tmp_path / "s.pkl"
    snap.serialize(p)
    back = Snapshot.deserialize(p)
    assert back.digest() == snap.digest()
    assert back.schema_version == SNAPSHOT_SCHEMA_VERSION
    # tamper: flip one qpos value inside the payload
    payload = pickle.loads(p.read_bytes())
    payload["snapshot"].physics["state"][3] += 1e-12
    p.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        Snapshot.deserialize(p)


def test_restore_is_exact_and_does_not_step(env_and_snap):
    env, snap = env_and_snap
    H.run_sequence(env, H.synthetic_sequence(11, 9))
    restore_full_state(snap, env)
    assert np.array_equal(get_integration_state(find_layers(env)["robosuite"]), snap.physics["state"])
    assert find_layers(env)["robosuite"].timestep == snap.task_env["attrs"]["timestep"]
    d1 = python_state_digest(env)
    restore_full_state(snap, env)
    assert python_state_digest(env) == d1


def test_history_mismatch_raises(env_and_snap):
    env, snap = env_and_snap
    restore_full_state(snap, env)
    ms = find_layers(env)["multistep"]
    good = ms._get_obs(ms.n_obs_steps)
    capture_full_state(env, history=good)
    with pytest.raises(ValueError, match="history"):
        capture_full_state(env, history=good + 1e-6)


def test_model_fingerprint_mismatch_raises(env_and_snap):
    env, snap = env_and_snap
    bad = pickle.loads(pickle.dumps(snap))
    bad.physics["model_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        restore_full_state(bad, env)


def test_pending_actions_roundtrip(env_and_snap):
    env, _ = env_and_snap
    q = [np.arange(7.0), np.ones(7)]
    s = capture_full_state(env, pending_actions=q)
    h = restore_full_state(s, env)
    assert len(h.pending_actions) == 2 and all(np.array_equal(a, b) for a, b in zip(q, h.pending_actions))
