"""Spec §4 A2 restoration tests against the real robomimic square env (make_env stack).

Run:  .venv/bin/python -m pytest mip/intent_selection/tests/test_a2_measurements.py -v -s
Measured deviations are written to outputs/intent_selection/audit/a2_measurements.json.

Tolerance: bit-identical (max abs deviation == 0.0 for every recorded quantity). TOL below is
the documented ceiling the spec allows (1e-8); every test also asserts exact byte equality.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples" / "intent_selection"))

pytest.importorskip("robosuite")
pytest.importorskip("mujoco")

import a2_harness as H  # noqa: E402
from mip.intent_selection.snapshots import (  # noqa: E402
    Snapshot,
    capture_full_state,
    find_layers,
    get_integration_state,
    python_state_digest,
    restore_full_state,
)

TOL = 1e-8
OUT = REPO / "outputs" / "intent_selection" / "audit" / "a2_measurements.json"
RESULTS: dict = {}
PY = sys.executable

RESET_SEED = 123
PRE_STEPS = 30
POST_STEPS = 64


def _summ(dev: dict) -> dict:
    return {k: float(v) for k, v in dev.items()}


def _worst(dev: dict) -> float:
    return float(max(dev.values())) if dev else 0.0


def _record(name, passed, max_dev, **extra):
    RESULTS[name] = {"passed": bool(passed), "max_abs_dev": float(max_dev), **extra}


@pytest.fixture(scope="module", autouse=True)
def _dump():
    yield
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev.update(RESULTS)
    prev["_tolerance"] = {"required": "bit-identical", "ceiling": TOL}
    OUT.write_text(json.dumps(prev, indent=2, sort_keys=True))


@pytest.fixture(scope="module")
def env():
    return H.build_env(construction_seed=0, reset_seed=RESET_SEED)


@pytest.fixture(scope="module")
def base(env):
    """Seeded reset, 30 fixed steps, snapshot. Returns (snapshot, sequences)."""
    env.reset(seed=RESET_SEED)
    H.run_sequence(env, H.synthetic_sequence(1000, PRE_STEPS))
    snap = capture_full_state(env)
    seqs = {q: H.synthetic_sequence(2000 + q, POST_STEPS) for q in range(3)}
    return snap, seqs


# --------------------------------------------------------------------------------------
# Test 1: snapshot -> 64 steps -> restore -> repeat
# --------------------------------------------------------------------------------------


def test_1_restore_and_repeat(env, base):
    snap, seqs = base
    restore_full_state(snap, env)
    a = H.run_sequence(env, seqs[0])
    restore_full_state(snap, env)
    b = H.run_sequence(env, seqs[0])
    dev = H.max_dev(a, b)
    bit = H.bitwise_equal(a, b)
    _record("test1_synthetic", bit and _worst(dev) <= TOL, _worst(dev), bitwise=bit,
            per_quantity=_summ(dev), steps=len(a["reward"]))
    assert len(a["reward"]) == POST_STEPS
    assert bit, dev


def test_1b_demo_replay_with_contact_and_success(env):
    """Same as test 1 on demo_1 actions from the demo's initial state (grasp, contact, success)."""
    d = H.demo("demo_1")
    T = len(d["actions"])
    snap_at = max(PRE_STEPS, T - POST_STEPS)
    H.reset_to_demo(env, d["states"][0])
    H.run_sequence(env, d["actions"][:snap_at])
    snap = capture_full_state(env)
    seq = d["actions"][snap_at:snap_at + POST_STEPS]
    a = H.run_sequence(env, seq)
    restore_full_state(snap, env)
    b = H.run_sequence(env, seq)
    dev = H.max_dev(a, b)
    bit = H.bitwise_equal(a, b)
    _record("test1b_demo_replay", bit, _worst(dev), bitwise=bit, per_quantity=_summ(dev),
            snapshot_step=snap_at, steps=len(a["reward"]), success_reached=bool(a["success"].any()),
            first_success_step=int(np.argmax(a["success"])) if a["success"].any() else None,
            max_reward=float(a["reward"].max()))
    assert bit, dev


# --------------------------------------------------------------------------------------
# Test 2: restore the same snapshot twice, same sequence
# --------------------------------------------------------------------------------------


def test_2_restore_twice_identical(env, base):
    snap, seqs = base
    runs = []
    for _ in range(2):
        restore_full_state(snap, env)
        runs.append(H.run_sequence(env, seqs[1]))
    dev = H.max_dev(*runs)
    bit = H.bitwise_equal(*runs)
    _record("test2_restore_twice", bit, _worst(dev), bitwise=bit)
    assert bit, dev


# --------------------------------------------------------------------------------------
# Test 3: 3 snapshots x 3 sequences, forward vs shuffled order
# --------------------------------------------------------------------------------------


def _three_snapshots(env):
    snaps = {}
    env.reset(seed=RESET_SEED)
    pre = H.synthetic_sequence(3000, 90)
    done = 0
    for sid, upto in enumerate((30, 60, 90)):
        H.run_sequence(env, pre[done:upto])
        done = upto
        snaps[sid] = capture_full_state(env)
    return snaps


@pytest.fixture(scope="module")
def grid(env):
    snaps = _three_snapshots(env)
    seqs = {q: H.synthetic_sequence(4000 + q, POST_STEPS) for q in range(3)}
    keys = [(s, q) for s in snaps for q in seqs]
    return snaps, seqs, keys


def _eval(env, snaps, seqs, keys, order):
    out = {}
    for i in order:
        s, q = keys[i]
        restore_full_state(snaps[s], env)
        out[(s, q)] = H.run_sequence(env, seqs[q])
    return out


def test_3_order_permutation(env, grid):
    snaps, seqs, keys = grid
    fwd = _eval(env, snaps, seqs, keys, range(len(keys)))
    order = list(range(len(keys)))
    random.Random(7).shuffle(order)
    shuf = _eval(env, snaps, seqs, keys, order)
    worst = max(_worst(H.max_dev(fwd[k], shuf[k])) for k in keys)
    bit = all(H.bitwise_equal(fwd[k], shuf[k]) for k in keys)
    distinct = len({fwd[k]["qpos"].tobytes() for k in keys})
    _record("test3_permutation", bit, worst, bitwise=bit, shuffled_order=order,
            n_keys=len(keys), distinct_trajectories=distinct)
    assert distinct == len(keys)  # the 9 branches really differ
    assert bit
    RESULTS["_grid_reference"] = {"note": "parent-process results used by test 4"}
    pytest.grid_reference = fwd


# --------------------------------------------------------------------------------------
# Test 4: serialize to disk; fresh process; 2 workers in different orders
# --------------------------------------------------------------------------------------


def _launch(jobs_path, order, out, cseed):
    return subprocess.Popen(
        [PY, str(REPO / "examples/intent_selection/a2_harness.py"), "worker", "--jobs", str(jobs_path),
         "--order", ",".join(map(str, order)), "--out", str(out), "--construction-seed", str(cseed)],
        cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")},
    )


def test_4_disk_fresh_process_and_workers(env, grid, tmp_path):
    snaps, seqs, keys = grid
    ref = getattr(pytest, "grid_reference", None) or _eval(env, snaps, seqs, keys, range(len(keys)))
    snap_dir = tmp_path / "snaps"
    for sid, s in snaps.items():
        s.serialize(snap_dir / f"snap_{sid}.pkl")
    loaded = {sid: Snapshot.deserialize(snap_dir / f"snap_{sid}.pkl") for sid in snaps}
    assert all(loaded[s].digest() == snaps[s].digest() for s in snaps)
    jobs = {"snapshots": {sid: (snap_dir / f"snap_{sid}.pkl").read_bytes() for sid in snaps},
            "seqs": seqs, "keys": keys}
    jp = tmp_path / "jobs.pkl"
    jp.write_bytes(pickle.dumps(jobs))

    # (a) one fresh process, all 9 jobs, forward order, construction seed 0
    p = _launch(jp, range(len(keys)), tmp_path / "fresh.pkl", 0)
    _, err = p.communicate(timeout=900)
    assert p.returncode == 0, err.decode()[-3000:]
    fresh = pickle.loads((tmp_path / "fresh.pkl").read_bytes())

    # (b) two concurrent workers, overlapping key sets, different orders and construction seeds
    o1 = [8, 3, 5, 0, 6, 1, 4]
    o2 = [4, 2, 7, 1, 8, 0]
    w1 = _launch(jp, o1, tmp_path / "w1.pkl", 7)
    w2 = _launch(jp, o2, tmp_path / "w2.pkl", 11)
    for w in (w1, w2):
        _, err = w.communicate(timeout=900)
        assert w.returncode == 0, err.decode()[-3000:]
    r1 = pickle.loads((tmp_path / "w1.pkl").read_bytes())
    r2 = pickle.loads((tmp_path / "w2.pkl").read_bytes())

    devs, bits = [], []
    for src in (fresh, r1, r2):
        for k, v in src["results"].items():
            devs.append(_worst(H.max_dev(ref[k], v)))
            bits.append(H.bitwise_equal(ref[k], v))
    covered = set(r1["results"]) | set(r2["results"])
    _record("test4_disk_subprocess_workers", all(bits), max(devs), bitwise=all(bits),
            n_comparisons=len(bits), worker_pids=[fresh["pid"], r1["pid"], r2["pid"]],
            worker_orders=[fresh["order"], o1, o2], keys_covered_by_pair=len(covered))
    assert covered == set(keys)
    assert all(bits)


# --------------------------------------------------------------------------------------
# Test 5: replan boundary history
# --------------------------------------------------------------------------------------


def test_5_replan_boundary_history(env):
    k = 3
    env.reset(seed=RESET_SEED)
    pre = H.synthetic_sequence(5000, 8 * k)
    rec = H.run_sequence(env, pre, chunk=8)
    ms = find_layers(env)["multistep"]
    hist = ms._get_obs(ms.n_obs_steps)
    snap = capture_full_state(env, history=hist, pending_actions=[])
    cont = H.synthetic_sequence(5001, 16)
    orig = H.run_sequence(env, cont, chunk=8)

    # restore into a *different*, freshly made env that went through its own episode
    other = H.build_env(construction_seed=31, reset_seed=31)
    H.run_sequence(other, H.synthetic_sequence(99, 13))
    h = restore_full_state(snap, other)
    expected = np.stack(rec["obs"][-2:])
    exact = np.array_equal(h.obs, expected)
    not_padded = not np.array_equal(h.obs[0], h.obs[1])
    restored = H.run_sequence(other, cont, chunk=8)
    bit = H.bitwise_equal(orig, restored)
    _record("test5_replan_boundary", exact and not_padded and bit and h.pending_actions == [],
            _worst(H.max_dev(orig, restored)), history_equal=exact, history_not_padded=not_padded,
            n_real_frames=h.n_real_frames, elapsed_steps=h.elapsed_steps,
            pending_actions=len(h.pending_actions), continuation_bitwise=bit,
            history_max_abs_dev=float(np.abs(h.obs - expected).max()))
    assert exact and not_padded
    assert h.n_real_frames == ms.n_obs_steps + 1 and h.elapsed_steps == 8 * k
    assert h.pending_actions == []
    assert bit


# --------------------------------------------------------------------------------------
# Test 6: controller state A / B / A
# --------------------------------------------------------------------------------------


def _controller_view(env):
    rs = find_layers(env)["robosuite"]
    r = rs.robots[0]
    osc = r.composite_controller.part_controllers["right"]
    g = r.composite_controller.part_controllers["right_gripper"]
    return {"goal_pos": np.array(osc.goal_pos, float), "goal_ori": np.array(osc.goal_ori, float),
            "osc_torques": np.array(osc.torques, float), "grip_torques": np.array(g.torques, float),
            "grip_goal_qvel": np.array(g.goal_qvel, float),
            "current_action": np.array(r.gripper["right"].current_action, float),
            "ctrl": rs.sim.data.ctrl.copy()}


def test_6_controller_state_ABA(env, base):
    snap, _ = base
    A = H.synthetic_sequence(6000, POST_STEPS, toggle_every=5)
    B = H.synthetic_sequence(6001, POST_STEPS, toggle_every=3)
    restore_full_state(snap, env)
    d0 = python_state_digest(env)
    a1 = H.run_sequence(env, A[:1])
    c1 = _controller_view(env)
    a1_rest = H.run_sequence(env, A[1:])
    restore_full_state(snap, env)
    b = H.run_sequence(env, B)
    restore_full_state(snap, env)
    d2 = python_state_digest(env)
    a2 = H.run_sequence(env, A[:1])
    c2 = _controller_view(env)
    a2_rest = H.run_sequence(env, A[1:])
    first_step_dev = max(float(np.abs(c1[k] - c2[k]).max()) for k in c1)
    dev = max(_worst(H.max_dev(a1, a2)), _worst(H.max_dev(a1_rest, a2_rest)))
    bit = H.bitwise_equal(a1_rest, a2_rest) and all(c1[k].tobytes() == c2[k].tobytes() for k in c1)
    differs_from_B = not np.array_equal(b["qpos"], a1_rest["qpos"][: len(b["qpos"])])
    digest_diff = [k for k in d0 if d0[k] != d2[k]]
    _record("test6_controller_ABA", bit and not digest_diff, dev, first_step_controller_max_dev=first_step_dev,
            first_step_components=sorted(c1), state_digest_components_differing=digest_diff,
            B_differs=differs_from_B)
    assert differs_from_B
    assert not digest_diff
    assert bit


# --------------------------------------------------------------------------------------
# Test 7: transition determinism x5, including fresh env instances
# --------------------------------------------------------------------------------------


def test_7_transition_determinism(env, base):
    snap, seqs = base
    seq = seqs[2]
    runs, labels = [], []
    for i in range(3):
        restore_full_state(snap, env)
        runs.append(H.run_sequence(env, seq))
        labels.append(f"same_env_{i}")
    fresh_reset = H.build_env(construction_seed=999, reset_seed=999)
    restore_full_state(snap, fresh_reset)
    runs.append(H.run_sequence(fresh_reset, seq))
    labels.append("fresh_env_seed999_reset")
    fresh_noreset = H.build_env(construction_seed=4242, reset_seed=None)
    restore_full_state(snap, fresh_noreset)
    runs.append(H.run_sequence(fresh_noreset, seq))
    labels.append("fresh_env_seed4242_never_reset")

    # stepping must not consume any RNG stream (env.rng, numpy global, python random)
    rs = find_layers(env)["robosuite"]
    restore_full_state(snap, env)
    st = (rs.rng.bit_generator.state, np.random.get_state()[1].tobytes(), random.getstate())
    H.run_sequence(env, seq)
    st2 = (rs.rng.bit_generator.state, np.random.get_state()[1].tobytes(), random.getstate())
    rng_untouched = st == st2

    devs = [_worst(H.max_dev(runs[0], r)) for r in runs[1:]]
    bit = all(H.bitwise_equal(runs[0], r) for r in runs[1:])
    _record("test7_transition_determinism", bit and rng_untouched, max(devs), repeats=len(runs),
            labels=labels, bitwise=bit, stepping_consumes_rng=not rng_untouched,
            post_chunk_caching_valid=bool(bit and rng_untouched))
    assert rng_untouched
    assert bit


# --------------------------------------------------------------------------------------
# Ablation: which components are load-bearing (each left stale = draft-like restore)
# --------------------------------------------------------------------------------------


def test_8_ablation_missing_components(env, base):
    """Which components are load-bearing? Full restore, then leave ONE component stale.

    The stale values come from a different episode (construction seed 555, reset seed 77,
    370 steps in), i.e. what a partial restore leaves behind when the harness restores across
    branches / env instances. Also reproduces the previous draft's restore (qpos, qvel, act,
    ctrl, time + forward, timestep/done, wrapper deques; nothing else).
    """
    import mujoco
    from mip.intent_selection.snapshots import (_restore_attrs, _restore_buffer,
                                                _restore_task_env, _restore_wrappers)

    snap, _ = base
    seq = H.synthetic_sequence(8000, POST_STEPS, toggle_every=4)
    restore_full_state(snap, env)
    ref = H.run_sequence(env, seq)
    other = H.build_env(construction_seed=555, reset_seed=77)
    H.run_sequence(other, H.synthetic_sequence(8001, 370))
    stale = capture_full_state(other)
    rs = find_layers(env)["robosuite"]
    robot = rs.robots[0]
    cc = robot.composite_controller

    def draft_restore(e):
        restore_full_state(stale, e, check_model=False)
        d, p = rs.sim.data, snap.physics
        d.qpos[:] = p["qpos"]; d.qvel[:] = p["qvel"]; d.act[:] = p["act"]; d.ctrl[:] = p["ctrl"]
        d.time = p["time"]
        rs.timestep = snap.task_env["attrs"]["timestep"]; rs.done = snap.task_env["attrs"]["done"]
        _restore_wrappers(snap.wrappers, find_layers(e))
        rs.sim.forward()

    def full_then(fn):
        def apply(e):
            restore_full_state(snap, e)
            fn(e)
        return apply

    cases = {
        "draft_restore_equivalent": draft_restore,
        "qacc_warmstart": full_then(lambda e: rs.sim.data.qacc_warmstart.__setitem__(
            slice(None), stale.physics["qacc_warmstart"])),
        "mj_forward_after_setState_without_reapply": full_then(
            lambda e: mujoco.mj_forward(rs.sim.model._model, rs.sim.data._data)),
        "gripper_model_current_action": full_then(lambda e: setattr(
            robot.gripper["right"], "current_action", np.array(stale.robots[0]["grippers"]["right"]))),
        "osc_controller_attrs": full_then(lambda e: _restore_attrs(
            cc.part_controllers["right"], stale.robots[0]["parts"]["right"])),
        "gripper_controller_attrs": full_then(lambda e: _restore_attrs(
            cc.part_controllers["right_gripper"], stale.robots[0]["parts"]["right_gripper"])),
        "robot_proprio_buffers": full_then(lambda e: [
            _restore_buffer(getattr(robot, k), v) for k, v in stale.robots[0]["buffers"].items()]),
        "observables_and_obs_cache": full_then(lambda e: _restore_task_env(
            dict(snap.task_env, observables=stale.task_env["observables"],
                 obs_cache=stale.task_env["obs_cache"]), rs)),
        "env_timestep_curtime_objects_on_pegs": full_then(lambda e: [
            setattr(rs, k, copy_(stale.task_env["attrs"][k]))
            for k in ("timestep", "cur_time", "objects_on_pegs", "done")]),
        "multistep_wrapper_deques": full_then(lambda e: _restore_wrappers(
            {"multistep": stale.wrappers["multistep"]}, find_layers(e))),
    }
    out = {}
    for name, fn in cases.items():
        fn(env)
        r = H.run_sequence(env, seq)
        dev = H.max_dev(ref, r)
        out[name] = {"max_abs_dev": _worst(dev), "qpos": dev["qpos"], "obs": dev["obs"],
                     "reward": dev["reward"], "done": dev["done"], "steps_run": int(len(r["reward"]))}
    out["_stale_vs_snapshot"] = {
        "gripper_current_action_snapshot": np.asarray(snap.robots[0]["grippers"]["right"]).tolist(),
        "gripper_current_action_stale": np.asarray(stale.robots[0]["grippers"]["right"]).tolist(),
        "elapsed_steps_snapshot": snap.meta["elapsed_steps"], "elapsed_steps_stale": stale.meta["elapsed_steps"]}
    RESULTS["ablation_missing_component"] = out
    assert out["draft_restore_equivalent"]["max_abs_dev"] > TOL
    assert out["qacc_warmstart"]["max_abs_dev"] > 0
    # load-bearing components measured on this env (see restoration_report.md)
    for k in ("gripper_model_current_action", "osc_controller_attrs", "observables_and_obs_cache"):
        assert out[k]["max_abs_dev"] > TOL, k
    assert out["multistep_wrapper_deques"]["steps_run"] < POST_STEPS  # stale reward list truncates


def copy_(v):
    import copy
    return copy.deepcopy(v)
