"""Harness correctness tests on the real square env (configs/test.yaml: T=64, 1 parent, M=2, R=2).

1. resume: an interrupted run (subset of jobs) resumed to completion gives records identical to an
   uninterrupted run; a changed identity is refused.
2. order / worker-count invariance: a full pipeline rerun (bank, candidates, branches) with a different
   worker count and a permuted job order gives identical records.
3. candidate identity is independent of the request batch size and of the process that generated it.
4. continuation seeds are identical across candidates of the same (state, replicate, policy).

Runtime: ~3-5 minutes (spawned worker processes load both checkpoints).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
pytest.importorskip("robosuite")

from mip.intent_selection import harness as H  # noqa: E402
from mip.intent_selection.branch_eval import (  # noqa: E402
    NONDETERMINISTIC_FIELDS,
    ResumeIdentityError,
    collect_records,
    continuation_seed,
    read_verified_record,
    record_identity,
    record_path,
)
from mip.intent_selection.rng import KeyedRNG  # noqa: E402

RUN = [str(REPO / ".venv/bin/python"), str(REPO / "examples/intent_selection/run_experiment.py"),
       "--config", str(REPO / "mip/intent_selection/configs/test.yaml"), "--no-report"]


def _run(out_dir, stage, *extra):
    env = {**os.environ, "PYTHONHASHSEED": str(np.random.randint(1 << 30))}  # hash() must not matter
    r = subprocess.run(RUN + ["--out-dir", str(out_dir), "--stage", stage, *extra], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


def _cfg(out_dir):
    return H.load_config(REPO / "mip/intent_selection/configs/test.yaml", {"output_dir": str(out_dir)})


def _records(out_dir):
    from mip.intent_selection.state_bank import load_bank

    cfg = _cfg(out_dir)
    df, missing = collect_records(cfg, load_bank(cfg))
    return df, missing


def _comparable(df):
    drop = [c for c in NONDETERMINISTIC_FIELDS if c in df.columns]
    d = df.drop(columns=drop).sort_values(["state_id", "arm", "candidate_id", "continuation_id"])
    d = d.reset_index(drop=True)
    d["tail_seeds"] = d["tail_seeds"].apply(lambda x: tuple(x))
    return d


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("harness")
    a, b = base / "A", base / "B"
    # A: full, 2 workers, canonical order
    _run(a, "bank", "--num-workers", "2")
    _run(a, "candidates", "--num-workers", "2")
    _run(a, "branch", "--num-workers", "2")
    # B: independent pipeline, 1 worker, permuted order, interrupted after 3 jobs, then resumed
    _run(b, "bank", "--num-workers", "1")
    _run(b, "candidates", "--num-workers", "1")
    _run(b, "branch", "--num-workers", "1", "--order-seed", "7", "--max-jobs", "3")
    partial, missing_partial = _records(b)
    _run(b, "branch", "--num-workers", "3", "--order-seed", "11")
    return {"A": a, "B": b, "partial": partial, "missing_partial": missing_partial}


def test_bank_and_candidates_identical_across_worker_counts(runs):
    ja = json.loads((runs["A"] / "bank_manifest.json").read_text())
    jb = json.loads((runs["B"] / "bank_manifest.json").read_text())
    assert ja["bank_hash"] == jb["bank_hash"]
    ca = json.loads((runs["A"] / "candidates/candidates_manifest.json").read_text())
    cb = json.loads((runs["B"] / "candidates/candidates_manifest.json").read_text())
    assert ca["candidate_pool_hash"] == cb["candidate_pool_hash"]


def test_resume_after_interruption_matches_uninterrupted(runs):
    assert runs["missing_partial"] > 0 and len(runs["partial"]) > 0, "interruption did not leave a partial run"
    da, ma = _records(runs["A"])
    db, mb = _records(runs["B"])
    assert ma == 0 and mb == 0
    pd.testing.assert_frame_equal(_comparable(da), _comparable(db))
    # records written before the interruption are untouched by the resume
    part = _comparable(runs["partial"]).set_index(["state_id", "arm", "candidate_id", "continuation_id"])
    full = _comparable(db).set_index(["state_id", "arm", "candidate_id", "continuation_id"]).loc[part.index]
    pd.testing.assert_frame_equal(part, full)


def test_order_and_worker_invariance_of_outcomes(runs):
    da, _ = _records(runs["A"])
    db, _ = _records(runs["B"])
    cols = ["success", "termination_reason", "steps_executed", "success_step", "return", "candidate_hash"]
    pd.testing.assert_frame_equal(_comparable(da)[cols], _comparable(db)[cols])
    # traces identical too
    for _, r in da.iterrows():
        if r["trace_key_tail"]:
            za, zb = np.load(runs["A"] / r["trace_key_tail"]), np.load(runs["B"] / r["trace_key_tail"])
            for k in za.files:
                assert np.array_equal(za[k], zb[k])


def test_resume_refuses_identity_change(runs, tmp_path):
    from mip.intent_selection.candidates import load_pool
    from mip.intent_selection.state_bank import load_bank

    cfg = _cfg(runs["A"])
    bank = load_bank(cfg)
    st = bank.states[0]
    pool = load_pool(cfg, st, "INTENT")
    ident = record_identity(cfg, st, "INTENT", 0, 0, "native", str(pool["candidate_hash"][0]))
    p = record_path(cfg, st["state_id"], "INTENT", 0, 0, "native")
    assert read_verified_record(p, ident) is not None
    changed = {**ident, "checkpoint_hash": "0" * 64}
    with pytest.raises(ResumeIdentityError):
        read_verified_record(p, changed)
    # tampered record content is refused
    q = tmp_path / "r.json"
    obj = json.loads(p.read_text())
    obj["record"]["success"] = not obj["record"]["success"]
    q.write_text(json.dumps(obj))
    with pytest.raises(ResumeIdentityError):
        read_verified_record(q, ident)
    # a config change (different experiment identity) is refused by the bank
    other = H.load_config(REPO / "mip/intent_selection/configs/test.yaml",
                          {"output_dir": str(runs["A"]), "nfe": 5})
    with pytest.raises(RuntimeError):
        load_bank(other)


def test_candidate_identity_independent_of_batch_size(runs):
    from mip.intent_selection.candidates import candidate_seed, generate_pool, load_pool
    from mip.intent_selection.state_bank import load_bank

    cfg = _cfg(runs["A"])
    H.setup_inference_process(cfg)
    bank = load_bank(cfg)
    st = bank.states[-1]
    seeds = [candidate_seed(cfg, "INTENT", st["state_id"], j) for j in range(5)]
    for arm in H.ARM_NAMES:
        pol = H.build_arm_policy(cfg, arm)
        seeds = [candidate_seed(cfg, arm, st["state_id"], j) for j in range(5)]
        hist = np.asarray(st["history"])
        p1 = generate_pool(pol, hist, seeds, request_batch_size=1)
        p5 = generate_pool(pol, hist, seeds, request_batch_size=5)
        p2 = generate_pool(pol, hist, seeds[::-1], request_batch_size=2)
        assert list(p1["candidate_hash"]) == list(p5["candidate_hash"]) == list(p2["candidate_hash"][::-1])
        assert np.array_equal(p1["env_action_chunk"], p5["env_action_chunk"])
        # identical to the pool generated in a spawned worker process
        stored = load_pool(cfg, st, arm)
        M = len(stored["candidate_id"])
        assert list(stored["candidate_hash"]) == list(p1["candidate_hash"][:M])
        # candidates are not degenerate
        assert len(set(p1["candidate_hash"])) == 5


def test_continuation_seeds_shared_across_candidates(runs):
    da, _ = _records(runs["A"])
    tails = da[da["tail_replans"] > 0]
    assert len(tails) > 0
    for (sid, arm, rep), g in tails.groupby(["state_id", "arm", "continuation_id"]):
        seqs = [list(s) for s in g["tail_seeds"]]
        n = min(len(s) for s in seqs)
        assert n > 0
        assert all(s[:n] == seqs[0][:n] for s in seqs), (sid, arm, rep)
        assert g["candidate_id"].nunique() == len(g)
    # different replicates / arms get different streams
    g = tails.groupby(["state_id", "arm", "continuation_id"])["tail_seeds"].first()
    assert len({tuple(x[:1]) for x in g}) == len(g)
    # the seed function has no candidate argument and KeyedRNG forbids candidate_id in this namespace
    cfg = _cfg(runs["A"])
    assert continuation_seed(cfg, 5, 0, 3, "P") == continuation_seed(cfg, 5, 0, 3, "P")
    with pytest.raises(ValueError):
        KeyedRNG("x").derive_seed("continuation", state_id=1, continuation_id=0, candidate_id=2)


def test_records_pass_metrics_validator(runs):
    from mip.intent_selection.metrics import OutcomeRecordValidator

    da, _ = _records(runs["A"])
    ok, errs, warns = OutcomeRecordValidator(da).validate()
    assert ok, errs
    assert not warns
    assert set(da["error_status"]) == {"ok"}


@pytest.fixture(scope="module", autouse=True)
def _cleanup(tmp_path_factory):
    yield
    shutil.rmtree(tmp_path_factory.getbasetemp() / "harness0", ignore_errors=True)
