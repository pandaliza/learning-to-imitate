"""Regression tests for review findings in rng.py and metrics.py."""

import numpy as np
import pandas as pd
import pytest

from mip.intent_selection.metrics import (
    _bootstrap_paired_metric,
    compute_m2_oracle_availability,
    compute_m3_cross_validated_selector,
)
from mip.intent_selection.rng import KeyedRNG


def test_every_supplied_key_changes_the_seed():
    rng = KeyedRNG("exp")
    base = rng.derive_seed("m3_ties", state_id=3, subset_index=0)
    assert base != rng.derive_seed("m3_ties", state_id=3, subset_index=1)
    boot = rng.derive_seed("bootstrap", state_id=0)
    assert boot != rng.derive_seed("bootstrap", state_id=0, continuation_id=7)


def test_namespaces_are_distinct_and_required_keys_enforced():
    rng = KeyedRNG("exp")
    assert rng.derive_seed("m3_subsets", state_id=1) != rng.derive_seed("bootstrap", state_id=1)
    with pytest.raises(ValueError):
        rng.derive_seed("m3_ties", state_id=1)
    with pytest.raises(ValueError):
        rng.derive_seed("continuation", state_id=1, continuation_id=0, candidate_id=2)


def _records(success_by_parent, arms=("INTENT", "ACTION"), M=4, R=2):
    rows = []
    for parent, arm_success in success_by_parent.items():
        for arm in arms:
            for j in range(M):
                for r in range(R):
                    rows.append({
                        "stratum": "mid", "tail_mode": "native", "arm": arm,
                        "parent_reset_id": parent, "state_id": parent,
                        "candidate_id": j, "continuation_id": r,
                        "success": bool(arm_success[arm]),
                    })
    return pd.DataFrame(rows)


def test_bootstrap_counts_duplicated_parents():
    # INTENT-ACTION difference is +1 on parent 0 and 0 on parent 1.
    df = _records({0: {"INTENT": 1, "ACTION": 0}, 1: {"INTENT": 0, "ACTION": 0}})

    def c1(sub, stratum, tail_mode):
        return compute_m2_oracle_availability(sub, sub["arm"].iloc[0], stratum, tail_mode, [1])[1]

    out = _bootstrap_paired_metric(df, c1, n_resamples=400, bootstrap_seed=0)
    # With two parents, resamples are {0,0}:1.0, {0,1}:0.5, {1,1}:0.0, so all three values appear.
    assert out["ci_lower"] == 0.0 and out["ci_upper"] == 1.0
    assert out["mean"] == pytest.approx(0.5, abs=0.08)


def _m3_df(M=16, R=4, arm="INTENT", seed=0, p=None):
    gen = np.random.default_rng(seed)
    rows = []
    for j in range(M):
        pj = 0.5 if p is None else p[j]
        for r in range(R):
            rows.append({
                "experiment_id": "exp", "stratum": "mid", "tail_mode": "native", "arm": arm,
                "parent_reset_id": 0, "state_id": 0, "candidate_id": j, "continuation_id": r,
                "success": bool(gen.random() < pj),
            })
    return pd.DataFrame(rows)


def test_m3_k4_subsets_shared_across_arms_and_reproducible():
    kw = {"state_id": 0, "experiment_id": "exp", "checkpoint_pair_id": "pair0", "k_values": [4]}
    a1, _ = compute_m3_cross_validated_selector(_m3_df(arm="INTENT"), arm="INTENT", **kw)
    a2, _ = compute_m3_cross_validated_selector(_m3_df(arm="INTENT"), arm="INTENT", **kw)
    assert a1 == a2
    # Identical outcomes under the other arm label give the same random baseline,
    # which holds only if both arms use the same subsets.
    b, _ = compute_m3_cross_validated_selector(_m3_df(arm="ACTION"), arm="ACTION", **kw)
    assert b[4]["random_baseline"] == pytest.approx(a1[4]["random_baseline"])


def test_m3_deterministic_candidates_select_the_oracle():
    p = [1.0 if j == 5 else 0.0 for j in range(16)]
    res, _ = compute_m3_cross_validated_selector(
        _m3_df(p=p), arm="INTENT", state_id=0, experiment_id="exp",
        checkpoint_pair_id="pair0", k_values=[16],
    )
    assert res[16]["selected_success"] == 1.0
