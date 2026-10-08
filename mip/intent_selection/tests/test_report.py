"""Tests for the reporting layer (report.py, audit_report.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from mip.intent_selection import report
from mip.intent_selection.audit_report import DEFAULT_AUDIT_DIR, build_audit_report
from mip.intent_selection.metrics import (
    _bootstrap_paired_metric,
    compute_m2_oracle_availability,
    compute_m3_cross_validated_selector,
)

N_BOOT = 100


def synthetic_records(n_parents=12, n_candidates=16, n_replicates=2, seed=0) -> pd.DataFrame:
    """Known effect: INTENT has 4 good candidates per state (p=0.85) and 12 poor ones (p=0.05);
    ACTION_STOCH candidates are homogeneous (p=0.10).
    """
    rng = np.random.RandomState(seed)
    rows = []
    for parent in range(n_parents):
        for s_idx, stratum in enumerate(("reset", "mid")):
            state_id = parent * 2 + s_idx
            for arm in ("ACTION_STOCH", "INTENT"):
                for cand in range(n_candidates):
                    p = (0.85 if cand < 4 else 0.05) if arm == "INTENT" else 0.10
                    for rep in range(n_replicates):
                        ok = bool(rng.rand() < p)
                        steps = int(rng.randint(40, 300))
                        rows.append({
                            "experiment_id": "synthetic_exp", "config_hash": "cfg", "checkpoint_pair_id": "seed_pair_0",
                            "arm": arm, "checkpoint_hash": f"ck_{arm}", "split": "dev", "stratum": stratum,
                            "parent_reset_id": parent, "state_id": state_id,
                            "source_policy": "none" if stratum == "reset" else ("ACTION" if parent % 2 else "INTENT"),
                            "elapsed_steps": 0 if stratum == "reset" else 120, "candidate_id": cand,
                            "candidate_seed": 1000 + cand, "decoder_seed": 7, "continuation_id": rep,
                            "tail_mode": "native", "success": ok,
                            "termination_reason": "success" if ok else "timeout", "steps_executed": steps,
                            "success_step": steps if ok else None, "return": float(ok),
                            "candidate_latency_ms": 5.0 if arm == "ACTION_STOCH" else 8.0,
                            "tail_latency_ms": float(steps * 2.0), "error_status": "ok",
                            "candidate_hash": f"{arm}_{state_id}_{cand}",
                        })
    return pd.DataFrame(rows)


def write_phase(tmp_path, phase, records):
    d = tmp_path / phase
    d.mkdir()
    records.to_parquet(d / "records.parquet")
    manifest = {"experiment_id": "synthetic_exp", "phase": phase,
                "analysis": {"bootstrap_resamples": N_BOOT, "primary_stratum": "mid"}}
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    recs = synthetic_records()
    d = write_phase(tmp_path_factory.mktemp("rep"), "pilot", recs)
    out = report.make_report(d)
    return d, recs, out


def _row(bt, stratum, metric, k, comp):
    return report._ci_row(bt, "native", stratum, metric, k, comp)


def test_make_report_writes_everything(pilot):
    d, _, out = pilot
    for name in ("metrics_by_state.csv", "aggregate_metrics.csv", "bootstrap_intervals.csv",
                 "per_seed_metrics.csv", "cost_by_arm.csv", "termination_breakdown.csv",
                 "experiment_report.md", "report_numbers.json"):
        assert (d / name).exists(), name
    figs = sorted(p.name for p in (d / "figures").glob("*.png"))
    assert figs == sorted(["oracle_availability_vs_k_native.png", "m3_selected_vs_random_native.png",
                           "paired_state_scatter_c16_native.png", "cost_by_arm_native.png"])
    for f in out["figures"]:
        assert (d / "figures" / f.split("/")[-1]).stat().st_size > 10_000


def test_point_estimates_match_metrics(pilot):
    _, recs, out = pilot
    bt = out["bootstrap"]
    for stratum in ("reset", "mid"):
        for arm in ("ACTION_STOCH", "INTENT"):
            direct = compute_m2_oracle_availability(recs, arm, stratum, "native", k_values=[1, 4, 16])
            for k in (1, 4, 16):
                assert _row(bt, stratum, "C", k, arm).estimate == pytest.approx(direct[k], abs=1e-12)
            sub = recs[(recs.arm == arm) & (recs.stratum == stratum)]
            m1 = sub.groupby("state_id")["success"].mean().mean()
            assert _row(bt, stratum, "M1", None, arm).estimate == pytest.approx(m1, abs=1e-12)
        d = compute_m2_oracle_availability(recs, "INTENT", stratum, "native", [16])[16] - \
            compute_m2_oracle_availability(recs, "ACTION_STOCH", stratum, "native", [16])[16]
        assert _row(bt, stratum, "C", 16, "INTENT-ACTION_STOCH").estimate == pytest.approx(d, abs=1e-12)
    # The known effect shows up in the expected direction.
    assert _row(bt, "mid", "C", 16, "INTENT-ACTION_STOCH").estimate > 0.1


def test_bootstrap_matches_metrics_bootstrap(pilot):
    _, recs, out = pilot
    seed = json.loads((pilot[0] / "report_numbers.json").read_text())["bootstrap_seed"]

    def c16(df_arm, stratum, tail):
        return compute_m2_oracle_availability(df_arm, df_arm["arm"].iloc[0], stratum, tail, [16])[16]

    ref = _bootstrap_paired_metric(recs, c16, stratum="mid", tail_mode="native", arm1="INTENT",
                                   arm2="ACTION_STOCH", n_resamples=N_BOOT, bootstrap_seed=seed)
    mine = _row(out["bootstrap"], "mid", "C", 16, "INTENT-ACTION_STOCH")
    assert mine.ci_lower == pytest.approx(ref["ci_lower"], abs=1e-12)
    assert mine.ci_upper == pytest.approx(ref["ci_upper"], abs=1e-12)
    assert mine.boot_mean == pytest.approx(ref["mean"], abs=1e-12)


def test_m3_matches_direct_call(pilot):
    _, recs, out = pilot
    st = out["state_table"]
    for state_id, arm in ((1, "INTENT"), (2, "ACTION_STOCH")):
        prim, _ = compute_m3_cross_validated_selector(
            recs, arm, state_id, "synthetic_exp", "seed_pair_0", selection_replicates=[0],
            heldout_replicates=[1], k_values=[1, 4, 16])
        row = st[(st.state_id == state_id) & (st.arm == arm)].iloc[0]
        for k in (1, 4, 16):
            assert row[f"m3_selected_k{k}"] == pytest.approx(prim[k]["selected_success"])
            assert row[f"m3_random_k{k}"] == pytest.approx(prim[k]["random_baseline"])
            assert row[f"m3_tie_fraction_k{k}"] == pytest.approx(prim[k]["tie_fraction"])


def test_report_numbers_appear_in_markdown(pilot):
    d, _, out = pilot
    md = (d / "experiment_report.md").read_text()
    prim = _row(out["bootstrap"], "mid", "C", 16, "INTENT-ACTION_STOCH")
    assert report._ci(prim) in md
    gain = _row(out["bootstrap"], "mid", "C_gain_vs_K1", 16, "INTENT")
    assert report._ci(gain) in md


def test_pilot_is_labelled_exploratory(pilot):
    d, _, out = pilot
    md = (d / "experiment_report.md").read_text()
    assert out["exploratory"] is True
    assert "EXPLORATORY" in md
    assert "Preview under the frozen rule (not a decision)" in md
    assert "### Frozen-rule verdict" not in md
    assert "one-selection/one-validation preview" in md
    assert pd.read_csv(d / "aggregate_metrics.csv")["exploratory"].all()
    assert json.loads((d / "report_numbers.json").read_text())["decisions"]["native"]["preview_only"] is True


def test_confirmation_emits_verdict(tmp_path):
    recs = synthetic_records(n_parents=8, n_replicates=4, seed=1)
    d = write_phase(tmp_path, "confirmation", recs)
    out = report.make_report(d)
    md = (d / "experiment_report.md").read_text()
    assert out["exploratory"] is False
    assert "EXPLORATORY" not in md
    assert "### Frozen-rule verdict" in md
    verdict = out["decisions"]["native"]["verdict"]
    assert verdict == "promising_coverage"
    assert f"`{verdict}`" in md


def test_incomplete_grid_is_rejected(tmp_path):
    recs = synthetic_records(n_parents=3)
    recs = recs.drop(recs[(recs.state_id == 1) & (recs.arm == "INTENT") & (recs.candidate_id == 3)
                          & (recs.continuation_id == 1)].index)
    d = write_phase(tmp_path, "pilot", recs)
    with pytest.raises(ValueError, match="Incomplete"):
        report.make_report(d)


@pytest.mark.skipif(not (DEFAULT_AUDIT_DIR / "rollout_audit.json").exists(), reason="audit artifacts absent")
def test_audit_report_reads_sources():
    md = build_audit_report(DEFAULT_AUDIT_DIR)
    roll = json.loads((DEFAULT_AUDIT_DIR / "rollout_audit.json").read_text())
    assert f"{roll['quat_representation']['windows']}" in md
    for tag in ("L1.", "L2.", "L3.", "L4.", "L5."):
        assert tag in md


def test_smoke_shape_k3(tmp_path):
    """Smoke phase: M=3 candidates, R=2, two parents. K values collapse to {1, 3}."""
    recs = synthetic_records(n_parents=2, n_candidates=3, n_replicates=2, seed=3)
    d = write_phase(tmp_path, "smoke", recs)
    out = report.make_report(d)
    nums = json.loads((d / "report_numbers.json").read_text())
    assert nums["k_values"] == [1, 3] and nums["k_primary"] == 3
    assert out["exploratory"] is True
    assert "EXPLORATORY" in (d / "experiment_report.md").read_text()
    assert len(out["figures"]) == 4
