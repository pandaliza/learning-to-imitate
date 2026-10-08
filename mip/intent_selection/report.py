"""Reporting layer for the intent-selection experiment (spec §8, §9, §10, §13).

``make_report(phase_dir)`` reads ``manifest.json`` and the outcome records in
``phase_dir``, validates them, runs ``metrics.analyze`` and writes:

- ``metrics_by_state.csv``, ``aggregate_metrics.csv``, ``bootstrap_intervals.csv``,
  ``per_seed_metrics.csv``, ``termination_breakdown.csv``, ``cost_by_arm.csv``,
  ``analyze_crosscheck.csv``, ``report_numbers.json``
- ``figures/*.png``: oracle availability vs K, M3 selected vs random,
  paired per-state C(16) scatter, cost bars (one set per tail mode)
- ``experiment_report.md``

Every number is computed from the saved records; the only other inputs are the
audit facts, read from the audit JSON files by ``audit_report.audit_facts``.

Metric definitions follow spec §8 and use the primitives in ``metrics.py``:
C_s(K) is the per-replicate oracle availability ``compute_oracle_availability``
averaged over continuation replicates, then states are averaged with equal weight.
M3 uses ``compute_m3_cross_validated_selector`` per (arm, state). Intervals are
95% percentile intervals from a parent-reset bootstrap that draws parents with
exactly the RNG sequence of ``metrics._bootstrap_paired_metric`` (same seed, same
``RandomState.choice`` calls), vectorised so one draw serves every endpoint.

Usage::

    .venv/bin/python -m mip.intent_selection.report outputs/intent_selection/pilot
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mip.intent_selection import metrics
from mip.intent_selection.audit_report import DEFAULT_AUDIT_DIR, audit_facts
from mip.intent_selection.rng import KeyedRNG

# ============================================================================
# Constants: arms, colours, phases
# ============================================================================

INTENT_ARM = "INTENT"
REFERENCE_ARMS = ("ACTION_STOCH", "ACTION")  # first one present is the comparison arm
EXPLORATORY_PHASES = {"smoke", "pilot"}
STRATUM_ORDER = ["reset", "mid"]

# Colorblind-safe categorical slots (validated pair: blue / orange); one colour per
# arm everywhere. Differences use neutral ink.
ARM_COLORS = {"ACTION_STOCH": "#2a78d6", "ACTION": "#2a78d6", "INTENT": "#eb6834"}
ARM_MARKERS = {"ACTION_STOCH": "o", "ACTION": "o", "INTENT": "s"}
DIFF_COLOR = "#3d3d3a"
INK = "#1f1f1e"
MUTED = "#6b6a66"
GRID = "#e4e3df"
STRATUM_MARKERS = {"reset": "o", "mid": "^"}

RECORD_FILE_CANDIDATES = (
    "records.parquet", "outcomes.parquet", "outcome_records.parquet",
    "records.csv", "outcomes.csv", "outcome_records.csv",
    "records.jsonl", "outcomes.jsonl", "outcome_records.jsonl",
)
RECORD_DIR_CANDIDATES = ("records", "outcomes", "outcome_records", "results", "jobs")


# ============================================================================
# Loading
# ============================================================================

def _mget(manifest: dict, *paths: str, default=None):
    """Return the first manifest value found at any dotted path (top level or under 'config')."""
    for root in (manifest, manifest.get("config", {}) or {}, manifest.get("resolved_config", {}) or {}):
        for path in paths:
            cur: Any = root
            ok = True
            for part in path.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if ok and cur is not None:
                return cur
    return default


def load_manifest(phase_dir: str | Path) -> dict:
    p = Path(phase_dir) / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    with open(p) as f:
        return json.load(f)


def _read_record_file(p: Path) -> pd.DataFrame:
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".jsonl":
        return pd.read_json(p, lines=True)
    if p.suffix == ".json":
        with open(p) as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "records" in obj:
            obj = obj["records"]
        return pd.DataFrame(obj if isinstance(obj, list) else [obj])
    raise ValueError(f"Unsupported record file {p}")


def load_records(phase_dir: str | Path, manifest: dict | None = None) -> pd.DataFrame:
    """Load outcome records from a file or a directory of per-job record files."""
    phase_dir = Path(phase_dir)
    manifest = manifest or {}
    explicit = _mget(manifest, "records_path", "outcome_records", "records", "outputs.records")
    paths: list[Path] = []
    if isinstance(explicit, str):
        p = Path(explicit)
        paths = [p if p.is_absolute() else phase_dir / p]
    else:
        paths = [phase_dir / n for n in RECORD_FILE_CANDIDATES if (phase_dir / n).exists()]
        if not paths:
            paths = [phase_dir / n for n in RECORD_DIR_CANDIDATES if (phase_dir / n).is_dir()]
    if not paths:
        raise FileNotFoundError(f"No outcome records found in {phase_dir}")
    p = paths[0]
    if p.is_dir():
        files = sorted(f for ext in ("*.parquet", "*.jsonl", "*.json", "*.csv") for f in p.rglob(ext))
        if not files:
            raise FileNotFoundError(f"No record files in {p}")
        df = pd.concat([_read_record_file(f) for f in files], ignore_index=True)
    else:
        df = _read_record_file(p)
    if "return" not in df.columns and "return_value" in df.columns:
        df = df.rename(columns={"return_value": "return"})
    df["success"] = df["success"].astype(bool)
    return df


# ============================================================================
# Per-state computation
# ============================================================================

def _success_grid(g: pd.DataFrame) -> tuple[np.ndarray, list, list]:
    """Y[candidate, replicate] for one (tail, stratum, arm, state). Raises on an incomplete grid."""
    piv = g.pivot_table(index="candidate_id", columns="continuation_id", values="success", aggfunc="first")
    if piv.isna().any().any():
        raise ValueError(f"Incomplete candidate x replicate grid for state {g['state_id'].iloc[0]} "
                         f"arm {g['arm'].iloc[0]}: {int(piv.isna().sum().sum())} missing cells")
    return piv.to_numpy(dtype=float), list(piv.index), list(piv.columns)


def oracle_availability_state(Y: np.ndarray, k: int) -> float:
    """Spec M2 for one state: mean over replicates r of C_{s,r}(K)."""
    return float(np.mean([metrics.compute_oracle_availability(Y[:, r].astype(bool), k)
                          for r in range(Y.shape[1])]))


def _m3_split(replicates: list[int], manifest: dict) -> tuple[list[int], list[int]]:
    sel = _mget(manifest, "m3_selection_replicates", "analysis.m3_selection_replicates")
    held = _mget(manifest, "m3_heldout_replicates", "analysis.m3_heldout_replicates")
    if sel is not None and held is not None:
        return list(sel), list(held)
    reps = sorted(int(r) for r in replicates)
    half = len(reps) // 2
    return reps[:half], reps[half:]


def compute_state_table(records: pd.DataFrame, manifest: dict, k_values: list[int],
                        experiment_id: str) -> pd.DataFrame:
    n_subsets_k4 = int(_mget(manifest, "analysis.subset_samples_k4", "subset_samples_k4", default=128))
    rows = []
    group_cols = ["tail_mode", "stratum", "arm", "state_id"]
    for (tail, stratum, arm, state_id), g in records.groupby(group_cols, sort=True):
        Y, cands, reps = _success_grid(g)
        first = g.iloc[0]
        row = {
            "tail_mode": tail, "stratum": stratum, "arm": arm, "state_id": state_id,
            "parent_reset_id": first["parent_reset_id"], "checkpoint_pair_id": first["checkpoint_pair_id"],
            "split": first["split"], "source_policy": first["source_policy"],
            "elapsed_steps": first["elapsed_steps"],
            "n_candidates": Y.shape[0], "n_replicates": Y.shape[1],
            "n_success": int(Y.sum()),
            "m1_random_success": float(Y.mean()),
            "n_candidates_any_success": int((Y.max(axis=1) > 0).sum()),
            "n_candidates_all_success": int((Y.min(axis=1) > 0).sum()),
            "candidate_success_std": float(Y.mean(axis=1).std()),
            "n_unique_candidate_hash": int(g["candidate_hash"].nunique()),
        }
        for k in k_values:
            row[f"c_k{k}"] = oracle_availability_state(Y, k)
            row[f"gain_k{k}"] = row[f"c_k{k}"] - row[f"c_k{min(k_values)}"]
        if Y.shape[1] >= 2:
            sel, held = _m3_split(reps, manifest)
            with contextlib.redirect_stdout(io.StringIO()):
                prim, sens = metrics.compute_m3_cross_validated_selector(
                    g, arm=arm, state_id=state_id, experiment_id=experiment_id,
                    checkpoint_pair_id=first["checkpoint_pair_id"], tail_mode=tail,
                    selection_replicates=sel, heldout_replicates=held,
                    k_values=k_values, n_subsets_k4=n_subsets_k4, split_sensitivity=True)
            for k in k_values:
                for name, res in (("m3", prim), ("m3swap", sens)):
                    r = res.get(k, {})
                    row[f"{name}_selected_k{k}"] = r.get("selected_success", math.nan)
                    row[f"{name}_random_k{k}"] = r.get("random_baseline", math.nan)
                    row[f"{name}_gain_k{k}"] = r.get("gain", math.nan)
                    row[f"{name}_tie_fraction_k{k}"] = r.get("tie_fraction", math.nan)
                    if name == "m3":
                        row[f"m3_analytic_k{k}"] = r.get("analytic_tied", math.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Parent bootstrap (vectorised, same draws as metrics._bootstrap_paired_metric)
# ============================================================================

def parent_draw_counts(parents: list, n_resamples: int, seed: int) -> np.ndarray:
    """counts[b, i] = number of times parents[i] is drawn in resample b.

    Uses the identical RandomState sequence as metrics._bootstrap_paired_metric.
    """
    rng = np.random.RandomState(seed)
    index = {p: i for i, p in enumerate(parents)}
    counts = np.zeros((n_resamples, len(parents)), dtype=np.int64)
    for b in range(n_resamples):
        for p in rng.choice(parents, size=len(parents), replace=True):
            counts[b, index[p]] += 1
    return counts


def _boot_means(states: pd.DataFrame, col: str, parents: list, counts: np.ndarray) -> np.ndarray:
    """Bootstrap distribution of the equal-weight state mean of ``col``; NaN where no state drawn."""
    s = states.groupby("parent_reset_id")[col].agg(["sum", "count"]).reindex(parents).fillna(0.0)
    num, den = counts @ s["sum"].to_numpy(), counts @ s["count"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _interval(estimate: float, boot: np.ndarray, ci: float = 0.95) -> dict[str, float]:
    b = boot[~np.isnan(boot)]
    if len(b) == 0:
        return {"estimate": estimate, "ci_lower": math.nan, "ci_upper": math.nan,
                "boot_mean": math.nan, "boot_std": math.nan, "n_resamples_used": 0}
    a = 1 - ci
    return {"estimate": estimate, "ci_lower": float(np.percentile(b, a / 2 * 100)),
            "ci_upper": float(np.percentile(b, (1 - a / 2) * 100)),
            "boot_mean": float(b.mean()), "boot_std": float(b.std()), "n_resamples_used": int(len(b))}


def _metric_columns(k_values: list[int], has_m3: bool) -> list[tuple[str, int | None, str]]:
    cols = [("m1_random_success", None, "M1")]
    for k in k_values:
        cols.append((f"c_k{k}", k, "C"))
    for k in k_values[1:]:
        cols.append((f"gain_k{k}", k, "C_gain_vs_K1"))
    if has_m3:
        for k in k_values:
            cols += [(f"m3_selected_k{k}", k, "M3_selected"), (f"m3_random_k{k}", k, "M3_random"),
                     (f"m3_gain_k{k}", k, "M3_gain"), (f"m3_analytic_k{k}", k, "M3_selected_analytic_ties"),
                     (f"m3swap_gain_k{k}", k, "M3_gain_swapped_split")]
    return cols


def bootstrap_table(state_df: pd.DataFrame, arms: list[str], ref_arm: str, k_values: list[int],
                    n_resamples: int, seed: int) -> pd.DataFrame:
    rows = []
    has_m3 = any(c.startswith("m3_") for c in state_df.columns)
    for (tail, stratum), sdf in state_df.groupby(["tail_mode", "stratum"], sort=True):
        parents = sorted(sdf["parent_reset_id"].unique())
        counts = parent_draw_counts(parents, n_resamples, seed)
        for col, k, metric in _metric_columns(k_values, has_m3):
            boots = {}
            for arm in arms:
                a = sdf[sdf["arm"] == arm]
                if len(a) == 0 or a[col].isna().all():
                    continue
                boots[arm] = _boot_means(a, col, parents, counts)
                rows.append({"tail_mode": tail, "stratum": stratum, "metric": metric, "K": k,
                             "comparison": arm, "n_parents": a["parent_reset_id"].nunique(),
                             "n_states": len(a), "n_resamples": n_resamples, "bootstrap_seed": seed,
                             **_interval(float(a[col].mean()), boots[arm])})
            if INTENT_ARM in boots and ref_arm in boots:
                ai, ar = sdf[sdf["arm"] == INTENT_ARM], sdf[sdf["arm"] == ref_arm]
                rows.append({"tail_mode": tail, "stratum": stratum, "metric": metric, "K": k,
                             "comparison": f"{INTENT_ARM}-{ref_arm}",
                             "n_parents": len(set(ai["parent_reset_id"]) & set(ar["parent_reset_id"])),
                             "n_states": len(set(ai["state_id"]) & set(ar["state_id"])),
                             "n_resamples": n_resamples, "bootstrap_seed": seed,
                             **_interval(float(ai[col].mean() - ar[col].mean()),
                                         boots[INTENT_ARM] - boots[ref_arm])})
    return pd.DataFrame(rows)


# ============================================================================
# Aggregate tables
# ============================================================================

def aggregate_table(state_df: pd.DataFrame, k_values: list[int], exploratory: bool,
                    group_extra: tuple[str, ...] = ()) -> pd.DataFrame:
    value_cols = [c for c in state_df.columns
                  if c.startswith(("m1_", "c_k", "gain_k", "m3_", "m3swap_", "candidate_success_std"))]
    keys = list(group_extra) + ["tail_mode", "stratum", "arm"]
    agg = state_df.groupby(keys, sort=True)[value_cols].mean().reset_index()
    sizes = state_df.groupby(keys, sort=True).agg(
        n_parents=("parent_reset_id", "nunique"), n_states=("state_id", "nunique"),
        n_candidates=("n_candidates", "max"), n_replicates=("n_replicates", "max")).reset_index()
    out = sizes.merge(agg, on=keys)
    out["exploratory"] = exploratory
    return out


def cost_table(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tail, stratum, arm), g in records.groupby(["tail_mode", "stratum", "arm"], sort=True):
        cand = g.drop_duplicates(["state_id", "candidate_id"])
        rows.append({
            "tail_mode": tail, "stratum": stratum, "arm": arm,
            "n_candidates_total": len(cand), "n_continuations": len(g),
            "candidate_latency_ms_mean": float(cand["candidate_latency_ms"].mean()),
            "candidate_latency_ms_std": float(cand["candidate_latency_ms"].std(ddof=0)),
            "tail_latency_ms_mean": float(g["tail_latency_ms"].mean()),
            "tail_latency_ms_std": float(g["tail_latency_ms"].std(ddof=0)),
            "tail_latency_s_total": float(g["tail_latency_ms"].sum() / 1000.0),
            "sim_steps_total": int(g["steps_executed"].sum()),
            "sim_steps_mean": float(g["steps_executed"].mean()),
        })
    return pd.DataFrame(rows)


def termination_table(records: pd.DataFrame) -> pd.DataFrame:
    t = (records.groupby(["tail_mode", "stratum", "arm", "termination_reason"]).size()
         .rename("count").reset_index())
    t["fraction"] = t["count"] / t.groupby(["tail_mode", "stratum", "arm"])["count"].transform("sum")
    return t


def analyze_crosscheck(analyze_tables: dict[str, pd.DataFrame], state_df: pd.DataFrame,
                       k_values: list[int]) -> pd.DataFrame:
    """Compare metrics.analyze per-state values with this report's per-state values."""
    a = analyze_tables.get("metrics_by_state")
    if a is None or len(a) == 0:
        return pd.DataFrame()
    keys = ["tail_mode", "stratum", "state_id", "arm"]
    m = state_df[keys + ["m1_random_success"] + [f"c_k{k}" for k in k_values]].merge(
        a, on=keys, suffixes=("_report", "_analyze"))
    rows = []
    for col in ["m1_random_success"] + [f"c_k{k}" for k in k_values]:
        if f"{col}_analyze" not in m:
            continue
        d = (m[f"{col}_report"] - m[f"{col}_analyze"]).abs()
        rows.append({"column": col, "n_rows": len(m), "n_rows_differ": int((d > 1e-12).sum()),
                     "max_abs_difference": float(d.max()) if len(d) else 0.0})
    return pd.DataFrame(rows)


# ============================================================================
# Figures
# ============================================================================

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    })
    return plt


def _ci_row(bt: pd.DataFrame, tail, stratum, metric, k, comparison):
    r = bt[(bt.tail_mode == tail) & (bt.stratum == stratum) & (bt.metric == metric)
           & (bt.comparison == comparison) & ((k == bt.K) if k is not None else bt.K.isna())]
    return r.iloc[0] if len(r) else None


def _yerr(r):
    return [[r.estimate - r.ci_lower], [r.ci_upper - r.estimate]]


def _strata(df) -> list[str]:
    present = list(df["stratum"].unique())
    return [s for s in STRATUM_ORDER if s in present] + sorted(s for s in present if s not in STRATUM_ORDER)


def fig_oracle_vs_k(bt, tail, arms, ref_arm, k_values, strata, tag, path):
    plt = _style()
    fig, axes = plt.subplots(1, len(strata) + 1, figsize=(4.2 * (len(strata) + 1), 3.8))
    offsets = {arm: (i - (len(arms) - 1) / 2) * 0.06 for i, arm in enumerate(arms)}
    for ax, stratum in zip(axes[:-1], strata, strict=True):
        for arm in arms:
            rs = [_ci_row(bt, tail, stratum, "C", k, arm) for k in k_values]
            if any(r is None for r in rs):
                continue
            x = np.log2(k_values) + offsets[arm]
            y = [r.estimate for r in rs]
            err = np.array([[r.estimate - r.ci_lower for r in rs], [r.ci_upper - r.estimate for r in rs]])
            ax.errorbar(x, y, yerr=err, color=ARM_COLORS.get(arm, DIFF_COLOR), marker=ARM_MARKERS.get(arm, "o"),
                        markersize=7, capsize=3, label=arm)
            ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
        ax.set_title(f"{stratum.upper()}: oracle availability C(K)")
        ax.set_ylim(-0.02, 1.04)
        ax.set_ylabel("C(K)")
    ax = axes[-1]
    for i, stratum in enumerate(strata):
        rs = [_ci_row(bt, tail, stratum, "C", k, f"{INTENT_ARM}-{ref_arm}") for k in k_values]
        if any(r is None for r in rs):
            continue
        x = np.log2(k_values) + (i - (len(strata) - 1) / 2) * 0.08
        err = np.array([[r.estimate - r.ci_lower for r in rs], [r.ci_upper - r.estimate for r in rs]])
        ax.errorbar(x, [r.estimate for r in rs], yerr=err, color=DIFF_COLOR,
                    marker=STRATUM_MARKERS.get(stratum, "D"), markersize=7, capsize=3,
                    linestyle="-" if stratum == "mid" else "--",
                    markerfacecolor=DIFF_COLOR if stratum == "mid" else "white", label=stratum.upper())
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_title(f"Paired difference {INTENT_ARM} − {ref_arm}")
    ax.set_ylabel("ΔC(K)")
    for a in axes:
        a.set_xticks(np.log2(k_values))
        a.set_xticklabels([str(k) for k in k_values])
        a.set_xlabel("K (candidates)")
        a.legend(loc="best", fontsize=8)
    fig.suptitle(f"Oracle availability vs K, tail={tail}, 95% parent-bootstrap CIs{tag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_m3(bt, tail, arms, k_values, strata, tag, path, m3_label):
    plt = _style()
    fig, axes = plt.subplots(1, len(strata), figsize=(5.2 * len(strata), 4.4), squeeze=False)
    axes = axes[0]
    offsets = {arm: (i - (len(arms) - 1) / 2) * 0.32 for i, arm in enumerate(arms)}
    for ax, stratum in zip(axes, strata, strict=True):
        for gi, k in enumerate(k_values):
            for arm in arms:
                rr = _ci_row(bt, tail, stratum, "M3_random", k, arm)
                rs = _ci_row(bt, tail, stratum, "M3_selected", k, arm)
                if rr is None or rs is None:
                    continue
                x0 = gi + offsets[arm]
                col = ARM_COLORS.get(arm, DIFF_COLOR)
                ax.plot([x0 - 0.07, x0 + 0.07], [rr.estimate, rs.estimate], color=col, linewidth=1.2, alpha=0.7)
                ax.errorbar(x0 - 0.07, rr.estimate, yerr=_yerr(rr), color=col, marker=ARM_MARKERS.get(arm, "o"),
                            markerfacecolor="white", markersize=7, capsize=3, linestyle="none")
                ax.errorbar(x0 + 0.07, rs.estimate, yerr=_yerr(rs), color=col, marker=ARM_MARKERS.get(arm, "o"),
                            markersize=7, capsize=3, linestyle="none")
                tie = bt[(bt.tail_mode == tail) & (bt.stratum == stratum) & (bt.comparison == arm)
                         & (bt.metric == "M3_tie_fraction") & (k == bt.K)]
                if len(tie) and k > 1:
                    ax.text(x0, 1.06, f"ties\n{tie.iloc[0].estimate:.2f}", ha="center", va="bottom",
                            fontsize=7, color=MUTED)
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels([f"K={k}" for k in k_values])
        ax.set_ylim(-0.02, 1.2)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_ylabel("held-out success")
        ax.set_title(f"{stratum.upper()}: M3 selected vs random")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ARM_COLORS.get(a, DIFF_COLOR), marker=ARM_MARKERS.get(a, "o"), linestyle="none",
                      markersize=7, label=a) for a in arms]
    handles += [Line2D([], [], color=MUTED, marker="o", markerfacecolor="white", linestyle="none", label="random (hollow)"),
                Line2D([], [], color=MUTED, marker="o", linestyle="none", label="selected (filled)")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"M3 simulator selector, tail={tail}, 95% CIs; tie fraction above each arm{tag}\n{m3_label}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_scatter(state_df, tail, ref_arm, k, strata, tag, path):
    plt = _style()
    fig, ax = plt.subplots(figsize=(5.0, 5.4))
    col = f"c_k{k}"
    sdf = state_df[state_df.tail_mode == tail]
    piv = sdf.pivot_table(index=["stratum", "state_id"], columns="arm", values=col).dropna()
    from matplotlib.lines import Line2D
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    handles = [Line2D([], [], color=MUTED, linewidth=1, linestyle="--", label="y = x")]
    notes = []
    for stratum in strata:
        if stratum not in piv.index.get_level_values(0) or INTENT_ARM not in piv or ref_arm not in piv:
            continue
        p = piv.loc[stratum]
        pts = p.groupby([ref_arm, INTENT_ARM]).size().reset_index(name="n")
        mid = stratum == "mid"
        dx = 0.012 if mid else -0.012
        ax.scatter(pts[ref_arm] + dx, pts[INTENT_ARM], s=36 + 30 * (pts["n"] - 1),
                   marker=STRATUM_MARKERS.get(stratum, "D"), facecolors=DIFF_COLOR if mid else "white",
                   edgecolors=DIFF_COLOR, linewidths=1.4, alpha=0.8)
        for _, q in pts[pts["n"] > 1].iterrows():
            ax.annotate(str(int(q["n"])), (q[ref_arm] + dx, q[INTENT_ARM]), xytext=(9 if mid else -9, -9),
                        textcoords="offset points", ha="center", fontsize=7, color=INK)
        handles.append(Line2D([], [], marker=STRATUM_MARKERS.get(stratum, "D"), linestyle="none", markersize=7,
                              color=DIFF_COLOR, markerfacecolor=DIFF_COLOR if mid else "white",
                              label=f"{stratum.upper()} (n={len(p)} states)"))
        above = int((p[INTENT_ARM] > p[ref_arm]).sum())
        below = int((p[INTENT_ARM] < p[ref_arm]).sum())
        notes.append(f"{stratum.upper()}: {above} above, {below} below, {len(p) - above - below} on y=x")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel(f"C_{ref_arm}({k}) per state")
    ax.set_ylabel(f"C_{INTENT_ARM}({k}) per state")
    ax.set_title(f"Paired per-state C({k}), tail={tail}{tag}", fontsize=10)
    fig.legend(handles=handles, loc="lower left", fontsize=8, ncol=1, bbox_to_anchor=(0.02, 0.0))
    fig.text(0.98, 0.01, "\n".join(notes) + "\nnumbers: states at that point", ha="right", va="bottom",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_cost(cost: pd.DataFrame, tail, arms, strata, tag, path):
    plt = _style()
    c = cost[cost.tail_mode == tail]
    panels = [("candidate_latency_ms_mean", "candidate_latency_ms_std", "Candidate latency (ms / candidate)", 1.0),
              ("tail_latency_ms_mean", "tail_latency_ms_std", "Tail latency (s / continuation)", 1e-3),
              ("sim_steps_total", None, *(("Simulator steps (total, millions)", 1e-6)
                                          if c["sim_steps_total"].max() >= 1e6
                                          else ("Simulator steps (total, thousands)", 1e-3)))]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    width = 0.8 / max(len(arms), 1)
    for ax, (col, sd, title, scale) in zip(axes, panels, strict=True):
        for i, arm in enumerate(arms):
            xs, ys, es = [], [], []
            for j, stratum in enumerate(strata):
                r = c[(c.arm == arm) & (c.stratum == stratum)]
                if len(r):
                    xs.append(j + (i - (len(arms) - 1) / 2) * width)
                    ys.append(r.iloc[0][col] * scale)
                    es.append(r.iloc[0][sd] * scale if sd else 0)
            ax.bar(xs, ys, width=width * 0.92, color=ARM_COLORS.get(arm, DIFF_COLOR), label=arm,
                   yerr=es if sd else None, capsize=3, ecolor=INK, error_kw={"linewidth": 1})
            for x, y in zip(xs, ys, strict=True):
                ax.annotate(f"{y:.3g}", (x, y), xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=7, color=INK)
        ax.set_xticks(range(len(strata)))
        ax.set_xticklabels([s.upper() for s in strata])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    fig.suptitle(f"Cost by arm, tail={tail} (error bars: ±1 SD across candidates / continuations){tag}", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================================
# Markdown report
# ============================================================================

def _f(x, nd=3, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def _ci(r, signed=True):
    if r is None:
        return "n/a"
    return f"{_f(r.estimate, signed=signed)} [{_f(r.ci_lower, signed=signed)}, {_f(r.ci_upper, signed=signed)}]"


CLAIMS = {
    "promising_coverage": ("In this one checkpoint pair on square-mh-state with native tails, INTENT's candidate "
                           "pool contains a successful continuation more often than {ref}'s at equal K on {stratum} "
                           "states, and simulator-selected INTENT candidates keep their success under held-out "
                           "continuation seeds. This is a system-level difference between two checkpoints; it does not "
                           "show that intent semantics cause it."),
    "promising_selectability": ("Coverage at K={k} is similar between INTENT and {ref} on {stratum} states, but "
                                "INTENT candidates selected on some continuation seeds keep their success on held-out "
                                "seeds better than random INTENT candidates (checkpoint-specific, native tails)."),
    "useful_headroom": ("INTENT gains coverage from K=1 to K={k} on {stratum} states, but its selected success still "
                        "trails {ref}. This is headroom within a weaker policy, not superiority."),
    "no_advantage": ("Under this protocol (one checkpoint pair, native tails) INTENT offers no coverage advantage "
                     "over {ref} at K={k} on {stratum} states; the interval excludes the practical threshold. This "
                     "does not refute other intent formulations."),
    "inconclusive": ("No claim about H1 is supported for {stratum} states at K={k}: the frozen rule returns "
                     "inconclusive (reason below)."),
}


def build_markdown(ctx: dict) -> str:
    bt, agg, tail_modes = ctx["bt"], ctx["agg"], ctx["tail_modes"]
    ref, kmax, k_values = ctx["ref_arm"], ctx["k_primary"], ctx["k_values"]
    pstrat, expl, phase = ctx["primary_stratum"], ctx["exploratory"], ctx["phase"]
    diff = f"{INTENT_ARM}-{ref}"
    tagline = "**EXPLORATORY**: " if expl else ""
    L = [f"# Experiment report: intent selection, phase `{phase}`", ""]
    if expl:
        L += [f"> **EXPLORATORY.** This is a `{phase}` run on development states "
              f"(max continuation replicates R={ctx['R']}). It is not a powered test and never enters the "
              "confirmation result. All numbers below are exploratory; no decision is drawn from them.", ""]
    L += [f"Experiment `{ctx['experiment_id']}`, checkpoint pair(s) {', '.join(f'`{c}`' for c in ctx['pairs'])}, "
          f"comparison arm `{ref}` vs `{INTENT_ARM}`, tail modes {', '.join(f'`{t}`' for t in tail_modes)}. "
          f"{ctx['n_records']} outcome records; candidate pool M={ctx['M']}, replicates R={ctx['R']}, "
          f"K ∈ {{{', '.join(map(str, k_values))}}}. Intervals: 95% percentile, {ctx['n_resamples']} parent-reset "
          f"bootstrap resamples (seed {ctx['seed']}). Every number is computed from the saved records by "
          "`mip/intent_selection/report.py`.", ""]
    if f"{ref}" == "ACTION_STOCH":
        L += [f"`{ref}` is the ACTION checkpoint sampled from Gaussian initial noise. Native ACTION evaluation is "
              "zero-noise and deterministic, so this arm is **non-native**.", ""]
    L += ["| Stratum | Parents | States per arm |", "|---|---|---|"]
    sized = agg[agg.arm.isin([INTENT_ARM, ref])]
    for tail, stratum in [(t, s) for t in tail_modes for s in ctx["strata"]]:
        g = sized[(sized.tail_mode == tail) & (sized.stratum == stratum)]
        if len(g) == 0:
            continue
        L.append(f"| {stratum.upper()} ({tail}) | {int(g.n_parents.max())} | "
                 + ", ".join(f"{a}={int(n)}" for a, n in zip(g.arm, g.n_states, strict=True)) + " |")
    L.append("")

    for tail in tail_modes:
        dec = ctx["decisions"].get(tail)
        L += [f"## Tail mode `{tail}`", ""]
        # Claim
        L += ["### Claim supported", ""]
        if dec is None:
            L += [f"The primary endpoint ({pstrat.upper()}, K={kmax}) is not available for this tail mode.", ""]
        elif expl:
            L += [f"{tagline}No confirmatory claim. This phase only estimates feasibility, runtime, saturation and "
                  f"effect size. Observed primary endpoint: paired ΔC({kmax}) on {pstrat.upper()} = "
                  f"{_ci(dec['primary_row'])}.", ""]
        else:
            L += [CLAIMS[dec["verdict"]].format(ref=ref, stratum=pstrat.upper(), k=kmax), ""]

        # Endpoints
        L += ["### Effect sizes (every endpoint, regardless of sign)", "",
              f"Paired differences are {INTENT_ARM} − {ref} on identical states. Estimate [95% CI].", "",
              "| Role | Endpoint | Stratum | K | Estimate [95% CI] | States |", "|---|---|---|---|---|---|"]
        other = [s for s in ctx["strata"] if s != pstrat]
        endpoints = [("**Primary**", "ΔC(K) oracle availability", pstrat, kmax, "C")]
        endpoints += [("Supporting", "ΔC(K) oracle availability", s, kmax, "C") for s in other]
        for k in k_values:
            if k not in (1, kmax):
                endpoints += [("Supporting", "ΔC(K) oracle availability", s, k, "C") for s in ctx["strata"]]
        if ctx["has_m3"]:
            for k in [k for k in k_values if k > 1][::-1]:
                for s in ctx["strata"]:
                    endpoints.append(("Supporting", "Δ M3 selected success", s, k, "M3_selected"))
                    endpoints.append(("Supporting", "Δ M3 gain (selected − random)", s, k, "M3_gain"))
        endpoints += [("Reference", "Δ M1 random-candidate success (= ΔC(1))", s, None, "M1") for s in ctx["strata"]]
        for role, name, s, k, metric in endpoints:
            r = _ci_row(bt, tail, s, metric, k, diff)
            L.append(f"| {role} | {name} | {s.upper()} | {k if k else '-'} | {_ci(r)} | "
                     f"{int(r.n_states) if r is not None else 'n/a'} |")
        L.append("")

        # Per-arm values
        L += ["### Per-arm values", "",
              "| Stratum | Arm | M1 | " + " | ".join(f"C({k})" for k in k_values) + " |",
              "|---|---|---|" + "---|" * len(k_values)]
        for s in ctx["strata"]:
            for arm in ctx["arms"]:
                cells = [_ci(_ci_row(bt, tail, s, "M1", None, arm), signed=False)]
                cells += [_ci(_ci_row(bt, tail, s, "C", k, arm), signed=False) for k in k_values]
                L.append(f"| {s.upper()} | {arm} | " + " | ".join(cells) + " |")
        L.append("")
        L += ["### Per-arm gain C(K) − C(1)", "",
              "> Caveat: a larger gain can simply reflect a weaker K=1 policy, which leaves more room to gain. "
              "Read gains together with the M1 row above.", "",
              "| Stratum | Arm | " + " | ".join(f"C({k})−C(1)" for k in k_values[1:]) + " |",
              "|---|---|" + "---|" * len(k_values[1:])]
        for s in ctx["strata"]:
            for arm in ctx["arms"]:
                L.append(f"| {s.upper()} | {arm} | " + " | ".join(
                    _ci(_ci_row(bt, tail, s, "C_gain_vs_K1", k, arm)) for k in k_values[1:]) + " |")
        L.append("")

        # M3
        if ctx["has_m3"]:
            L += ["### M3 cross-validated simulator selector", "", f"_{ctx['m3_label']}._", "",
                  "| Stratum | Arm | K | Random | Selected | Gain [95% CI] | Tie fraction | Selected (analytic ties) | "
                  "Gain, swapped split |", "|---|---|---|---|---|---|---|---|---|"]
            for s in ctx["strata"]:
                for arm in ctx["arms"]:
                    for k in k_values:
                        g = lambda m, t=tail, st=s, kk=k, a=arm: _ci_row(bt, t, st, m, kk, a)  # noqa: E731
                        L.append(f"| {s.upper()} | {arm} | {k} | {_ci(g('M3_random'), False)} | "
                                 f"{_ci(g('M3_selected'), False)} | {_ci(g('M3_gain'))} | "
                                 f"{_f(g('M3_tie_fraction').estimate if g('M3_tie_fraction') is not None else None, 2)} | "
                                 f"{_f(g('M3_selected_analytic_ties').estimate if g('M3_selected_analytic_ties') is not None else None)} | "
                                 f"{_ci(g('M3_gain_swapped_split'))} |")
            L += ["", "M3 uses simulator outcomes; it is not a learned or deployable selector.", ""]
        else:
            L += ["### M3", "", "Not computed: fewer than two continuation replicates.", ""]

        # Decision
        L += ["### Frozen-rule verdict" if not expl else "### Preview under the frozen rule (not a decision)", ""]
        if dec is None:
            L += ["Not available.", ""]
        else:
            if expl:
                L += [f"{tagline}The frozen practical rule (`metrics.apply_practical_rule`) applied to these "
                      "exploratory numbers would return the line below. It is shown only as a preview of how the "
                      "rule behaves on pilot-sized data and is **not a conclusion**.", ""]
            L += [f"- Rule output: `{dec['verdict']}`", f"- Reason: {dec['reason']}", "- Inputs:"]
            for k_, v in dec["inputs"].items():
                L.append(f"  - `{k_}` = {v}")
            L.append("")

        # Costs
        cost = ctx["cost"][ctx["cost"].tail_mode == tail]
        L += ["### Cost", "", "| Stratum | Arm | Candidate latency ms (mean ± SD) | Tail latency s / continuation "
              "(mean) | Simulator steps (total) | Steps / continuation |", "|---|---|---|---|---|---|"]
        for _, r in cost.iterrows():
            L.append(f"| {r.stratum.upper()} | {r.arm} | {r.candidate_latency_ms_mean:.1f} ± {r.candidate_latency_ms_std:.1f} "
                     f"| {r.tail_latency_ms_mean / 1000:.2f} | {int(r.sim_steps_total):,} | {r.sim_steps_mean:.1f} |")
        L.append("")

    # Confounders
    fa = ctx["audit"]
    sri, sas = fa.get("sr_intent"), fa.get("sr_action_stoch")
    L += ["## Confounders and limits", ""]
    if sri and sas and sri["n_episodes"] and sas["n_episodes"]:
        L.append(f"- **Checkpoint quality.** Measured full-episode success in the audit: INTENT "
                 f"{sri['rate']:.3f} ({sri['n_success']}/{sri['n_episodes']}) vs {ref} "
                 f"{sas['min_rate']:.2f}–{sas['max_rate']:.2f} across runs ({sas['n_success']}/{sas['n_episodes']} "
                 "pooled). Coverage differences are confounded with this quality gap; see the M1 row.")
    else:
        L.append("- **Checkpoint quality.** Audit success rates not found in the audit artifacts.")
    sh = fa.get("saved_history", {}).get("intent")
    if sh:
        L.append(f"- **INTENT saved SR is inflated.** The saved {sh['max']:.2f} is the maximum of {sh['n_evals']} "
                 f"periodic evaluations (mean {sh['mean']:.3f}).")
    L += ["- **Architecture.** INTENT is an intent flow plus an MLP action decoder; ACTION is a direct action flow. "
          "The arms differ in proposal dimensionality and training steps as well as supervision.",
          f"- **Non-native {ref}.**" + (" Native ACTION samples with zero initial noise (deterministic), so candidate "
                                        "diversity in this arm comes from a sampling mode the checkpoint was not "
                                        "evaluated with." if ref == "ACTION_STOCH" else ""),
          f"- **One checkpoint pair** ({', '.join(ctx['pairs'])}). Intervals describe uncertainty over reset families "
          "conditional on these checkpoints, not over training seeds.",
          "- **Native tails.** The primary protocol does not isolate candidate quality from continuation-policy "
          "quality; the common-tail follow-up does.",
          "- Oracle availability C(K) observes outcomes; it is an optimistic finite-pool quantity, not a deployable "
          "selection score.", ""]
    if fa.get("missing"):
        L += [f"Audit artifacts missing: {', '.join(fa['missing'])}.", ""]

    # Integrity
    L += ["## Analysis integrity", "",
          f"- Records validated by `metrics.OutcomeRecordValidator` via `metrics.analyze`: "
          f"{len(ctx['val_warnings'])} warnings.",
          "- Candidate x replicate grids complete for every (tail, stratum, arm, state)."]
    xc = ctx["crosscheck"]
    if len(xc):
        for _, r in xc.iterrows():
            L.append(f"- `metrics.analyze` per-state `{r.column}`: {r.n_rows_differ}/{r.n_rows} rows differ from this "
                     f"report (max |Δ| {r.max_abs_difference:.3g}).")
        if (xc[xc.column != "m1_random_success"].n_rows_differ > 0).any():
            L.append("  `metrics.analyze` builds per-state C(K) from candidates that succeed in *any* replicate, "
                     "which is not spec M2 (C_{s,r}(K) averaged over replicates). This report uses the spec "
                     "definition through `metrics.compute_oracle_availability`.")
    L += ["- Machine-readable numbers: `report_numbers.json`. Figures: `figures/`.", ""]
    return "\n".join(L).rstrip() + "\n"


# ============================================================================
# Main entry point
# ============================================================================

def _decision(bt, agg, tail, pstrat, ref, kmax, k_values, effect_threshold, has_m3):
    diff = f"{INTENT_ARM}-{ref}"
    prim = _ci_row(bt, tail, pstrat, "C", kmax, diff)
    if prim is None:
        return None
    ci_ = lambda m, a, k=kmax: _ci_row(bt, tail, pstrat, m, k, a)  # noqa: E731
    c_i, c_a = ci_("C", INTENT_ARM), ci_("C", ref)
    c1_i, c1_a = ci_("C", INTENT_ARM, min(k_values)), ci_("C", ref, min(k_values))
    gain_i = ci_("C_gain_vs_K1", INTENT_ARM)
    m3g = ci_("M3_gain", INTENT_ARM) if has_m3 else None
    m3s_i = ci_("M3_selected", INTENT_ARM) if has_m3 else None
    m3s_a = ci_("M3_selected", ref) if has_m3 else None
    nan = math.nan
    inputs = {
        "c_intent_k16": c_i.estimate, "c_action_k16": c_a.estimate,
        "paired_diff_k16_ci": (prim.ci_lower, prim.ci_upper),
        "m3_gain": m3g.estimate if m3g is not None else nan,
        "m3_gain_ci": (m3g.ci_lower, m3g.ci_upper) if m3g is not None else (nan, nan),
        "c_intent_k1": c1_i.estimate, "c_action_k1": c1_a.estimate,
        "c_intent_gain_k16": gain_i.estimate if gain_i is not None else None,
        "m3_selected_intent": m3s_i.estimate if m3s_i is not None else None,
        "m3_selected_action": m3s_a.estimate if m3s_a is not None else None,
        "effect_threshold": effect_threshold,
    }
    verdict, reason = metrics.apply_practical_rule(**inputs)

    def show(v):
        if isinstance(v, tuple):
            return "(" + ", ".join(_f(x, 4, True) for x in v) + ")"
        return _f(v, 4) if isinstance(v, float) else str(v)
    return {"verdict": verdict, "reason": reason, "primary_row": prim,
            "inputs": {k: show(v) for k, v in inputs.items()}, "raw_inputs": inputs}


def make_report(phase_dir: str | Path, audit_dir: str | Path | None = None,
                n_resamples: int | None = None) -> dict[str, Any]:
    """Build every table, figure and the markdown report for one phase directory."""
    phase_dir = Path(phase_dir)
    manifest = load_manifest(phase_dir)
    records = load_records(phase_dir, manifest)

    experiment_id = _mget(manifest, "experiment_id") or str(records["experiment_id"].iloc[0])
    phase = str(_mget(manifest, "phase", default=phase_dir.name)).lower()
    n_resamples = int(n_resamples or _mget(manifest, "analysis.bootstrap_resamples", "bootstrap_resamples",
                                           default=2000))
    effect_threshold = float(_mget(manifest, "analysis.practical_effect_threshold", default=0.05))
    pstrat = str(_mget(manifest, "analysis.primary_stratum", "primary_stratum", default="mid")).lower()
    seed = int(_mget(manifest, "analysis.bootstrap_seed", "bootstrap_seed",
                     default=KeyedRNG(experiment_id).derive_seed("bootstrap", state_id=0)))

    arms = sorted(records["arm"].unique())
    ref_arm = next((a for a in REFERENCE_ARMS if a in arms), None)
    if ref_arm is None or INTENT_ARM not in arms:
        raise ValueError(f"Need arm {INTENT_ARM} and one of {REFERENCE_ARMS}; found {arms}")
    arms = [ref_arm, INTENT_ARM] + [a for a in arms if a not in (ref_arm, INTENT_ARM)]

    M = int(records.groupby(["arm", "state_id", "tail_mode"])["candidate_id"].nunique().max())
    R = int(records["continuation_id"].nunique())
    report_k = list(_mget(manifest, "report_k", "analysis.report_k", default=[1, 4, 16]))
    k_values = sorted({k for k in report_k if k <= M} | {1, M})
    k_primary = int(_mget(manifest, "analysis.primary_k", default=16))
    k_primary = k_primary if k_primary in k_values else max(k_values)
    exploratory = phase in EXPLORATORY_PHASES or R < 4 or bool(_mget(manifest, "exploratory", default=False))

    # Validation + metrics.analyze (also used as a cross-check)
    validator = metrics.OutcomeRecordValidator(records)
    ok, errors, warnings = validator.validate()
    if not ok:
        raise ValueError(f"Outcome record validation failed: {errors}")
    with contextlib.redirect_stdout(io.StringIO()):
        analyze_tables = metrics.analyze(manifest, records, n_resamples=n_resamples, bootstrap_seed=seed,
                                         continuation_replicates=R, report_k_values=k_values)

    state_df = compute_state_table(records, manifest, k_values, experiment_id)
    has_m3 = R >= 2
    bt = bootstrap_table(state_df, arms, ref_arm, k_values, n_resamples, seed)
    if has_m3:  # tie fraction: point estimate only (no interval)
        tie_rows = []
        for (tail, stratum, arm), g in state_df.groupby(["tail_mode", "stratum", "arm"]):
            for k in k_values:
                tie_rows.append({"tail_mode": tail, "stratum": stratum, "metric": "M3_tie_fraction", "K": k,
                                 "comparison": arm, "n_parents": g.parent_reset_id.nunique(), "n_states": len(g),
                                 "estimate": float(g[f"m3_tie_fraction_k{k}"].mean())})
        bt = pd.concat([bt, pd.DataFrame(tie_rows)], ignore_index=True)
    bt["exploratory"] = exploratory

    agg = aggregate_table(state_df, k_values, exploratory)
    per_seed = aggregate_table(state_df, k_values, exploratory, group_extra=("checkpoint_pair_id",))
    cost = cost_table(records)
    term = termination_table(records)
    xc = analyze_crosscheck(analyze_tables, state_df, k_values)
    strata = _strata(state_df)
    tail_modes = sorted(records["tail_mode"].unique(), key=lambda t: (t != "native", t))
    if pstrat not in strata:
        pstrat = strata[-1]

    # Paired-difference rows in aggregate_metrics.csv (point estimates + CI for C(K) and M1)
    diff_rows = []
    for tail in tail_modes:
        for s in strata:
            row = {"tail_mode": tail, "stratum": s, "arm": f"{INTENT_ARM}-{ref_arm}", "exploratory": exploratory}
            for metric, col_fmt in (("M1", "m1_random_success"), ("C", "c_k{k}")):
                for k in (k_values if metric == "C" else [None]):
                    r = _ci_row(bt, tail, s, metric, k, f"{INTENT_ARM}-{ref_arm}")
                    if r is not None:
                        col = col_fmt.format(k=k)
                        row[col], row[f"{col}_ci_lower"], row[f"{col}_ci_upper"] = r.estimate, r.ci_lower, r.ci_upper
                        row["n_states"] = r.n_states
            diff_rows.append(row)
    agg_out = pd.concat([agg, pd.DataFrame(diff_rows)], ignore_index=True)

    m3_label = ("M3 one-selection/one-validation preview (R=2: selection replicate 0, held-out 1), EXPLORATORY"
                if R == 2 else f"M3 with selection replicates {_m3_split(list(range(R)), manifest)[0]} and held-out "
                f"{_m3_split(list(range(R)), manifest)[1]}")
    tag = " [EXPLORATORY]" if exploratory else ""
    decisions = {t: _decision(bt, agg, t, pstrat, ref_arm, k_primary, k_values, effect_threshold, has_m3)
                 for t in tail_modes}

    # Write tables
    phase_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = phase_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    state_df.assign(exploratory=exploratory).to_csv(phase_dir / "metrics_by_state.csv", index=False)
    agg_out.to_csv(phase_dir / "aggregate_metrics.csv", index=False)
    bt.to_csv(phase_dir / "bootstrap_intervals.csv", index=False)
    per_seed.to_csv(phase_dir / "per_seed_metrics.csv", index=False)
    cost.to_csv(phase_dir / "cost_by_arm.csv", index=False)
    term.to_csv(phase_dir / "termination_breakdown.csv", index=False)
    xc.to_csv(phase_dir / "analyze_crosscheck.csv", index=False)

    # Figures
    figures = []
    for tail in tail_modes:
        targets = {
            "oracle": fig_dir / f"oracle_availability_vs_k_{tail}.png",
            "m3": fig_dir / f"m3_selected_vs_random_{tail}.png",
            "scatter": fig_dir / f"paired_state_scatter_c{k_primary}_{tail}.png",
            "cost": fig_dir / f"cost_by_arm_{tail}.png",
        }
        fig_oracle_vs_k(bt, tail, arms[:2], ref_arm, k_values, strata, tag, targets["oracle"])
        if has_m3:
            fig_m3(bt, tail, arms[:2], k_values, strata, tag, targets["m3"], m3_label)
        else:
            targets.pop("m3")
        fig_scatter(state_df, tail, ref_arm, k_primary, strata, tag, targets["scatter"])
        fig_cost(cost, tail, arms[:2], strata, tag, targets["cost"])
        figures += [str(t) for t in targets.values()]

    ctx = {"bt": bt, "agg": agg, "tail_modes": tail_modes, "ref_arm": ref_arm, "k_primary": k_primary, "k_values": k_values,
               "primary_stratum": pstrat, "exploratory": exploratory, "phase": phase, "experiment_id": experiment_id,
               "pairs": sorted(map(str, records["checkpoint_pair_id"].unique())), "n_records": len(records), "M": M, "R": R,
               "n_resamples": n_resamples, "seed": seed, "strata": strata, "arms": arms[:2], "has_m3": has_m3, "m3_label": m3_label,
               "decisions": decisions, "cost": cost, "audit": audit_facts(audit_dir or DEFAULT_AUDIT_DIR),
               "val_warnings": warnings, "crosscheck": xc}
    md = build_markdown(ctx)
    (phase_dir / "experiment_report.md").write_text(md)

    numbers = {
        "phase": phase, "exploratory": exploratory, "experiment_id": experiment_id, "ref_arm": ref_arm,
        "k_values": k_values, "k_primary": k_primary, "primary_stratum": pstrat, "M": M, "R": R,
        "n_resamples": n_resamples, "bootstrap_seed": seed,
        "decisions": {t: (None if d is None else {"verdict": d["verdict"], "reason": d["reason"],
                                                  "preview_only": exploratory,
                                                  "inputs": {k: (list(v) if isinstance(v, tuple) else v)
                                                             for k, v in d["raw_inputs"].items()}})
                      for t, d in decisions.items()},
        "intervals": json.loads(bt.to_json(orient="records")),
    }
    (phase_dir / "report_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    return {"phase_dir": str(phase_dir), "figures": figures, "report": str(phase_dir / "experiment_report.md"),
            "exploratory": exploratory, "decisions": numbers["decisions"], "state_table": state_df,
            "bootstrap": bt, "aggregate": agg_out, "analyze_tables": analyze_tables}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the intent-selection experiment report for one phase.")
    ap.add_argument("phase_dir")
    ap.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    ap.add_argument("--n-resamples", type=int, default=None)
    args = ap.parse_args()
    out = make_report(args.phase_dir, audit_dir=args.audit_dir, n_resamples=args.n_resamples)
    print(f"Wrote {out['report']}")
    for f in out["figures"]:
        print(f"Wrote {f}")


if __name__ == "__main__":
    main()
