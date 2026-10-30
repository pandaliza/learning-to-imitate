"""Aggregate of the per-training-seed selection analyses (analyze_selection.py outputs) — spec §11/§13.

Selection is a prespecified SECONDARY family (spec §13): unadjusted 95% intervals, no Holm (the spec defines Holm only
for the primary confirmatory family: intent arms vs A-MIX on random-policy safe success).

Estimand per arm/variant and metric: equal-weight mean over training seeds of the per-seed mean over eligible parents.
Uncertainty: crossed bootstrap over parents (the same sel_test_parents bank is shared by all training seeds) and
training seeds, 5,000 resamples (seed fixed): per resample draw 3 seeds and 100 parent indices with replacement;
statistic = mean over drawn seeds of the mean over drawn parents that are present (eligible state) for that seed.
Paired differences vs A-MIX use the same draws on (seed, parent) cells where both groups exist, preserving all
arm/candidate/tail dependencies (each cell's metric already integrates its 16 candidates x 4 tails).

  python aggregate_selection.py --inputs a_s0.json a_s1.json a_s2.json --out agg.json
"""
import argparse
import json

import numpy as np

PRED = "primary_safe_success"
KS = (1, 4, 16)
B = 5000
SEED = 20261008


def metrics_of(row):
    d = row[PRED]
    out = {"random_K1": d["random_K1"]}
    for K in KS:
        out[f"oracle_C{K}"] = d[f"oracle_C{K}_all"]
        out[f"oracle_C{K}_validate"] = d[f"oracle_C{K}_validate"]
        out[f"select_K{K}"] = d[f"select_K{K}"]
        out[f"select_K{K}_random_baseline"] = d[f"select_K{K}_random_baseline"]
        out[f"select_K{K}_gain_vs_own_random"] = d[f"select_K{K}"] - d[f"select_K{K}_random_baseline"]
        out[f"select_K{K}_gain_vs_own_K1"] = d[f"select_K{K}"] - d["random_K1"]
        out[f"select_K{K}_tie_rate"] = d[f"select_K{K}_tie_rate"]
    for p in ("success", "secondary_safe_success", "compat_safe_success"):
        out[f"{p}__random_K1"] = row[p]["random_K1"]
        out[f"{p}__select_K16"] = row[p]["select_K16"]
        out[f"{p}__oracle_C16"] = row[p]["oracle_C16_all"]
    return out


def crossed(X, draws):
    """X: [S, P] with NaN for missing cells. Returns point, ci95."""
    def stat(Xs):
        with np.errstate(invalid="ignore"):
            per_seed = np.nanmean(Xs, axis=1)
        return float(np.nanmean(per_seed))
    point = stat(X)
    bs = []
    for s_idx, p_idx in draws:
        bs.append(stat(X[s_idx][:, p_idx]))
    bs = np.array(bs)
    bs = bs[~np.isnan(bs)]
    return {"mean": point, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "per_seed": [None if np.all(np.isnan(r)) else float(np.nanmean(r)) for r in X],
            "n_cells": int(np.sum(~np.isnan(X)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = [json.load(open(f)) for f in a.inputs]
    seeds = [r["training_seed"] for r in runs]
    parents = sorted({int(p) for r in runs for rows in r["per_state"].values() for p in rows})
    P = max(parents) + 1
    S = len(runs)
    arms = sorted({k for r in runs for k in r["per_state"]})
    cube = {}  # arm -> metric -> [S, P]
    for arm in arms:
        for si, r in enumerate(runs):
            for p, row in r["per_state"].get(arm, {}).items():
                for mname, v in metrics_of(row).items():
                    cube.setdefault(arm, {}).setdefault(mname, np.full((S, P), np.nan))[si, int(p)] = v
    rng = np.random.RandomState(SEED)
    draws = [(rng.randint(0, S, S), rng.randint(0, P, P)) for _ in range(B)]
    out = {"inputs": a.inputs, "training_seeds": seeds, "n_parents_bank": P, "predicate": PRED,
           "family": "selection = prespecified secondary family (spec §13); unadjusted 95% CIs, no Holm",
           "bootstrap": f"crossed over parents (shared bank) x training seeds, {B} resamples, seed {SEED}",
           "incomplete_groups_per_seed": [r["incomplete_groups"] for r in runs], "arms": {}, "paired_vs_amix": {}}
    for arm in arms:
        out["arms"][arm] = {m: crossed(X, draws) for m, X in cube[arm].items()}
        out["arms"][arm]["n_states_per_seed"] = [int(np.sum(~np.isnan(cube[arm]["random_K1"][s]))) for s in range(S)]
    base = cube.get("amix__primary")
    for arm in arms:
        if arm == "amix__primary" or base is None:
            continue
        c = {}
        for m in ("random_K1", "oracle_C1", "oracle_C4", "oracle_C16", "select_K1", "select_K4", "select_K16",
                  "select_K4_gain_vs_own_K1", "select_K16_gain_vs_own_K1",
                  "select_K4_gain_vs_own_random", "select_K16_gain_vs_own_random"):
            D = cube[arm][m] - base[m]   # NaN unless both cells exist
            c[m] = crossed(D, draws)
        out["paired_vs_amix"][f"{arm} - amix__primary"] = c
    json.dump(out, open(a.out, "w"), indent=1)
    for arm in arms:
        x = out["arms"][arm]
        print(arm, {k: round(x[k]["mean"], 3) for k in ("random_K1", "oracle_C4", "oracle_C16", "select_K4", "select_K16")})


if __name__ == "__main__":
    main()
