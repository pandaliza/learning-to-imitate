"""Summary of the all-60 I-PATH rollout-time intent ablation (spec addendum 2026-10-29 23:31 UTC; diagnostic only).

  python summarize_all60_ablation.py [--out-md reports/wipe_v2/all60_ablation.md] [--out-json ...]

Cells: arm x mode on P-layout60, step_30000, rollout seeds 0,1,2 (140 resets x 3 = 420 per cell).
  sampled   runs/wipe_v2/all60/eval/<rid>/P-layout60__step_30000__*      (existing all60 records, untouched)
  shuffled  runs/wipe_v2/all60/ablation/<rid>/P-layout60__shuffled__step_30000__*
  zero      runs/wipe_v2/all60/ablation/<rid>/P-layout60__zero__step_30000__*
Metrics reuse summarize_wipe_v2.stats/by_group/wilson/load (same definitions as the other wipe v2 reports).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wv2_common as C  # noqa: E402
import summarize_wipe_v2 as S  # noqa: E402

RUNS = [f"ipath_wipe_countertop_all60_s{s}" for s in (0, 1, 2)]
MODES = ["sampled", "shuffled", "zero"]
GROUPS = [f"g0{i}" for i in range(7)]
POOL = "P-layout60"
EXPECTED_N = 420
ALL60 = os.path.join(C.V2_RUNS, "all60")


def cell_dir(rid, mode):
    if mode == "sampled":
        ds = glob.glob(os.path.join(ALL60, "eval", rid, f"{POOL}__step_30000__*"))
    else:
        ds = glob.glob(os.path.join(ALL60, "ablation", rid, f"{POOL}__{mode}__step_30000__*"))
    assert len(ds) == 1, (rid, mode, ds)  # exactly one eval key per cell
    return ds[0]


def load_cell(rid, mode):
    d = cell_dir(rid, mode)
    recs, infra = S.load(d)
    if mode != "sampled":
        assert all(r.get("intent_ablation") == mode for r in recs), (d, mode)
    prov = {}
    for r in recs:
        for k, v in (r.get("intent_ablation_provenance") or {}).items():
            prov[k] = prov.get(k, 0) + v
    return {"dir": d, "key": d.rsplit("__", 1)[-1], "recs": recs, "infra": infra, "prov": prov}


def pct(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def ms(vals, pc=True):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return "–"
    sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    if pc:
        return f"{100 * v.mean():.1f} ± {100 * sd:.1f}%"
    return f"{v.mean():.3f} ± {sd:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", default=os.path.join(C.V2_REPORTS, "all60_ablation.md"))
    ap.add_argument("--out-json", default=os.path.join(C.V2_REPORTS, "all60_ablation.json"))
    a = ap.parse_args()

    cells, out = {}, {"pool": POOL, "expected_n_per_cell": EXPECTED_N, "cells": {}}
    for rid in RUNS:
        for m in MODES:
            c = load_cell(rid, m)
            c["stats"] = S.stats(c["recs"])
            c["groups"] = S.by_group(c["recs"])
            cells[(rid, m)] = c
            out["cells"][f"{rid}|{m}"] = {"dir": c["dir"], "key": c["key"], "n": c["stats"]["n"], "infra_errors": c["infra"],
                                          "provenance": c["prov"], "stats": c["stats"],
                                          "groups": {g: st for g, st in c["groups"].items()}}

    L = []
    L.append("# all-60 I-PATH rollout-time intent ablation (P-layout60)\n")
    L.append("Spec: `reports/wipe_v2/wipe_v2_spec.md`, addendum 2026-10-29 23:31 UTC. **Diagnostic only; no pass/fail.** "
             "Arms: I-PATH all-60 checkpoints s0, s1, s2 at step 30000. Modes: `sampled` (existing records), `shuffled` "
             "(intent decoded for the previous chunk of a different reset in the same batch), `zero` (normalised intent token = "
             "training mean). Pool P-layout60 (140 resets, 20 per layout g00-g06), rollout seeds 0,1,2, so n = 420 per cell. "
             "Alias on, timeout 1250, exec horizon 8. Wilson 95% intervals; median is of the final proportion wiped.\n")

    L.append("## Per checkpoint and mode\n")
    L.append("| Checkpoint | Mode | n | Task success k/n (rate) | Wilson 95% | Primary safe k (rate) | Mean prop. wiped | "
             "Median prop. wiped | Infra errors | Eval key |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for rid in RUNS:
        for m in MODES:
            c = cells[(rid, m)]
            st = c["stats"]
            lo, hi = st["wilson95"]
            L.append(f"| {rid.replace('ipath_wipe_countertop_', '')} | {m} | {st['n']} | {st['success']}/{st['n']} "
                     f"({pct(st['success_rate'])}) | {pct(lo)}–{pct(hi)} | {st['primary_safe_success']} "
                     f"({pct(st['primary_safe_success'] / st['n'] if st['n'] else float('nan'))}) | "
                     f"{st['mean_prop_wiped']:.3f} | {st['prop_wiped_quantiles'].get(50, float('nan')):.3f} | "
                     f"{c['infra']} | `{c['key']}` |")
    L.append("")

    L.append("## Mode mean ± sd over the three checkpoints (overall, P-layout60)\n")
    L.append("| Mode | Task success | Primary safe success | Mean prop. wiped | Median prop. wiped | Pooled k/n (3 ckpts) |")
    L.append("|---|---|---|---|---|---|")
    mode_means = {}
    for m in MODES:
        sts = [cells[(rid, m)]["stats"] for rid in RUNS]
        rates = [s["success_rate"] for s in sts]
        psafe = [s["primary_safe_success"] / s["n"] for s in sts]
        mw = [s["mean_prop_wiped"] for s in sts]
        med = [s["prop_wiped_quantiles"][50] for s in sts]
        k = sum(s["success"] for s in sts)
        n = sum(s["n"] for s in sts)
        mode_means[m] = {"success_rate": rates, "primary_safe_rate": psafe, "mean_prop_wiped": mw, "median_prop_wiped": med,
                         "pooled": [k, n]}
        L.append(f"| {m} | {ms(rates)} | {ms(psafe)} | {ms(mw, pc=False)} | {ms(med, pc=False)} | {k}/{n} ({pct(k / n)}) |")
    L.append("")
    out["mode_means"] = mode_means

    L.append("## Per layout group (mean ± sd over the three checkpoints; each checkpoint contributes 20 resets x 3 seeds = 60 episodes per layout)\n")
    hdr = "| Mode | Metric | " + " | ".join(GROUPS) + " |"
    L.append(hdr)
    L.append("|---|---|" + "---|" * len(GROUPS))
    per_group = {}
    for m in MODES:
        row_s, row_w, row_n = [], [], []
        for g in GROUPS:
            gs = [cells[(rid, m)]["groups"].get(g) for rid in RUNS]
            gs = [x for x in gs if x]
            rates = [x["success_rate"] for x in gs]
            mw = [x["mean_prop_wiped"] for x in gs]
            per_group.setdefault(m, {})[g] = {"success_rate": rates, "mean_prop_wiped": mw, "n": [x["n"] for x in gs]}
            row_s.append(ms(rates))
            row_w.append(ms(mw, pc=False))
            row_n.append(str(sum(x["n"] for x in gs)))
        L.append(f"| {m} | task success | " + " | ".join(row_s) + " |")
        L.append(f"| {m} | mean prop. wiped | " + " | ".join(row_w) + " |")
        L.append(f"| {m} | n (3 ckpts) | " + " | ".join(row_n) + " |")
    L.append("")
    out["per_group"] = per_group

    L.append("## Provenance of the shuffled intents (chunk counts summed over the 420 episodes of each checkpoint)\n")
    for rid in RUNS:
        L.append(f"- {rid}: {json.dumps(cells[(rid, 'shuffled')]['prov'], sort_keys=True)}")
    L.append("")
    open(a.out_md, "w").write("\n".join(L) + "\n")
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
