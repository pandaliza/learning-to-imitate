"""Summaries for the B1K diagnostic arm (all tables DIAGNOSTIC).

python summarize_b1k.py gate TASK          -> dev-gate table + verdict for A-MIX (b1k_diag_protocol.md §7)
python summarize_b1k.py test TASK          -> per-arm test table + paired differences vs A-MIX / A-SAFE
Paired differences: resample the 50 shared test resets (5,000 bootstrap draws, seed 0), percentile 95% CI.
Wilson 95% CIs for single proportions. Infra errors are excluded and counted separately.
"""
import glob
import json
import math
import os
import sys

import numpy as np

EV = "/edrive1/lizad/oopsieverse_full_study/runs/b1k_diag/eval"
ARMS = ["amix", "asafe", "imean", "ipath", "ipathg", "auxpathg"]
LABEL = {"amix": "A-MIX", "asafe": "A-SAFE", "imean": "I-MEAN", "ipath": "I-PATH", "ipathg": "I-PATH-G",
         "auxpathg": "AUX-PATH-G"}


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(task, arm, pool, rseed=0):
    d = os.path.join(EV, f"{arm}_{task}_b1kv1_s0", pool)
    recs = {}
    for f in sorted(glob.glob(f"{d}/{pool}*__r{rseed}.json")):
        r = json.load(open(f))
        recs[r["reset"]["index"]] = r
    return recs


def row(recs):
    ok = [r for r in recs.values() if "infra_error" not in r]
    n = len(ok)
    c = lambda key: sum(bool(r.get(key)) for r in ok)  # noqa: E731
    out = {"n": n, "infra": len(recs) - n, "success": c("success"), "primary_safe_success": c("primary_safe_success"),
           "secondary_safe_success": c("secondary_safe_success"), "compat_safe_success": c("compat_safe_success"),
           "damaging_success": sum(r["outcome"] == "damaging_success" for r in ok),
           "safe_failure": sum(r["outcome"] == "safe_failure" for r in ok),
           "damaging_failure": sum(r["outcome"] == "damaging_failure" for r in ok),
           "invalid": c("invalid_action"),
           "mean_worst_loss": float(np.mean([r["worst_entity_loss"] for r in ok])) if ok else float("nan"),
           "mean_steps": float(np.mean([r["steps"] for r in ok])) if ok else float("nan"),
           "median_max_impact": float(np.median([r["max_impact_force_N"] for r in ok])) if ok else float("nan")}
    ents = sorted({e for r in ok for e in r["entity_loss"]})
    out["mean_entity_loss"] = {e: float(np.mean([r["entity_loss"].get(e, 0.0) for r in ok])) for e in ents}
    return out


def fmt_pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / max(n, 1):.0f}%, {lo:.0f}–{hi:.0f})"


def paired(a, b, key, B=5000, seed=0):
    ids = sorted(set(i for i in a if "infra_error" not in a[i]) & set(i for i in b if "infra_error" not in b[i]))
    if not ids:
        return None
    x = np.array([float(bool(a[i][key])) - float(bool(b[i][key])) for i in ids])
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(B, len(x)))].mean(1)
    return 100 * x.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), len(ids)


def main():
    mode, task = sys.argv[1], sys.argv[2]
    if mode == "gate":
        r = row(load(task, "amix", "dev"))
        verdict = r["success"] >= 10 and r["damaging_success"] >= 5
        print(json.dumps(dict(r, gate_pass=verdict, rule=">=10 successes and >=5 damaging successes on 30 dev resets"), indent=1))
        return
    lines = [f"### {task}: random-policy test evaluation (DIAGNOSTIC; 50 frozen test resets x rollout seed 0; training seed 0)", "",
             "| Arm | n | Task success | Safe success (primary) | Damaging success | Safe failure | Damaging failure | Secondary safe success | Compat safe success | Mean worst-entity loss | Mean steps | Infra |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    data = {a: load(task, a, "test") for a in ARMS}
    for a in ARMS:
        if not data[a]:
            continue
        r = row(data[a])
        lines.append(f"| {LABEL[a]} | {r['n']} | {fmt_pct(r['success'], r['n'])} | {fmt_pct(r['primary_safe_success'], r['n'])} | "
                     f"{r['damaging_success']} | {r['safe_failure']} | {r['damaging_failure']} | {r['secondary_safe_success']} | "
                     f"{r['compat_safe_success']} | {r['mean_worst_loss']:.1f} | {r['mean_steps']:.0f} | {r['infra']} |")
    lines += ["", f"Paired differences on shared resets (percentage points, bootstrap 95% CI over resets; DIAGNOSTIC):", "",
              "| Comparison | Δ safe success (primary) | Δ task success | Δ secondary safe success | n paired |", "|---|---|---|---|---|"]
    for ref in ("amix", "asafe"):
        for a in ARMS:
            if a == ref or not data[a] or not data[ref]:
                continue
            ps = paired(data[a], data[ref], "primary_safe_success")
            ts = paired(data[a], data[ref], "success")
            ss = paired(data[a], data[ref], "secondary_safe_success")
            if ps is None:
                continue
            f = lambda t: f"{t[0]:+.0f} ({t[1]:+.0f}, {t[2]:+.0f})"  # noqa: E731
            lines.append(f"| {LABEL[a]} − {LABEL[ref]} | {f(ps)} | {f(ts)} | {f(ss)} | {ps[3]} |")
    print("\n".join(lines))
    json.dump({a: row(data[a]) for a in ARMS if data[a]}, open(os.path.join(EV, f"summary_test_{task}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
