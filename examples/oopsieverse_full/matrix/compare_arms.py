"""Primary policy comparison (spec §10/§13): random-policy test evaluation at the fixed 30k checkpoint.

Paired differences vs A-MIX (and vs A-SAFE) with a crossed bootstrap over shared test reset families and training
seeds (5,000 resamples), rollout seeds averaged within (training seed, reset family). Holm correction over the
primary confirmatory family {I-MEAN, I-PATH, I-PATH-G} vs A-MIX on primary safe success. Seed-level points shown.

  python compare_arms.py --task pick_egg
"""

import argparse
import glob
import json
import os

import numpy as np

RUNS = "/edrive1/lizad/oopsieverse_full_study/runs/matrix_eval"
STUDY = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study"
ARMS = ["amix", "asafe", "imean", "ipath", "ipathg", "auxpathg"]
NAMES = {"amix": "A-MIX", "asafe": "A-SAFE", "imean": "I-MEAN", "ipath": "I-PATH", "ipathg": "I-PATH-G",
         "auxpathg": "AUX-PATH-G", "ssmix": "safe-sized mix", "cmean": "canonical mean"}
FAMILY = ["imean", "ipath", "ipathg"]
METRICS = ["primary_safe_success", "success", "secondary_safe_success", "compat_safe_success", "success_h010"]
SEEDS = [0, 1, 2]
B = 5000
TARGET_PP = 5.0


def load(task, arm, seed):
    d = sorted(glob.glob(f"{RUNS}/{arm}_{task}_v1_s{seed}/test_s30000_*"))
    if len(d) != 1:
        return None
    recs = [json.load(open(f)) for f in glob.glob(f"{d[0]}/{task}__eval_test__*__r*.json")]
    return recs


def table(task, arms):
    """Y[metric][arm] -> array (seeds, families) of rollout-seed means, plus family ids and in_support mask."""
    fams, data, insup = None, {m: {} for m in METRICS}, {}
    for a in arms:
        per_seed = []
        for s in SEEDS:
            recs = load(task, a, s)
            if recs is None or len(recs) != 600:
                raise SystemExit(f"{a} seed {s}: {0 if recs is None else len(recs)} records (need 600)")
            by = {}
            for r in recs:
                by.setdefault(r["reset"]["family"], []).append(r)
                insup[r["reset"]["family"]] = r["reset"]["in_support"]
            per_seed.append(by)
        f = sorted(per_seed[0])
        assert all(sorted(b) == f for b in per_seed), f"{a}: reset families differ across seeds"
        if fams is None:
            fams = f
        assert f == fams, f"{a}: reset families differ from the first arm"
        for m in METRICS:
            data[m][a] = np.array([[np.mean([bool(r[m]) for r in by[x]]) for x in fams] for by in per_seed])
    return fams, data, np.array([insup[x] for x in fams])


def crossed_boot(diff, rng):
    """diff: (S, F) paired per-cell differences. Resample seeds and families independently (crossed)."""
    S, F = diff.shape
    si = rng.integers(0, S, size=(B, S))
    fi = rng.integers(0, F, size=(B, F))
    out = np.empty(B)
    for b in range(B):
        out[b] = diff[np.ix_(si[b], fi[b])].mean()
    return out


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_egg")
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    arms = a.arms.split(",")
    fams, Y, insup = table(a.task, arms)
    out = {"task": a.task, "n_reset_families": len(fams), "training_seeds": SEEDS, "rollout_seeds_per_family": 3,
           "bootstrap": f"crossed over training seeds x reset families, B={B}, seed 20261008", "arms": {}, "contrasts": []}
    for arm in arms:
        out["arms"][arm] = {m: {"mean_pct": 100 * Y[m][arm].mean(),
                                "seed_pct": [100 * x for x in Y[m][arm].mean(1)]} for m in METRICS}
        out["arms"][arm]["in_support_primary_safe_pct"] = 100 * Y["primary_safe_success"][arm][:, insup].mean()
        out["arms"][arm]["out_of_support_primary_safe_pct"] = 100 * Y["primary_safe_success"][arm][:, ~insup].mean()
    for ref in ("amix", "asafe"):
        for arm in arms:
            if arm == ref:
                continue
            row = {"arm": arm, "ref": ref}
            for m in METRICS:
                rng = np.random.default_rng([20261008, METRICS.index(m), ARMS.index(arm), ARMS.index(ref)])
                d = Y[m][arm] - Y[m][ref]
                bs = crossed_boot(d, rng)
                p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
                row[m] = {"diff_pp": 100 * d.mean(), "ci95_pp": [100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)],
                          "p_boot": p, "seed_diff_pp": [100 * x for x in d.mean(1)]}
            out["contrasts"].append(row)
    fam = [c for c in out["contrasts"] if c["ref"] == "amix" and c["arm"] in FAMILY]
    adj = holm(np.array([c["primary_safe_success"]["p_boot"] for c in fam]))
    for c, p in zip(fam, adj):
        c["primary_safe_success"]["p_holm"] = float(p)
        c["meets_project_rule"] = bool(c["primary_safe_success"]["diff_pp"] >= TARGET_PP
                                       and c["success"]["diff_pp"] >= -TARGET_PP)
    os.makedirs(f"{STUDY}/matrix/comparison", exist_ok=True)
    jp = f"{STUDY}/matrix/comparison/{a.task}_primary_comparison.json"
    json.dump(out, open(jp, "w"), indent=1)

    L = [f"# {a.task}: primary policy comparison (random-policy test eval, 30k checkpoint)", "",
         f"{len(fams)} shared test reset families × 3 rollout seeds × training seeds {SEEDS}. Crossed bootstrap over "
         f"training seeds and reset families (B={B}). Rollout seeds averaged within cells. Holm over "
         "{I-MEAN, I-PATH, I-PATH-G} vs A-MIX on primary safe success (spec §13). Project rule (chosen tolerance, "
         "spec §10): safe success +5 pp or more with task success loss of 5 pp or less.", "",
         "## Per arm (mean over seeds; seed values in brackets)", "",
         "| Arm | Primary safe success % | Task success % | Secondary safe % | Compat safe % | Success at 0.10 m % | Safe % in / out of support |",
         "|---|---|---|---|---|---|---|"]
    for arm in arms:
        r = out["arms"][arm]
        f = lambda m: f"{r[m]['mean_pct']:.1f} [{', '.join(f'{x:.1f}' for x in r[m]['seed_pct'])}]"
        L.append(f"| {NAMES[arm]} | {f('primary_safe_success')} | {f('success')} | {r['secondary_safe_success']['mean_pct']:.1f} | "
                 f"{r['compat_safe_success']['mean_pct']:.1f} | {r['success_h010']['mean_pct']:.1f} | "
                 f"{r['in_support_primary_safe_pct']:.1f} / {r['out_of_support_primary_safe_pct']:.1f} |")
    for ref in ("amix", "asafe"):
        L += ["", f"## Paired differences vs {NAMES[ref]} (percentage points, 95% CI)", "",
              "| Arm | Primary safe Δ | seed Δs | p (boot) | p (Holm) | Task success Δ | Secondary safe Δ | Meets +5/−5 rule |",
              "|---|---|---|---|---|---|---|---|"]
        for c in out["contrasts"]:
            if c["ref"] != ref:
                continue
            ps, su, se = c["primary_safe_success"], c["success"], c["secondary_safe_success"]
            ci = lambda x: f"{x['diff_pp']:+.1f} [{x['ci95_pp'][0]:+.1f}, {x['ci95_pp'][1]:+.1f}]"
            L.append(f"| {NAMES[c['arm']]} | {ci(ps)} | {', '.join(f'{x:+.1f}' for x in ps['seed_diff_pp'])} | {ps['p_boot']:.3f} | "
                     f"{ps.get('p_holm', float('nan')):.3f} | {ci(su)} | {ci(se)} | "
                     f"{'yes' if c.get('meets_project_rule') else ('no' if 'meets_project_rule' in c else '—')} |")
    L += ["", "p (Holm) applies only to the confirmatory family; other rows are descriptive. Three training seeds give "
          "weak training-variance estimates (spec §13)."]
    open(f"{STUDY}/matrix/comparison/{a.task}_primary_comparison.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
