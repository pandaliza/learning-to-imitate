"""Analyze the tool_hang I-PATH vs no-intent fresh-reset evaluation (scripts/tool_hang_fresh_eval.sh).

Reads $ROOT/fresh_eval/<run>_<ckpt>_r<rollout>.jsonl and prints a markdown report:
  1. success per arm x training seed on the shared reset pool, and the paired per-reset difference with a
     bootstrap CI over resets;
  2. determinism of the MIP baseline across rollout seeds;
  3. hard starts, cross-fitted so the selection does not bias the comparison: for each held-out baseline
     seed j, H_j = resets that BOTH other baseline seeds fail on every rollout; compare held-out baseline
     seed j against I-PATH (all seeds, all rollouts) on H_j. The naive set "all 3 baseline seeds fail" is
     reported too, but the baseline is 0% there by construction;
  4. the same in reverse (starts I-PATH consistently misses).
Usage: python scripts/tool_hang_intent_analysis.py [--root DIR] [--ckpt best|latest]
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ARMS = {"mip": "no intent (MIP + MLP)", "ipath": "I-PATH"}  # add "ipath_sampled" to compare that arm


def load(root, ckpt):
    """-> {arm: {seed: {rollout: {reset_seed: success}}}}"""
    data = defaultdict(lambda: defaultdict(dict))
    for path in sorted(glob.glob(os.path.join(root, "fresh_eval", f"state_*_{ckpt}_r*.jsonl"))):
        m = re.match(rf"state_(\w+?)_s(\d+)_{ckpt}_r(\d+)\.jsonl", os.path.basename(path))
        if not m or m.group(1) not in ARMS:
            continue
        arm, seed, rollout = m.group(1), int(m.group(2)), int(m.group(3))
        with open(path) as f:
            data[arm][seed][rollout] = {r["reset_seed"]: r["success"] for r in map(json.loads, f)}
    return data


def matrix(runs, resets):
    """{rollout: {reset: success}} -> (n_rollouts, n_resets) array."""
    return np.array([[runs[r][s] for s in resets] for r in sorted(runs)])


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def hard_sets(per_seed, j):
    """Resets where every seed except j fails on every rollout."""
    others = [per_seed[k] for k in per_seed if k != j]
    return np.all([m.max(axis=0) == 0 for m in others], axis=0)


def crossfit(sel_arm, other_arm, label_sel, label_other):
    lines = [f"| held-out {label_sel} seed | # starts | {label_sel} held-out seed | {label_other} (all seeds) |",
             "|---|---|---|---|"]
    tot = [0, 0.0, 0.0]
    for j in sorted(sel_arm):
        h = hard_sets(sel_arm, j)
        n = int(h.sum())
        a = sel_arm[j][:, h].mean() if n else float("nan")
        b = np.mean([m[:, h].mean() for m in other_arm.values()]) if n else float("nan")
        lines.append(f"| s{j} | {n} | {100 * a:.1f}% | {100 * b:.1f}% |")
        if n:
            tot[0] += n; tot[1] += a * n; tot[2] += b * n
    if tot[0]:
        lines.append(f"| pooled | {tot[0]} | {100 * tot[1] / tot[0]:.1f}% | {100 * tot[2] / tot[0]:.1f}% |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/edrive4/ldahiya/mip_runs/tool_hang")
    ap.add_argument("--ckpt", default="best")
    args = ap.parse_args()
    data = load(args.root, args.ckpt)
    assert set(data) == set(ARMS), f"need both arms, found {sorted(data)}"
    pools = [set(runs[r]) for arm in data.values() for runs in arm.values() for r in runs]
    resets = sorted(set.intersection(*pools))
    assert all(len(p) == len(resets) for p in pools), "every run must cover the same reset pool"
    M = {arm: {s: matrix(runs, resets) for s, runs in data[arm].items()} for arm in data}

    out = [f"# tool_hang: I-PATH vs no intent on fresh resets (checkpoint: {args.ckpt})", "",
           f"Shared pool: {len(resets)} resets (seeds {resets[0]}..{resets[-1]}).", "",
           "## 1. Success on the pool", "", "| arm | per training seed (mean over rollouts) | mean ± sd over seeds |",
           "|---|---|---|"]
    for arm in ARMS:
        per = [M[arm][s].mean() for s in sorted(M[arm])]
        out.append(f"| {ARMS[arm]} | " + " / ".join(f"s{s} {100 * v:.1f}%" for s, v in zip(sorted(M[arm]), per))
                   + f" ({M[arm][0].shape[0]} rollouts each) | {100 * np.mean(per):.1f} ± {100 * np.std(per, ddof=1):.1f}% |")
    p = {arm: np.mean([m.mean(axis=0) for m in M[arm].values()], axis=0) for arm in ARMS}
    d = p["ipath"] - p["mip"]
    lo, hi = boot_ci(d)
    out += ["", f"Paired per-reset difference (I-PATH − no intent): **{100 * d.mean():+.1f} pts**, "
            f"95% bootstrap CI over resets [{100 * lo:+.1f}, {100 * hi:+.1f}]. "
            f"Resets better under I-PATH: {(d > 0).sum()}, worse: {(d < 0).sum()}, equal: {(d == 0).sum()}. "
            "(The CI treats the 3 training seeds per arm as fixed; with 3 seeds, seed-to-seed spread above is the "
            "other uncertainty to weigh.)"]

    out += ["", "## 2. Baseline determinism across rollout seeds", ""]
    for s in sorted(M["mip"]):
        m = M["mip"][s]
        agree = np.mean(np.all(m == m[0], axis=0))
        out.append(f"- no intent s{s}: identical outcome on all {m.shape[0]} rollouts for {100 * agree:.1f}% of resets")
    for s in sorted(M["ipath"]):
        m = M["ipath"][s]
        out.append(f"- I-PATH s{s}: identical outcome on all {m.shape[0]} rollouts for "
                   f"{100 * np.mean(np.all(m == m[0], axis=0)):.1f}% of resets")

    out += ["", "## 3. Starts the baseline consistently misses (cross-fitted)", "",
            "H_j = starts that both baseline seeds other than j fail on every rollout. The held-out baseline seed "
            "and I-PATH were not used to pick H_j, so both rates are unbiased on it.", ""]
    out += crossfit(M["mip"], M["ipath"], "no-intent", "I-PATH")
    naive = np.all([m.max(axis=0) == 0 for m in M["mip"].values()], axis=0)
    out += ["", f"Naive set (all {len(M['mip'])} baseline seeds fail every rollout): {int(naive.sum())} starts; "
            f"baseline 0% by construction; I-PATH {100 * np.mean([m[:, naive].mean() for m in M['ipath'].values()]):.1f}%."
            if naive.any() else "", ""]
    if naive.any():
        solved = [resets[i] for i in np.flatnonzero(naive)
                  if np.mean([M["ipath"][s][:, i].max() for s in M["ipath"]]) >= 2 / 3]
        out.append(f"Of these, solved by ≥2 of 3 I-PATH seeds (at least one rollout): {len(solved)} — "
                   f"reset seeds {solved[:30]}{' …' if len(solved) > 30 else ''}")

    out += ["", "## 4. Reverse: starts I-PATH consistently misses (cross-fitted)", ""]
    out += crossfit(M["ipath"], M["mip"], "I-PATH", "no-intent")
    print("\n".join(out))


if __name__ == "__main__":
    main()
