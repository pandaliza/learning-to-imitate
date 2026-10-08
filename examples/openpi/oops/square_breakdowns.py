"""Pre-registered secondary breakdowns for the square-mh-state shift study (declared before shifted outcomes existed).

1. Direction class: shift is y-dominant (|dy| > |dx|) vs x-dominant. The demonstrated nut start positions span
   ~11 cm in y but only ~5 mm in x, so y-dominant shifts stay closer to the demonstrated range.
2. Distance of the shifted start outside the bounding box of demonstrated start positions (bins, metres).
Population = valid matched cases (as the primary analysis). Raw counts only; no hypothesis tests.

  python examples/openpi/oops/square_breakdowns.py --manifest-dir ... --episodes-dir ... --out-dir ...
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import oops_common as oc  # noqa: E402

BINS = [(0.0, 0.0), (1e-9, 0.02), (0.02, 0.05), (0.05, 0.08), (0.08, 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--baseline-arm", default="A_baseline")
    ap.add_argument("--intent-arm", default="I_flow_intent")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    cases = {c["case_id"]: c for c in oc.read_jsonl(pathlib.Path(args.manifest_dir) / "cases.jsonl")}

    def load(arm):
        return {(r["case_id"], r["policy_seed"]): r for p in (pathlib.Path(args.episodes_dir) / arm).glob("*.jsonl")
                for r in oc.read_jsonl(p) if not r.get("infrastructure_error")}
    A, I = load(args.baseline_arm), load(args.intent_arm)
    keys = [k for k in A if k in I and cases[k[0]]["validity_status"] == "valid"]

    by_dir = collections.defaultdict(lambda: [0, 0, 0])
    by_bin = collections.defaultdict(lambda: [0, 0, 0])
    for k in keys:
        c = cases[k[0]]
        a, i = int(A[k]["success"]), int(I[k]["success"])
        d = round(c["delta_m"], 6)
        cls = "none" if d == 0 else ("y_dominant" if abs(c["dy_m"]) > abs(c["dx_m"]) else "x_dominant")
        for tab, key in ((by_dir, (d, cls)), (by_bin, next(f"{lo:.2f}-{hi:.2f}" if hi > 0 else "inside"
                                                             for lo, hi in BINS
                                                             if (hi == 0 and c["dist_outside_demo_box_m"] == 0)
                                                             or (hi > 0 and lo <= c["dist_outside_demo_box_m"] < hi)))):
            tab[key][0] += 1
            tab[key][1] += a
            tab[key][2] += i
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "breakdown_direction_class.csv", "w") as f:
        f.write("delta_m,direction_class,n_matched,succ_A,succ_I,sr_A,sr_I,gap_I_minus_A\n")
        for (d, cls), (n, a, i) in sorted(by_dir.items()):
            f.write(f"{d},{cls},{n},{a},{i},{a / n:.3f},{i / n:.3f},{(i - a) / n:+.3f}\n")
    with open(out / "breakdown_outside_demo_box.csv", "w") as f:
        f.write("dist_outside_demo_box_m,n_matched,succ_A,succ_I,sr_A,sr_I,gap_I_minus_A\n")
        order = [("inside" if hi == 0 else f"{lo:.2f}-{hi:.2f}") for lo, hi in BINS]
        for b in order:
            if b in by_bin:
                n, a, i = by_bin[b]
                f.write(f"{b},{n},{a},{i},{a / n:.3f},{i / n:.3f},{(i - a) / n:+.3f}\n")
    print((out / "breakdown_direction_class.csv").read_text())
    print((out / "breakdown_outside_demo_box.csv").read_text())
    plot(out)




def plot(out_dir):
    """SR vs distance outside the demonstrated start box (bins), both arms; raw counts in tick labels."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analyze_robustness import COLOR, GRID, INK2
    rows = list(csv.DictReader(open(pathlib.Path(out_dir) / "breakdown_outside_demo_box.csv")))
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(len(rows))
    for k, col, lab, off in (("A", "sr_A", "Baseline flow policy", -0.06), ("I", "sr_I", "Flow-intent policy", 0.06)):
        ax.plot(x + off, [float(r[col]) for r in rows], color=COLOR[k], lw=2, marker="o", ms=5,
                markeredgecolor="white", label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([("inside" if r["dist_outside_demo_box_m"] == "inside" else r["dist_outside_demo_box_m"].replace("1.00", "")
                         ) + f"\nn={r['n_matched']}" for r in rows], fontsize=6.5)
    ax.set_xlabel("Nut start distance outside demonstrated range (m)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 0.6)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(colors=INK2)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(pathlib.Path(out_dir) / f"fig_sr_vs_outside_demo_box.{ext}", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
