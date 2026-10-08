"""Paired analysis of the spatial-shift study: counts, fixed-task macro averages, reset-cluster bootstrap,
figures and tables. Every number it writes is recomputed from the manifest + episode JSONL records.

  python examples/openpi/oops/analyze_robustness.py --manifest-dir oops_runs/manifest_pv1 \
      --episodes-dir oops_runs/episodes --baseline-arm M0 --intent-arm M1 --out-dir docs/oops_workshop/results

Population (spec section 8): predeclared VALID physical cases (manifest status, decided before any policy ran)
on which BOTH arms completed without an infrastructure error, matched by (case_id, policy_seed).
Policy failures (non-finite actions) count as failures and are reported separately. Empty cells are NA.
"""

import argparse
import collections
import hashlib
import json
import pathlib
import sys
import textwrap

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import oops_common as oc  # noqa: E402

COLOR = {"A": "#2a78d6", "I": "#eb6834"}          # validated categorical slots 1-2 (light surface)
VALIDITY_COLOR = {"valid": "#1baf7a", "unknown": "#eda100", "invalid": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_arm(episodes_dir, arm):
    recs = {}
    for p in sorted((pathlib.Path(episodes_dir) / arm).glob("*.jsonl")):
        for r in oc.read_jsonl(p):
            recs[(r["case_id"], r["policy_seed"])] = r  # last write wins (resume never rewrites a completed key)
    return recs


def single_identity(recs, arm):
    ids = collections.Counter((r["checkpoint_hash"], r["evaluator_hash"]) for r in recs.values())
    if len(ids) > 1:
        raise SystemExit(f"arm {arm}: records from {len(ids)} (checkpoint, evaluator) identities {dict(ids)}; "
                         f"these are separate cohorts -- move them to separate --episodes-dir before pooling")
    return next(iter(ids)) if ids else (None, None)


def sr(s, n):
    return float(s) / n if n else float("nan")


def macro(cells, tasks, deltas, key):
    """Equal-weight task mean per delta; NA if any task in the fixed set lacks data at that delta."""
    out = {}
    for d in deltas:
        vals = [cells[(t, d)][key] for t in tasks if (t, d) in cells and cells[(t, d)]["n"] > 0]
        out[d] = float(np.mean(vals)) if len(vals) == len(tasks) and tasks else float("nan")
    return out


def cell_table(rows, deltas):
    """rows: list of matched valid trials {task, delta, reset, a, i}. -> {(task, delta): {...}}"""
    cells = {}
    for r in rows:
        c = cells.setdefault((r["task"], r["delta"]), {"n": 0, "sa": 0, "si": 0, "cases": set(), "resets": set()})
        c["n"] += 1
        c["sa"] += r["a"]
        c["si"] += r["i"]
        c["cases"].add(r["case_id"])
        c["resets"].add(r["reset"])
    for c in cells.values():
        c["sr_a"], c["sr_i"] = sr(c["sa"], c["n"]), sr(c["si"], c["n"])
        c["gap"] = c["sr_i"] - c["sr_a"]
    return cells


def bootstrap(rows, tasks, deltas, n_boot, seed):
    """Paired cluster bootstrap: resample benchmark reset IDs WITHIN each task (keeping every direction,
    policy repeat, delta and both arms of a reset together). Resamples with an empty fixed-set cell are NA
    for that delta and counted -- never filled with zero."""
    rng = np.random.default_rng(seed)
    by_tr = collections.defaultdict(list)
    for r in rows:
        by_tr[(r["task"], r["reset"])].append(r)
    resets = {t: sorted({r["reset"] for r in rows if r["task"] == t}) for t in tasks}
    stats = {k: {d: [] for d in deltas} for k in ("A", "I", "gap", "dA", "dI", "dgap")}
    n_na = {d: 0 for d in deltas}
    for _ in range(n_boot):
        sample = []
        for t in tasks:
            for rid in rng.choice(resets[t], size=len(resets[t]), replace=True):
                sample.extend(by_tr[(t, rid)])
        cells = cell_table(sample, deltas)
        ma, mi = macro(cells, tasks, deltas, "sr_a"), macro(cells, tasks, deltas, "sr_i")
        for d in deltas:
            if np.isnan(ma[d]) or np.isnan(mi[d]):
                n_na[d] += 1
                continue
            stats["A"][d].append(ma[d])
            stats["I"][d].append(mi[d])
            stats["gap"][d].append(mi[d] - ma[d])
            if not (np.isnan(ma[deltas[0]]) or np.isnan(mi[deltas[0]])):
                stats["dA"][d].append(ma[d] - ma[deltas[0]])
                stats["dI"][d].append(mi[d] - mi[deltas[0]])
                stats["dgap"][d].append((mi[d] - mi[deltas[0]]) - (ma[d] - ma[deltas[0]]))
    ci = {k: {d: ([float(np.percentile(v[d], 2.5)), float(np.percentile(v[d], 97.5))] if v[d] else [None, None])
              for d in deltas} for k, v in stats.items()}
    return ci, n_na


def fmt(x, nd=3):
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--baseline-arm", default="M0")
    ap.add_argument("--intent-arm", default="M1")
    ap.add_argument("--baseline-label", default="Baseline (no intent)")
    ap.add_argument("--intent-label", default="Intent-conditioned")
    ap.add_argument("--training-seeds", default="1 (run1) per arm")
    ap.add_argument("--deltas", default="0.0,0.05,0.10", help="displayed deltas (first = reference)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default=None, help="main-figure title")
    ap.add_argument("--provisional", action="store_true", help="stamp outputs PROVISIONAL (evidence not frozen)")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    deltas = [round(float(x), 6) for x in args.deltas.split(",")]
    mdir = pathlib.Path(args.manifest_dir)
    cases = {c["case_id"]: c for c in oc.read_jsonl(mdir / "cases.jsonl")}
    meta = json.loads((mdir / "manifest_meta.json").read_text())
    A, I = load_arm(args.episodes_dir, args.baseline_arm), load_arm(args.episodes_dir, args.intent_arm)
    idA, idI = single_identity(A, args.baseline_arm), single_identity(I, args.intent_arm)
    for recs, arm in ((A, args.baseline_arm), (I, args.intent_arm)):
        stray = [k for k in recs if k[0] not in cases]
        if stray:
            raise SystemExit(f"arm {arm}: {len(stray)} records reference case_ids absent from this manifest")

    seeds = sorted({k[1] for k in list(A) + list(I)}) or [0]
    tasks_all = sorted({c["task_name"] for c in cases.values()})

    # ---- per-trial pairing + coverage ----
    matched, coverage = [], collections.Counter()
    xt = collections.defaultdict(lambda: collections.Counter())
    for cid, c in cases.items():
        d = round(c["delta_m"], 6)
        key_td = (c["task_name"], d)
        xt[key_td]["attempted_cases"] += 1
        xt[key_td][f"{c['validity_status']}_cases"] += 1
        for ps in seeds:
            ra, ri = A.get((cid, ps)), I.get((cid, ps))
            for r, tag in ((ra, "A"), (ri, "I")):
                if r is None:
                    coverage[f"missing_{tag}"] += 1
                    xt[key_td][f"missing_{tag}"] += 1
                elif r.get("infrastructure_error"):
                    coverage[f"infra_{tag}"] += 1
                    xt[key_td][f"infra_errors_{tag}"] += 1
                else:
                    coverage[f"completed_{tag}"] += 1
                    if r.get("policy_failure"):
                        xt[key_td][f"policy_failures_{tag}"] += 1
            okA = ra is not None and not ra.get("infrastructure_error")
            okI = ri is not None and not ri.get("infrastructure_error")
            if okA and okI:
                coverage["matched_pairs"] += 1
                st = c["validity_status"]
                xt[key_td][f"matched_{st}_trials"] += 1
                xt[key_td][f"{st}_successes_A"] += int(bool(ra["success"]))
                xt[key_td][f"{st}_successes_I"] += int(bool(ri["success"]))
                if st == "valid":
                    matched.append({"task": c["task_name"], "delta": d, "reset": c["reset_id"], "case_id": cid,
                                    "angle_idx": c["angle_idx"], "seed": ps,
                                    "a": int(bool(ra["success"])), "i": int(bool(ri["success"]))})
            elif okA or okI:
                coverage["unpaired_excluded"] += 1
                xt[key_td]["unpaired_excluded"] += 1

    # ---- cross-tab (spec section 6) ----
    xcols = ["attempted_cases", "valid_cases", "invalid_cases", "unknown_cases",
             "matched_valid_trials", "matched_invalid_trials", "matched_unknown_trials",
             "valid_successes_A", "valid_successes_I", "invalid_successes_A", "invalid_successes_I",
             "unknown_successes_A", "unknown_successes_I", "infra_errors_A", "infra_errors_I",
             "policy_failures_A", "policy_failures_I", "missing_A", "missing_I", "unpaired_excluded"]
    with open(out / "crosstab_validity.csv", "w") as f:
        f.write("task_name,delta_m," + ",".join(xcols) + "\n")
        for (t, d) in sorted(xt):
            f.write(f"{t},{d}," + ",".join(str(xt[(t, d)][k]) for k in xcols) + "\n")

    # invalid-reason breakdown (pre-policy)
    reasons = collections.Counter()
    for c in cases.values():
        if c["validity_status"] != "valid":
            for rr in c["validity_reason"].split(";"):
                reasons[(round(c["delta_m"], 6), c["validity_status"], rr)] += 1
    with open(out / "validity_reasons.csv", "w") as f:
        f.write("delta_m,status,reason,n_cases\n")
        for (d, s, rr), n in sorted(reasons.items()):
            f.write(f"{d},{s},{rr},{n}\n")

    # ---- per-task cells on matched valid trials ----
    cells = cell_table(matched, deltas)
    with open(out / "per_task_cells.csv", "w") as f:
        f.write("task_name,delta_m,n_valid_cases,n_matched_trials,n_reset_families,"
                "succ_A,succ_I,sr_A,sr_I,gap_I_minus_A\n")
        for t in tasks_all:
            for d in deltas:
                c = cells.get((t, d))
                if c is None:
                    f.write(f"{t},{d},0,0,0,NA,NA,NA,NA,NA\n")
                else:
                    f.write(f"{t},{d},{len(c['cases'])},{c['n']},{len(c['resets'])},{c['sa']},{c['si']},"
                            f"{fmt(c['sr_a'])},{fmt(c['sr_i'])},{fmt(c['gap'])}\n")

    # discordant pairs (descriptive; not a hypothesis test)
    disc = collections.Counter()
    for r in matched:
        disc[(r["delta"], r["a"], r["i"])] += 1

    # ---- fixed task set: valid matched data at EVERY displayed delta ----
    fixed = [t for t in tasks_all if all((t, d) in cells and cells[(t, d)]["n"] > 0 for d in deltas)]
    dropped = [t for t in tasks_all if t not in fixed]
    mA, mI = macro(cells, fixed, deltas, "sr_a"), macro(cells, fixed, deltas, "sr_i")
    micro = {}
    for d in deltas:
        rs = [r for r in matched if r["delta"] == d and r["task"] in fixed]
        micro[d] = (sr(sum(r["a"] for r in rs), len(rs)), sr(sum(r["i"] for r in rs), len(rs)), len(rs))
    ci, n_na = bootstrap([r for r in matched if r["task"] in fixed], fixed, deltas, args.n_boot, args.seed) \
        if fixed else ({}, {})
    min_resets = min((len({r["reset"] for r in matched if r["task"] == t}) for t in fixed), default=0)
    exploratory = min_resets < 5

    n_valid_cases = {d: sum(len(cells[(t, d)]["cases"]) for t in fixed if (t, d) in cells) for d in deltas}
    n_trials = {d: sum(cells[(t, d)]["n"] for t in fixed if (t, d) in cells) for d in deltas}
    sA = {d: sum(cells[(t, d)]["sa"] for t in fixed if (t, d) in cells) for d in deltas}
    sI = {d: sum(cells[(t, d)]["si"] for t in fixed if (t, d) in cells) for d in deltas}

    summary = {
        "provisional": args.provisional, "experiment": oc.EXPERIMENT, "config_version": oc.CONFIG_VERSION,
        "validity_version": meta["validity_rules"]["version"], "physics_stack": meta.get("physics_stack"),
        "baseline_arm": args.baseline_arm, "intent_arm": args.intent_arm,
        "baseline_identity": {"checkpoint_hash": idA[0], "evaluator_hash": idA[1]},
        "intent_identity": {"checkpoint_hash": idI[0], "evaluator_hash": idI[1]},
        "policy_seeds": seeds, "deltas": deltas, "fixed_task_set": fixed, "tasks_dropped_from_macro": dropped,
        "coverage": dict(coverage), "n_valid_cases_fixed_set": n_valid_cases, "n_matched_trials_fixed_set": n_trials,
        "successes_A": sA, "successes_I": sI,
        "macro_sr_A": mA, "macro_sr_I": mI, "macro_gap": {d: mI[d] - mA[d] for d in deltas},
        "macro_change_from_ref_A": {d: mA[d] - mA[deltas[0]] for d in deltas},
        "macro_change_from_ref_I": {d: mI[d] - mI[deltas[0]] for d in deltas},
        "micro_secondary_trial_weighted": {d: {"sr_A": micro[d][0], "sr_I": micro[d][1], "n": micro[d][2]} for d in deltas},
        "bootstrap": {"unit": "benchmark reset id within task", "n_boot": args.n_boot, "ci95": ci,
                      "n_resamples_na": n_na, "min_reset_families_per_task": min_resets,
                      "exploratory": exploratory,
                      "note": "conditional on the evaluated task set and checkpoints; not training-seed variability"},
        "discordant_pairs": {str(d): {"A_only": disc[(d, 1, 0)], "I_only": disc[(d, 0, 1)],
                                      "both": disc[(d, 1, 1)], "neither": disc[(d, 0, 0)]} for d in deltas},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=oc._json_default))

    # ---- compact table (CSV + LaTeX) ----
    tag = " [PROVISIONAL]" if args.provisional else ""
    with open(out / "table_performance.csv", "w") as f:
        f.write("model,arm,training_seeds," + ",".join(f"SR_d{d:g},succ_d{d:g},n_d{d:g}" for d in deltas) + "\n")
        for lab, arm, m, s in ((args.baseline_label, args.baseline_arm, mA, sA), (args.intent_label, args.intent_arm, mI, sI)):
            f.write(f"{lab},{arm},{args.training_seeds}," +
                    ",".join(f"{fmt(m[d])},{s[d]},{n_trials[d]}" for d in deltas) + "\n")
    with open(out / "table_performance.tex", "w") as f:
        f.write("% generated by analyze_robustness.py" + tag + "\n")
        f.write("\\begin{tabular}{@{}l c " + "c" * len(deltas) + "@{}}\n\\toprule\n")
        f.write("Model & Train seeds & " + " & ".join(f"SR($\\delta{{=}}{d:g}$\\,m)" for d in deltas) + " \\\\\n\\midrule\n")
        for lab, m, s in ((args.baseline_label, mA, sA), (args.intent_label, mI, sI)):
            f.write(f"{lab} & 1 & " + " & ".join(
                f"{fmt(m[d], 2)} \\scriptsize({s[d]}/{n_trials[d]})" for d in deltas) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"% macro-average over fixed task set ({len(fixed)} tasks): {', '.join(fixed)}\n")
        f.write(f"% parentheses: pooled successes / matched valid trials (raw counts; SR is the task macro-average)\n")

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})

    def save(fig, name):
        for ext in ("pdf", "svg", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # (1) main paired curve
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.array(deltas)
    for k, lab, m in (("A", args.baseline_label, mA), ("I", args.intent_label, mI)):
        y = np.array([m[d] for d in deltas])
        lo = np.array([ci.get(k, {}).get(d, [None, None])[0] if ci else None for d in deltas], dtype=float)
        hi = np.array([ci.get(k, {}).get(d, [None, None])[1] if ci else None for d in deltas], dtype=float)
        off = -0.0015 if k == "A" else 0.0015
        ax.errorbar(x + off, y, yerr=[np.nan_to_num(y - lo), np.nan_to_num(hi - y)], color=COLOR[k], lw=2,
                    marker="o", ms=5, capsize=2.5, elinewidth=1, label=lab,
                    markeredgecolor="white", markeredgewidth=1)
    ax.set_xlabel("XY shift of manipulated object $\\delta$ (m)")
    ax.set_ylabel("Success rate" + (" (task macro-avg.)" if len(fixed) > 1 else ""))
    ax.set_ylim(0, 1.02)
    ax.set_xticks(deltas)
    ax.set_xticklabels([f"{d:g}\nn={n_trials[d]}" for d in deltas], fontsize=6.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title(args.title or (f"{len(fixed)} task{'s' if len(fixed) != 1 else ''}, valid matched cases{tag}"),
                 fontsize=7.5, color=INK2)
    save(fig, "fig_main_sr_vs_delta")

    # (2) per-task small multiples
    if tasks_all:
        nc = 4
        nr = int(np.ceil(len(tasks_all) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(7.0, 1.9 * nr), sharex=True, sharey=True, squeeze=False)
        for ax, t in zip(axes.flat, tasks_all):
            for k, key in (("A", "sr_a"), ("I", "sr_i")):
                ys = [cells[(t, d)][key] if (t, d) in cells else np.nan for d in deltas]
                ax.plot(deltas, ys, color=COLOR[k], lw=2, marker="o", ms=4, markeredgecolor="white")
            ns = [cells[(t, d)]["n"] if (t, d) in cells else 0 for d in deltas]
            ax.set_title(textwrap.fill(t.replace("_", " "), 28), fontsize=6.5, color=INK)
            ax.text(0.02, 0.04, "n=" + "/".join(map(str, ns)), transform=ax.transAxes, fontsize=5.5, color=INK2)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(deltas)
            ax.grid(axis="y", color=GRID, lw=0.5)
        for ax in list(axes.flat)[len(tasks_all):]:
            ax.axis("off")
        handles = [plt.Line2D([], [], color=COLOR["A"], lw=2, marker="o", label=args.baseline_label),
                   plt.Line2D([], [], color=COLOR["I"], lw=2, marker="o", label=args.intent_label)]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.03))
        fig.subplots_adjust(hspace=0.6)
        fig.supxlabel("$\\delta$ (m)", fontsize=7)
        fig.supylabel("Success rate (valid matched)", fontsize=7)
        save(fig, "fig_per_task")

    # (3) validity breakdown by delta (pre-policy)
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    alld = sorted({round(c["delta_m"], 6) for c in cases.values()})
    bottom = np.zeros(len(alld))
    for st in ("valid", "unknown", "invalid"):
        vals = np.array([sum(1 for c in cases.values() if round(c["delta_m"], 6) == d and c["validity_status"] == st)
                         for d in alld], dtype=float)
        bars = ax.bar([str(d) for d in alld], vals, bottom=bottom, color=VALIDITY_COLOR[st], width=0.6,
                      edgecolor="white", linewidth=1, label=st)
        for b, v, b0 in zip(bars, vals, bottom):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, b0 + v / 2, f"{int(v)}", ha="center", va="center",
                        fontsize=6, color="white" if st != "unknown" else INK)
        bottom += vals
    ax.set_xlabel("$\\delta$ (m)")
    ax.set_ylabel("Physical cases")
    ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", color=GRID, lw=0.5)
    save(fig, "fig_validity_by_delta")

    # ---- provenance of this analysis ----
    def h(p):
        return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]
    inputs = {"manifest_cases": h(mdir / "cases.jsonl"), "manifest_meta": h(mdir / "manifest_meta.json")}
    for arm in (args.baseline_arm, args.intent_arm):
        for p in sorted((pathlib.Path(args.episodes_dir) / arm).glob("*.jsonl")):
            inputs[f"{arm}/{p.name}"] = h(p)
    inputs["analyze_robustness.py"] = h(__file__)
    (out / "analysis_inputs.json").write_text(json.dumps(inputs, indent=2))

    # ---- console ----
    print(f"fixed task set ({len(fixed)}): {fixed}" + (f"   dropped: {dropped}" if dropped else ""))
    print(f"coverage: {dict(coverage)}")
    print(f"{'delta':>6} {'n':>5} {'SR_A':>6} {'SR_I':>6} {'gap':>7}  CI_gap  {'(exploratory)' if exploratory else ''}")
    for d in deltas:
        g = ci.get("gap", {}).get(d, [None, None]) if ci else [None, None]
        print(f"{d:6.2f} {n_trials[d]:5d} {fmt(mA[d]):>6} {fmt(mI[d]):>6} {fmt(mI[d] - mA[d]):>7}  "
              f"[{fmt(g[0])}, {fmt(g[1])}]")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
