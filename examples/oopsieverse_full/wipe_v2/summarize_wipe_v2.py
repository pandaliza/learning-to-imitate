"""Summaries for wipe v2 evaluation records (eval_wipe_v2.py out dirs).

  python summarize_wipe_v2.py --run-dir RUN [--run-dir ...] [--ckpt step_30000.pth] [--json OUT.json]
      per (run, pool) table + the spec's pass criterion against the W0 reference: the POOLED W0 s0-s2 P-train task
      success (67/360 = 18.6%; equal n per seed, so = the mean of the three seed rates). Reference decision recorded in
      wipe_v2_spec.md 2026-10-26 00:45 UTC, before any v2 arm had records.
  python summarize_wipe_v2.py --calibration     -> reports/wipe_v2/calibration_w0.md (+ .json)

Metrics (spec "Metrics reported"): task success (Wilson 95%), primary / secondary / compatibility safe success,
mean final proportion wiped (+ distribution), mean markers wiped, damaging outcomes (primary predicate violated).
Only completed records count; infra errors are reported separately. The latest eval key per (run, pool, ckpt) is used
and the key is printed.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wv2_common as C  # noqa: E402

EVAL_ROOT = os.path.join(C.V2_RUNS, "eval")
POOLS = ["P-train", "P-devlayout", "P-random"]
MATRIX = os.path.join(C.OVS, "runs/matrix")
W0 = [f"{MATRIX}/amix_wipe_countertop_v1_s{s}" for s in (0, 1, 2)]
OTHER = [f"{MATRIX}/{a}_wipe_countertop_v1_s0" for a in ("asafe", "imean", "ipath", "ipathg", "auxpathg")]
EXPECTED = {"P-train": 120, "P-devlayout": 80, "P-random": 100}  # resets x 2 rollout seeds


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def out_dir(run_dir, pool, ckpt="step_30000.pth"):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    rid = meta.get("run_id") or os.path.basename(run_dir.rstrip("/"))
    ds = sorted(glob.glob(os.path.join(EVAL_ROOT, rid, f"{pool}__{os.path.splitext(ckpt)[0]}__*")), key=os.path.getmtime)
    return ds[-1] if ds else None


def load(d):
    recs, infra = [], 0
    for f in sorted(glob.glob(os.path.join(d, "*__r*.json"))):
        r = json.load(open(f))
        if "infra_error" in r:
            infra += 1
        else:
            recs.append(r)
    return recs, infra


def stats(recs):
    n = len(recs)
    k = sum(r["success"] for r in recs)
    lo, hi = wilson(k, n)
    pw = np.asarray([r["final_proportion_wiped"] for r in recs]) if n else np.zeros(0)
    mw = np.asarray([r["final_markers_wiped"] for r in recs]) if n else np.zeros(0)
    bins = {"0": int((mw == 0).sum()), "1-24": int(((mw >= 1) & (mw <= 24)).sum()), "25-44": int(((mw >= 25) & (mw <= 44)).sum()),
            "45-49": int(((mw >= 45) & (mw <= 49)).sum()), "50": int((mw == 50).sum())}
    out = {"n": n, "success": k, "success_rate": k / n if n else float("nan"), "wilson95": [lo, hi],
           "primary_safe_success": sum(r["primary_safe_success"] for r in recs),
           "secondary_safe_success": sum(r["secondary_safe_success"] for r in recs),
           "compat_safe_success": sum(r["compat_safe_success"] for r in recs),
           "damaging": sum(r["damaging"] for r in recs),
           "mean_prop_wiped": float(pw.mean()) if n else float("nan"),
           "prop_wiped_quantiles": {q: float(np.quantile(pw, q / 100)) for q in (10, 25, 50, 75, 90)} if n else {},
           "mean_markers_wiped": float(mw.mean()) if n else float("nan"),
           "markers_wiped_bins": bins,
           "mean_steps": float(np.mean([r["steps"] for r in recs])) if n else float("nan"),
           "reset_state_reproduced": sum(bool(r["reset_check"]["state_equal_ref"]) for r in recs),
           "counter_visible_eye_in_hand_at_reset": sum(r["reset_check"]["counter_px_at_reset"].get("robot0_eye_in_hand/alias", 0) > 0 for r in recs)}
    return out


def by_group(recs):
    g = {}
    for r in recs:
        g.setdefault(r.get("layout_group") or "-", []).append(r)
    return {k: stats(v) for k, v in sorted(g.items())}


def summarize_run(run_dir, ckpt="step_30000.pth", pools=POOLS):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    res = {"run_dir": run_dir, "run_id": meta.get("run_id"), "arm": meta["arm"], "seed": meta.get("seed"), "pools": {}}
    for p in pools:
        d = out_dir(run_dir, p, ckpt)
        if d is None:
            continue
        recs, infra = load(d)
        res["pools"][p] = dict(stats(recs), infra_errors=infra, out_dir=d, eval_key=d.rsplit("__", 1)[-1],
                               expected=EXPECTED[p], groups=by_group(recs) if p != "P-random" else None)
    return res


REF_NOTE = ("**Reference decision (recorded in wipe_v2_spec.md, 2026-10-26 00:45 UTC, before any v2 arm had records):** the P-train "
            "reference for the \"not below W0\" condition is the mean over the three W0 seeds = the pooled W0 s0–s2 P-train task success "
            "({k}/{n} = {rate})  [per seed: {per}].")


def w0_train_reference():
    """Pooled W0 s0-s2 P-train task success over complete runs (k, n, rate, per-seed rates)."""
    k = n = 0
    per = []
    for rd in W0:
        st = summarize_run(rd, pools=["P-train"])["pools"]["P-train"]
        assert st["n"] == EXPECTED["P-train"], f"{rd}: W0 P-train incomplete ({st['n']})"
        k += st["success"]
        n += st["n"]
        per.append(st["success_rate"])
    return k, n, k / n, per


def pass_criterion(s, w0_train_rate):
    dv = s["pools"].get("P-devlayout")
    tr = s["pools"].get("P-train")
    if not dv or not tr:
        return {"evaluable": False}
    c1 = dv["success_rate"] >= 0.30
    c2 = dv["wilson95"][0] > 0.10
    c3 = tr["success_rate"] >= w0_train_rate
    complete = dv["n"] == EXPECTED["P-devlayout"] and tr["n"] == EXPECTED["P-train"]
    return {"evaluable": True, "complete": complete, "dev_rate": dv["success_rate"], "dev_wilson_lo": dv["wilson95"][0],
            "dev_rate_ge_30pct": c1, "dev_wilson_lo_gt_10pct": c2, "train_rate": tr["success_rate"],
            "w0_train_rate_ref": w0_train_rate, "train_not_below_w0": c3, "PASS": bool(c1 and c2 and c3 and complete)}


def pct(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def row(name, st):
    lo, hi = st["wilson95"]
    q = st["prop_wiped_quantiles"]
    return (f"| {name} | {st['n']} | {st['success']} ({pct(st['success_rate'])}) | {pct(lo)}–{pct(hi)} | "
            f"{st['primary_safe_success']} / {st['secondary_safe_success']} / {st['compat_safe_success']} | "
            f"{st['mean_prop_wiped']:.3f} | {q.get(25, float('nan')):.2f} / {q.get(50, float('nan')):.2f} / {q.get(75, float('nan')):.2f} | "
            f"{st['mean_markers_wiped']:.1f} | {st['markers_wiped_bins']['0']}/{st['markers_wiped_bins']['1-24']}/"
            f"{st['markers_wiped_bins']['25-44']}/{st['markers_wiped_bins']['45-49']}/{st['markers_wiped_bins']['50']} | "
            f"{st['damaging']} | {st['mean_steps']:.0f} |")


HDR = ("| Run | n | Task success | Wilson 95% | Safe success P / S / C | Mean prop. wiped | Prop. wiped q25 / q50 / q75 | "
       "Mean markers | Markers 0 / 1–24 / 25–44 / 45–49 / 50 | Damaging | Mean steps |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|")


def pooled(stats_list_recs):
    return stats(stats_list_recs)


def calibration():
    pools_doc = C.load_pools()
    out = {"W0": [summarize_run(r) for r in W0], "other_s0": [summarize_run(r, pools=["P-train", "P-devlayout"]) for r in OTHER]}
    rk, rn, w0_ref, w0_per = w0_train_reference()
    out["w0_train_reference"] = {"k": rk, "n": rn, "rate": w0_ref, "per_seed": w0_per, "definition": "pooled W0 s0-s2 P-train"}
    for s in out["W0"] + out["other_s0"]:
        s["pass_criterion_vs_W0pooled"] = pass_criterion(s, w0_ref)
    allrec = {p: sum((load(s["pools"][p]["out_dir"])[0] for s in out["W0"] if p in s["pools"]), []) for p in POOLS}
    out["W0_pooled_seeds"] = {p: stats(v) for p, v in allrec.items()}
    L = []
    L.append("# wipe v2 calibration: W0 (official A-MIX) on the frozen screening pools\n")
    L.append(f"Generated by `examples/oopsieverse_full/wipe_v2/summarize_wipe_v2.py --calibration`. Pools: "
             f"`runs/wipe_v2/pools/pools_v1.json`, content_sha256 `{pools_doc['content_sha256']}`. Evaluator: "
             f"`wipe_v2/eval_wipe_v2.py` (frozen `run_episode`, counter-label alias). Checkpoint `step_30000.pth`; rollout seeds 0 and 1 per reset; "
             f"timeout 1250. Safe success P / S / C = primary / secondary / compatibility predicates. Damaging = primary predicate violated "
             f"(any entity ≤ 95 health). Markers bins count episodes by final markers wiped (of 50).\n")
    L.append("## Frozen pools\n")
    L.append(f"`pools_v1.json` file sha256 `{C.sha256_file(C.POOLS_FILE)}`, content_sha256 `{pools_doc['content_sha256']}` "
             f"(frozen {pools_doc['created']}, before any v2 policy evaluation). Checks: " +
             ", ".join(f"{k} = {v}" for k, v in pools_doc["checks"].items()) + ".\n")
    L.append("| Pool | Group | Source session | Dirt hash | Base xy | Resets | Sponge xy span (cm) | Min pairwise sponge dist (mm) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for pool in ("P-train", "P-devlayout"):
        for g in pools_doc["pools"][pool]["groups"]:
            Lg, pg = pools_doc["layouts"][g], pools_doc["per_group"][g]
            L.append(f"| {pool} | {g} | {Lg['source_demo']} ({len(Lg['members'])} demos, identical ep_meta) | `{Lg['dirt_hash12']}` | "
                     f"{Lg['init_robot_base_pos'][0]:.4f}, {Lg['init_robot_base_pos'][1]:.4f} | {pg['n']} | {pg['sponge_xy_span_cm']} | "
                     f"{pg['sponge_xy_pairwise_min_mm']:.2f} |")
    L.append(f"| P-random | – | eval_dev[0:50] of reset_banks_v1 (content_sha256 `{pools_doc['P-random']['bank_content_sha256'][:16]}…`) | per reset | random | 50 | – | – |\n")
    va = os.path.join(C.V2_RUNS, "verify/verify_alias.json")
    if os.path.exists(va):
        v = json.load(open(va))
        A, B = v["A_training_frames"], v["B_eval_resets"]
        pre = [x for x in A if x["training_total_wiped_at_t"] == 0 and x["path"] == "step"]
        L.append("## Counter-label alias verification (`verify_alias.py`)\n")
        L.append(f"- Alias LUT identical to wipe_diag's `AliasPolicy.seg_lut` on all {v['summary']['C_n']} class maps met: {v['summary']['C_all_identical']}.")
        L.append(f"- Pool resets reloaded through the playback path (training naming): min per-view agreement, alias vs playback = "
                 f"{v['summary']['B_min_alias_vs_playback']}; frozen evaluator LUT vs playback, eye-in-hand = "
                 f"{v['summary']['B_min_frozen_vs_playback_eye_in_hand']} (counter 0 px with the frozen LUT on every pool, including sticky-ep_meta resets).")
        L.append(f"- Training frames (PlaybackDataset sample of the training file, step path, before any marker is wiped): "
                 f"min agreement {min(min(x['seg_equal_frac_alias_vs_training'].values()) for x in pre):.6f} over {len(pre)} frames x 3 views. "
                 f"After markers are wiped (t = 150) agreement drops to {min(min(x['seg_equal_frac_alias_vs_training'].values()) for x in A):.3f} "
                 f"because a state restore does not restore wiped-marker visibility (mismatches are training-counter -> eval-dirt pixels); "
                 f"excluding dirt pixels the minimum is {min(min(x['seg_equal_frac_excluding_dirt_pixels'].values()) for x in A):.4f}. "
                 "This is a verification artifact, not an evaluator difference: in a rollout markers are wiped by the env itself.\n")
    L.append("## W0 per pool\n")
    for p in POOLS:
        L.append(f"### {p} ({EXPECTED[p]} episodes per run)\n")
        L.append(HDR)
        for s in out["W0"]:
            if p in s["pools"]:
                L.append(row(f"W0 s{s['seed']}", s["pools"][p]))
        L.append(row("W0 s0–s2 pooled", out["W0_pooled_seeds"][p]))
        L.append("")
        if p != "P-random":
            L.append("Per layout group (task success k/n, mean markers wiped):\n")
            gs = sorted({g for s in out["W0"] if p in s["pools"] for g in s["pools"][p]["groups"]})
            L.append("| Group | " + " | ".join(f"W0 s{s['seed']}" for s in out["W0"]) + " |")
            L.append("|---|" + "---|" * len(out["W0"]))
            for g in gs:
                cells = []
                for s in out["W0"]:
                    x = s["pools"].get(p, {}).get("groups", {}).get(g)
                    cells.append(f"{x['success']}/{x['n']}, {x['mean_markers_wiped']:.1f}" if x else "–")
                L.append(f"| {g} | " + " | ".join(cells) + " |")
            L.append("")
    L.append("## Other primary wipe arms, training seed 0 (context)\n")
    for p in ("P-train", "P-devlayout"):
        L.append(f"### {p}\n")
        L.append(HDR)
        L.append(row("W0 (amix) s0", out["W0"][0]["pools"][p]))
        for s in out["other_s0"]:
            if p in s["pools"]:
                L.append(row(f"{s['arm']} s0", s["pools"][p]))
        L.append("")
    L.append("## Pass criterion (spec, declared 2026-10-25)\n")
    L.append("An arm passes if on P-devlayout (80 episodes) task success ≥ 30% **and** its Wilson 95% lower bound > 10%, **and** its "
             "P-train task success is not below W0's.\n")
    L.append(REF_NOTE.format(k=rk, n=rn, rate=pct(w0_ref), per=", ".join(f"s{i} {pct(x)}" for i, x in enumerate(w0_per))) + "\n")
    L.append("| Run | P-devlayout success | Wilson lower | ≥ 30% | lower > 10% | P-train success | ≥ W0 pooled | complete | PASS |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in out["W0"] + out["other_s0"]:
        c = s["pass_criterion_vs_W0pooled"]
        if not c.get("evaluable"):
            continue
        L.append(f"| {s['arm']} s{s['seed']} | {pct(c['dev_rate'])} | {pct(c['dev_wilson_lo'])} | {c['dev_rate_ge_30pct']} | "
                 f"{c['dev_wilson_lo_gt_10pct']} | {pct(c['train_rate'])} | {c['train_not_below_w0']} | {c['complete']} | **{c['PASS']}** |")
    L.append("")
    L.append("## Integrity\n")
    L.append("| Run | Pool | Records / expected | Infra errors | Post-reset state = frozen hash | Counter visible at reset (eye-in-hand, alias) | Eval key |")
    L.append("|---|---|---|---|---|---|---|")
    for s in out["W0"] + out["other_s0"]:
        for p, st in s["pools"].items():
            L.append(f"| {s['run_id']} | {p} | {st['n']} / {st['expected']} | {st['infra_errors']} | {st['reset_state_reproduced']}/{st['n']} | "
                     f"{st['counter_visible_eye_in_hand_at_reset']}/{st['n']} | `{st['eval_key']}` |")
    md = "\n".join(L) + "\n"
    os.makedirs(C.V2_REPORTS, exist_ok=True)
    open(os.path.join(C.V2_REPORTS, "calibration_w0.md"), "w").write(md)
    json.dump(out, open(os.path.join(C.V2_REPORTS, "calibration_w0.json"), "w"), indent=1, default=float)
    print(md)


ABL_ARMS = [("I-PATH s0", f"{MATRIX}/ipath_wipe_countertop_v1_s0"),
            ("W3 s0", f"{C.V2_RUNS}/train/w3_wipe_countertop_v2_s0")]


def ablation():
    """reports/wipe_v2/ablation_intent.md: P-devlayout, sampled (existing main-run records) vs shuffled vs zero."""
    out, L = {}, []
    L.append("# wipe v2 secondary analysis: rollout-time intent ablation (P-devlayout)\n")
    L.append("Spec section \"Secondary analysis: rollout-time intent ablation\" (declared 2026-10-26 00:55 UTC). P-devlayout (40 resets x "
             "rollout seeds 0, 1 = 80 episodes per row), checkpoint `step_30000.pth`, same evaluator and episode protocol as the main runs "
             "(`eval_wipe_v2_ablate.py`, counter-label alias). **sampled** = the existing main-run P-devlayout records (no new episodes); "
             "**shuffled** = action flow conditioned on another reset's previous-chunk sampled intent; **zero** = the normalised intent "
             "token set to its training mean (MinMax[-1,1] targets, so not the zero vector).\n")
    L.append("**The comparison is distributional, not paired.** Per-chunk noise draws (x0, z) are identical across modes, but the "
             "inference server batches whatever requests are queued and GPU float round-off depends on batch size (measured: identical "
             "batch -> 0 difference; same inputs in another batch size -> up to 6e-5), so even `sampled` re-runs diverge from the stored "
             "records over a 1250-step episode. Compare rates over the 80 episodes, not episode by episode. This analysis does not pass or "
             "fail an arm.\n")
    L.append("| Arm | Mode | n | Task success | Wilson 95% | Primary safe success | Mean final prop. wiped | Mean markers wiped | "
             "Markers 0 / 1–24 / 25–44 / 45–49 / 50 | Damaging | Infra | Intent provenance (chunks) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, rd in ABL_ARMS:
        meta = json.load(open(os.path.join(rd, "meta.json")))
        rid = meta["run_id"]
        rows = {"sampled": out_dir(rd, "P-devlayout")}
        for m in ("shuffled", "zero"):
            ds = sorted(glob.glob(os.path.join(C.V2_RUNS, "ablation", rid, f"P-devlayout__step_30000__{m}__*")), key=os.path.getmtime)
            rows[m] = ds[-1] if ds else None
        out[name] = {}
        for m, d in rows.items():
            if d is None:
                L.append(f"| {name} | {m} | – | not run | | | | | | | | |")
                continue
            recs, infra = load(d)
            st = stats(recs)
            prov = {}
            for r in recs:
                for k, v in (r.get("intent_ablation_provenance") or {}).items():
                    prov[k] = prov.get(k, 0) + v
            if m != "sampled":
                assert all(r.get("intent_ablation") == m for r in recs), (d, m)
            lo, hi = st["wilson95"]
            b = st["markers_wiped_bins"]
            ptxt = ", ".join(f"{k} {v}" for k, v in sorted(prov.items())) if prov else "– (sampled)"
            L.append(f"| {name} | {m} | {st['n']} | {st['success']} ({pct(st['success_rate'])}) | {pct(lo)}–{pct(hi)} | "
                     f"{st['primary_safe_success']} ({pct(st['primary_safe_success'] / st['n'] if st['n'] else float('nan'))}) | "
                     f"{st['mean_prop_wiped']:.3f} | {st['mean_markers_wiped']:.1f} | {b['0']}/{b['1-24']}/{b['25-44']}/{b['45-49']}/{b['50']} | "
                     f"{st['damaging']} | {infra} | {ptxt} |")
            out[name][m] = dict(st, infra_errors=infra, out_dir=d, provenance=prov)
    L.append("")
    L.append("Provenance: `batch_prev_chunk` = previous-chunk sampled intent of another worker's (different) reset in the same server "
             "batch; `fallback_ring` = most recent previous-chunk intent of another worker on a different reset (batch had none); "
             "`fallback_mean` = training-mean token (no other intent available yet); `train_mean` = zero mode.\n")
    L.append("Record directories:\n")
    for name, v in out.items():
        for m, st in v.items():
            L.append(f"- {name} {m}: `{st['out_dir']}`")
    md = "\n".join(L) + "\n"
    open(os.path.join(C.V2_REPORTS, "ablation_intent.md"), "w").write(md)
    json.dump(out, open(os.path.join(C.V2_REPORTS, "ablation_intent.json"), "w"), indent=1, default=float)
    print(md)


DEMO_RUNS = [f"{MATRIX}/amix_wipe_countertop_v1_s{s}" for s in (0, 1, 2)] + [f"{MATRIX}/ipath_wipe_countertop_v1_s0"]
REPLAY = os.path.join(C.OVS, "runs/replay/open_teleop/wipe_countertop__{cond}__{demo}.npz")


def replay_reference(entries):
    """Open-loop replay of the recorded actions (actions[1:] from states[0], no forcing; runs/replay/open_teleop, render-free
    env; wipe_diag h4 showed identical wiped traces in the evaluator env for 6/6 demos). Success = predicate at the last
    replayed step; primary safe = every entity's final health (min over its links) > 95."""
    rows = []
    for e in entries:
        z = np.load(REPLAY.format(cond=e["cond"], demo=e["demo"]), allow_pickle=True)
        names = [str(x) for x in z["link_names"]]
        H = z["rp_health"][-1]
        ent = {}
        for v, n in zip(H, names):
            o = n.split("@", 1)[0]
            ent[o] = min(ent.get(o, 100.0), float(v))
        succ = bool(z["rp_success"][-1])
        safe = all(v > 95.0 for v in ent.values())
        rows.append({"layout_group": e["layout_group"], "success": succ, "primary_safe_success": succ and safe,
                     "final_proportion_wiped": float(z["rp_wiped"][-1]) / 50.0, "steps": int(len(z["rp_wiped"]))})
    return rows


def _rate_cells(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({pct(k / n if n else float('nan'))}; {pct(lo)}–{pct(hi)})"


def demo_starts():
    import build_pool_demo as BPD
    pd = json.load(open(BPD.POOL_DEMO_FILE))
    groups = ["g01", "g02", "g06"]
    L = ["# wipe v2 diagnostic: P-demo (exact training-demonstration starts)\n"]
    L.append(f"Spec addendum (2026-10-26). Pool `runs/wipe_v2/pools/pools_demo_v1.json`, file sha256 `{C.sha256_file(BPD.POOL_DEMO_FILE)}`, "
             f"content_sha256 `{pd['content_sha256']}`: the full initial state (ep_meta + model_file + states[0]) of the 40 splits_v1 TRAIN demos "
             "(layout groups g01 = fb/demo_10–14, g02 = fb/demo_15–19, g06 = nofb/demo_0–29), restored as wipe_diag/openloop_replay.py does "
             "(rollout_eval.apply_reset) plus an observable refresh, then normal physics with policy actions. 3 rollout seeds -> 120 episodes "
             "per run; checkpoint step_30000; evaluator `eval_wipe_v2_demo.py` (eval_wipe_v2 machinery, counter-label alias). Diagnostic, no pass criterion.\n")
    ck = pd["checks"]
    md = ck["max_deviation_vs_recorded_obs0"]
    L.append("**Restore verification** (max over the 40 demos, restored observation vs the demo's recorded first observation obs[0]): " +
             ", ".join(f"{k} {v:.2g}" for k, v in md.items()) +
             f". Second shuffled restore pass reproduced the state {ck['second_pass_state_reproduced']}; counter visible with the alias "
             f"{ck['counter_visible_alias_eye_in_hand']} (alias LUT = frozen LUT on {ck['alias_lut_equals_frozen_lut']} restores, because "
             "the model_file reload names the fixture DamageableCounter). Without the refresh the first observation is stale "
             f"(EEF off by up to {md.get('stale_without_refresh_eef_pos', float('nan')):.2f} m): robosuite returns cached observables after "
             "set_state (roster alignment_audit.md, same defect as official playback obs[0]).\n")
    L.append("## Overall and per layout group\n")
    L.append("Cells: task success k/n (rate; Wilson 95%) · primary safe success k · mean final proportion wiped.\n")
    L.append("| Run | Overall (n=120) | g01 (n=15) | g02 (n=15) | g06 (n=90) |")
    L.append("|---|---|---|---|---|")
    out = {"runs": {}}

    def cells(recs):
        k = sum(r["success"] for r in recs)
        ps = sum(r["primary_safe_success"] for r in recs)
        pw = float(np.mean([r["final_proportion_wiped"] for r in recs])) if recs else float("nan")
        return f"{_rate_cells(k, len(recs))} · {ps} · {pw:.3f}", {"n": len(recs), "success": k, "primary_safe_success": ps, "mean_prop_wiped": pw}
    for rd in DEMO_RUNS:
        meta = json.load(open(os.path.join(rd, "meta.json")))
        d = out_dir(rd, "P-demo")
        recs, infra = load(d)
        row, o = [f"{meta['arm']} s{meta['seed']}"], {"out_dir": d, "infra": infra}
        c, o["overall"] = cells(recs)
        row.append(c)
        for g in groups:
            c, o[g] = cells([r for r in recs if r.get("layout_group") == g])
            row.append(c)
        L.append("| " + " | ".join(row) + " |")
        out["runs"][meta["run_id"]] = o
    rep = replay_reference(pd["entries"])
    row = ["open-loop replay of the recorded actions (reference; 1 per demo)"]
    c, out["replay_overall"] = cells(rep)
    row.append(c.replace("/40", "/40"))
    for g in groups:
        c, out[f"replay_{g}"] = cells([r for r in rep if r["layout_group"] == g])
        row.append(c)
    L.append("| " + " | ".join(row) + " |")
    L.append("\nReplay reference: `runs/replay/open_teleop` (render-free env, actions[1:] from states[0], runs the demo's own length, "
             "no 10-step post-success window; success = predicate at the last replayed step; primary safe = every entity > 95). "
             "Its denominators are 40 / 5 / 5 / 30 (one replay per demo).\n")
    L.append("## Per-layout comparison with P-train and P-devlayout (task success k/n · mean proportion wiped)\n")
    L.append("| Run | P-demo g01 | P-train g01 | P-demo g02 | P-train g02 | P-demo g06 | P-train g06 | P-devlayout g00 | P-devlayout g03 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for rd in DEMO_RUNS:
        meta = json.load(open(os.path.join(rd, "meta.json")))
        pdm = load(out_dir(rd, "P-demo"))[0]
        ptr = load(out_dir(rd, "P-train"))[0]
        pdv = load(out_dir(rd, "P-devlayout"))[0]

        def kc(recs, g):
            rr = [r for r in recs if r.get("layout_group") == g]
            return f"{sum(r['success'] for r in rr)}/{len(rr)} · {np.mean([r['final_proportion_wiped'] for r in rr]):.2f}" if rr else "–"
        L.append(f"| {meta['arm']} s{meta['seed']} | {kc(pdm, 'g01')} | {kc(ptr, 'g01')} | {kc(pdm, 'g02')} | {kc(ptr, 'g02')} | "
                 f"{kc(pdm, 'g06')} | {kc(ptr, 'g06')} | {kc(pdv, 'g00')} | {kc(pdv, 'g03')} |")
    L.append("\nP-train re-rolls joint noise and sponge placement around the same layout and base pose; P-demo restores the demo's "
             "exact start; P-devlayout layouts are unseen. P-train / P-devlayout have 20 resets x 2 seeds = 40 per group; P-demo 5 or 30 demos x 3 seeds.\n")
    L.append("## Integrity\n")
    for rid, o in out["runs"].items():
        L.append(f"- {rid}: {o['overall']['n']}/120 records, infra errors {o['infra']}, `{o['out_dir']}`")
    mdtxt = "\n".join(L) + "\n"
    open(os.path.join(C.V2_REPORTS, "demo_starts.md"), "w").write(mdtxt)
    json.dump(out, open(os.path.join(C.V2_REPORTS, "demo_starts.json"), "w"), indent=1, default=float)
    print(mdtxt)


HZ_RUNS = [("A-MIX s1", f"{MATRIX}/amix_wipe_countertop_v1_s1"), ("I-PATH s0", f"{MATRIX}/ipath_wipe_countertop_v1_s0")]


def exec_horizon():
    L = ["# wipe v2 diagnostic: execution-horizon sweep (existing checkpoints)\n"]
    L.append("Spec addendum (2026-10-26). Same checkpoints (step_30000), same evaluator and episode protocol; only the number of actions "
             "executed per predicted 8-step chunk before re-observing / re-planning changes (h = 8 official, 4, 1). Fresh x0 (and z for "
             "I-PATH) per chunk from the per-episode generator; frame stack 2; timeout 1250 env steps. Rollout seeds 0, 1. "
             "**h = 8 rows are the existing records** (P-devlayout main run; P-demo main run restricted to rollout seeds 0, 1), so "
             "comparisons are distributional. Smoothness = mean over steps of the L2 norm of the change in the executed 12-D action "
             "(arm-only 6-D in brackets); not recorded for h = 8 (the existing records do not log every action). P-demo uses the "
             "observable refresh after the state restore (as in demo_starts.md). Diagnostic, no pass criterion.\n")
    out = {}
    for pool in ("P-demo", "P-devlayout"):
        L.append(f"## {pool} (80 episodes per row)\n")
        L.append("| Arm | h | n | Task success | Wilson 95% | Primary safe success | Mean prop. wiped | Damaging | Mean steps to success | "
                 "Smoothness [arm 6-D] | Policy calls / episode | Infra |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, rd in HZ_RUNS:
            meta = json.load(open(os.path.join(rd, "meta.json")))
            for H in (8, 4, 1):
                if H == 8:
                    d = out_dir(rd, pool)
                else:
                    ds = sorted(glob.glob(os.path.join(C.V2_RUNS, "horizon", meta["run_id"], f"{pool}__step_30000__h{H}__*")), key=os.path.getmtime)
                    d = ds[-1] if ds else None
                if d is None:
                    L.append(f"| {name} | {H} | – | not run yet | | | | | | | | |")
                    continue
                recs, infra = load(d)
                recs = [r for r in recs if r["rollout_seed"] in (0, 1)]
                n = len(recs)
                k = sum(r["success"] for r in recs)
                lo, hi = wilson(k, n)
                ps = sum(r["primary_safe_success"] for r in recs)
                pw = float(np.mean([r["final_proportion_wiped"] for r in recs])) if n else float("nan")
                dmg = sum(r["damaging"] for r in recs)
                fs = [r["first_success_step"] for r in recs if r["success"] and r.get("first_success_step")]
                sm = [r["smoothness_mean_action_change_norm"] for r in recs if r.get("smoothness_mean_action_change_norm") is not None]
                sa = [r["smoothness_mean_action_change_norm_arm6"] for r in recs if r.get("smoothness_mean_action_change_norm_arm6") is not None]
                calls = float(np.mean([r["chunks"] for r in recs])) if n else float("nan")
                smt = f"{np.mean(sm):.4f} [{np.mean(sa):.4f}]" if sm else "–"
                L.append(f"| {name} | {H} | {n} | {k} ({pct(k / n if n else float('nan'))}) | {pct(lo)}–{pct(hi)} | {ps} | {pw:.3f} | {dmg} | "
                         f"{(np.mean(fs) if fs else float('nan')):.0f} | {smt} | {calls:.0f} | {infra} |")
                out[f"{pool}/{name}/h{H}"] = {"n": n, "success": k, "wilson95": [lo, hi], "primary_safe_success": ps, "mean_prop_wiped": pw,
                                              "damaging": dmg, "mean_steps_to_success": float(np.mean(fs)) if fs else None,
                                              "smoothness": float(np.mean(sm)) if sm else None, "policy_calls": calls, "out_dir": d}
        L.append("")
    md = "\n".join(L) + "\n"
    open(os.path.join(C.V2_REPORTS, "exec_horizon.md"), "w").write(md)
    json.dump(out, open(os.path.join(C.V2_REPORTS, "exec_horizon.json"), "w"), indent=1, default=float)
    print(md)


def all60(run_dirs, out_md=None, title="matched-condition check"):
    """Per run x pool (P-demo60, P-layout60, P-random under runs/wipe_v2/all60/eval): overall + per layout group."""
    root = os.path.join(C.V2_RUNS, "all60", "eval")
    L = [f"# wipe v2 {title}\n", "Cells: task success k/n (rate; Wilson 95%) · primary safe success k · mean / median final proportion wiped.\n"]
    seedrates = {}
    out = {}
    for pool in ("P-demo60", "P-layout60", "P-random"):
        rows = []
        for rd in run_dirs:
            meta = json.load(open(os.path.join(rd, "meta.json")))
            ds = sorted(glob.glob(os.path.join(root, meta["run_id"], f"{pool}__step_30000__*")), key=os.path.getmtime)
            if not ds:
                continue
            recs, infra = load(ds[-1])
            rows.append((meta, recs, infra, ds[-1]))
        if not rows:
            continue
        groups = sorted({r.get("layout_group") for _, recs, _, _ in rows for r in recs if r.get("layout_group")})
        L.append(f"## {pool}\n")
        L.append("| Run | Overall | " + "".join(g + " | " for g in groups) + "Infra |")
        L.append("|---|---|" + "---|" * len(groups) + "---|")

        def cell(rr):
            k = sum(r["success"] for r in rr)
            n = len(rr)
            lo, hi = wilson(k, n)
            ps = sum(r["primary_safe_success"] for r in rr)
            pw = float(np.mean([r["final_proportion_wiped"] for r in rr])) if n else float("nan")
            pm = float(np.median([r["final_proportion_wiped"] for r in rr])) if n else float("nan")
            return f"{k}/{n} ({pct(k / n if n else float('nan'))}; {pct(lo)}–{pct(hi)}) · {ps} · {pw:.2f} / {pm:.2f}"
        for meta, recs, infra, d in rows:
            L.append(f"| {meta['run_id']} | {cell(recs)} | "
                     + "".join(cell([r for r in recs if r.get('layout_group') == g]) + " | " for g in groups) + f"{infra} |")
            out[f"{pool}/{meta['run_id']}"] = {"n": len(recs), "success": sum(r["success"] for r in recs), "out_dir": d}
            rid = meta["run_id"]
            if "_all60_s" in rid and recs:
                arm = rid.split("_wipe_countertop_all60_")[0]
                seedrates.setdefault((pool, arm), []).append(
                    (rid, sum(r["success"] for r in recs) / len(recs), sum(r["primary_safe_success"] for r in recs) / len(recs),
                     float(np.mean([r["final_proportion_wiped"] for r in recs]))))
        L.append("")
    if seedrates:
        L.append("## Training-seed means (all60 runs; mean ± sd over training seeds of per-run rates)\n")
        L.append("| Pool | Arm | n seeds | Success | Safe success | Mean wiped | Seeds |")
        L.append("|---|---|---|---|---|---|---|")
        for (pool, arm), v in sorted(seedrates.items()):
            a_ = np.array([x[1:] for x in v])
            sd = a_.std(axis=0, ddof=1) if len(v) > 1 else np.full(3, np.nan)
            L.append(f"| {pool} | {arm} | {len(v)} | {pct(a_[:, 0].mean())} ± {pct(sd[0])} | {pct(a_[:, 1].mean())} ± {pct(sd[1])}"
                     f" | {a_[:, 2].mean():.2f} ± {sd[2]:.2f} | " + ", ".join(f"s{x[0][-1]} {pct(x[1])}" for x in v) + " |")
        L.append("")
    md = "\n".join(L) + "\n"
    if out_md:
        open(out_md, "w").write(md)
    print(md)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append")
    ap.add_argument("--ckpt", default="step_30000.pth")
    ap.add_argument("--calibration", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--demo-starts", action="store_true")
    ap.add_argument("--exec-horizon", action="store_true")
    ap.add_argument("--all60", action="store_true", help="with --run-dir: tables over the all60 pools")
    ap.add_argument("--out-md")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.calibration:
        return calibration()
    if a.ablation:
        return ablation()
    if a.demo_starts:
        return demo_starts()
    if a.exec_horizon:
        return exec_horizon()
    if a.all60:
        return all60([r.rstrip("/") for r in a.run_dir], a.out_md)
    ref = None
    try:
        ref = w0_train_reference()[2]
    except Exception as e:
        print("W0 reference unavailable:", e)
    res = []
    for rd in a.run_dir or []:
        s = summarize_run(rd.rstrip("/"), a.ckpt)
        if ref is not None:
            s["pass_criterion_vs_W0pooled"] = pass_criterion(s, ref)
        res.append(s)
        print(f"\n## {s['run_id']} ({s['arm']} s{s['seed']})\n")
        print(HDR)
        for p, st in s["pools"].items():
            print(row(f"{p} ({st['n']}/{st['expected']}, infra {st['infra_errors']})", st))
        if s.get("pass_criterion_vs_W0pooled"):
            print("\npass criterion (P-train reference = pooled W0 s0-s2):", json.dumps(s["pass_criterion_vs_W0pooled"]))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
