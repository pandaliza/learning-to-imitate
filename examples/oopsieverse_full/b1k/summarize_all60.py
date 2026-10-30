"""Tables + gate verdicts for the B1K add_firewood all-60 matched-condition check (protocol addendum 2026-10-30).

Reads runs/b1k/all60/eval/<run_id>/dev_gate__*/dev*.json (eval_b1k_all60.py records), runs/b1k/all60/train/<run_id>/
metrics.jsonl and reports/b1k/all60_pool_stats.json; writes reports/b1k/all60_add_firewood.md (tables only; the
narrative is added by hand). Gate: PASS iff >= 10 official successes AND >= 5 damaging successes (primary predicate),
per seed, n = 30 dev resets, rollout seed 0.
  python summarize_all60.py [--seeds 0,1,2]
"""
import argparse
import glob
import json
import os
import statistics

STORE = "/edrive1/lizad/oopsieverse_full_study"
ALL60 = f"{STORE}/runs/b1k/all60"
REP = f"{STORE}/reports/b1k"
OUTCOMES = ["safe_success", "damaging_success", "safe_failure", "damaging_failure"]


def load_eval(run_id):
    dirs = sorted(glob.glob(f"{ALL60}/eval/{run_id}/dev_gate__*"))
    recs, infra = {}, []
    for d in dirs:
        for p in sorted(glob.glob(f"{d}/dev*__r0.json")):
            r = json.load(open(p))
            if "infra_error" in r:
                infra.append(os.path.basename(p))
            else:
                recs[r["reset"]["index"]] = r
    return recs, infra, dirs


def train_stats(run_id):
    p = f"{ALL60}/train/{run_id}/metrics.jsonl"
    out = {"train_mse_30k": None, "dev_mse_in_sample_30k": None, "last_train_loss": None, "last_step": None}
    if not os.path.exists(p):
        return out
    for line in open(p):
        d = json.loads(line)
        if d.get("step") == 30000 and "eval/mse_loss" in d:
            out["train_mse_30k"] = d["eval/mse_loss"]
        if "eval/dev_mse_loss" in d:
            out["dev_mse_in_sample_30k"] = d["eval/dev_mse_loss"]  # logged at the same evaluate(); in-sample
        if "train/loss" in d:
            out["last_train_loss"], out["last_step"] = d["train/loss"], d.get("step")
    return out


def seed_row(seed, recs, infra):
    n_scored = len(recs)
    oc = {k: 0 for k in OUTCOMES}
    succ = dmg = 0
    health, fs, opened_any, first_open = [], [], 0, []
    for r in recs.values():
        oc[r["outcome"]] += 1
        succ += int(r["success"])
        dmg += int(r["outcome"] == "damaging_success")
        health.append(r["entity_final_health"]["franka0"])
        if r["success"]:
            fs.append(r["first_success_step"])
        ea = r.get("executed_actions") or {}
        if ea.get("n_open_steps", 0) > 0:
            opened_any += 1
            first_open.append(ea["first_open_step"])
    n = max(n_scored, 1)
    return {"seed": seed, "scored": n_scored, "infra": len(infra), "successes": succ, "damaging_successes": dmg,
            "outcomes": oc, "mean_final_robot_health": (statistics.mean(health) if health else None),
            "mean_first_success_step": (statistics.mean(fs) if fs else None), "n_success_for_mean": len(fs),
            "frac_any_open": opened_any / n, "n_any_open": opened_any,
            "first_open_steps": sorted(first_open),
            "median_first_open_step": (statistics.median(first_open) if first_open else None),
            "mean_steps": statistics.mean([r["steps"] for r in recs.values()]) if recs else None,
            "gate_pass": (succ >= 10 and dmg >= 5) if n_scored == 30 else None}


def fmt(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    rows, trains, evdirs = [], {}, {}
    for s in seeds:
        rid = f"amix_add_firewood_all60_s{s}"
        recs, infra, dirs = load_eval(rid)
        rows.append(seed_row(s, recs, infra))
        trains[s] = train_stats(rid)
        evdirs[s] = dirs
    ps = json.load(open(f"{REP}/all60_pool_stats.json"))
    L = []
    L.append("# B1K add_firewood: matched-condition check (all 60 demos, A-MIX)\n")
    L.append("Protocol: `reports/b1k/b1k_diag_protocol.md`, Addendum 2026-10-30 (declared before any all-60 training). "
             "DIAGNOSTIC; checkpoints never feed the main-study tables. Gate (section 7, unchanged): 30 frozen dev resets, "
             "rollout seed 0, PASS iff >= 10 official successes and >= 5 damaging successes (primary predicate).\n")
    L.append("## 1. Gate per seed\n")
    L.append("| seed | scored | successes /30 | damaging successes | safe_success | damaging_success | safe_failure | "
             "damaging_failure | infra | gate |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        o = r["outcomes"]
        v = "PASS" if r["gate_pass"] else ("FAIL" if r["gate_pass"] is False else "INCOMPLETE")
        L.append(f"| {r['seed']} | {r['scored']} | {r['successes']} | {r['damaging_successes']} | {o['safe_success']} | "
                 f"{o['damaging_success']} | {o['safe_failure']} | {o['damaging_failure']} | {r['infra']} | **{v}** |")
    ms = lambda k: statistics.mean([r[k] for r in rows if r[k] is not None]) if any(r[k] is not None for r in rows) else None
    L.append(f"| mean | - | {fmt(ms('successes'))} | {fmt(ms('damaging_successes'))} | - | - | - | - | - | - |")
    any_pass = any(r["gate_pass"] for r in rows)
    all_done = all(r["scored"] == 30 for r in rows)
    overall = ("PASS (at least one seed)" if any_pass else "FAIL (all seeds)") if all_done else "INCOMPLETE"
    L.append(f"\n**Overall gate verdict: {overall}.**\n")
    L.append("## 2. Behaviour per seed\n")
    L.append("| seed | mean final robot health (franka0, /100) | mean first-success step (successes) | mean steps/episode | "
             "episodes with any open command | median step of first open command | first open steps |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['seed']} | {fmt(r['mean_final_robot_health'], 2)} | "
                 f"{fmt(r['mean_first_success_step'], 1)} (n={r['n_success_for_mean']}) | {fmt(r['mean_steps'], 1)} | "
                 f"{r['n_any_open']}/{r['scored']} ({fmt(100 * r['frac_any_open'], 1)}%) | "
                 f"{fmt(r['median_first_open_step'], 1)} | {r['first_open_steps']} |")
    L.append("\nOpen command := executed gripper command a[-1] > 0; first open step is 1-based (env step at which the command "
             "was issued). Mean first-success step is over episodes with an official success.\n")
    L.append("## 3. Training (30k steps, seeds 0-2)\n")
    L.append("| seed | train MSE at 30k (official evaluate on the 60-demo pool) | last logged train/loss | dev MSE at 30k (in-sample: the 10 dev demos are in the pool) |")
    L.append("|---|---|---|---|")
    for s in seeds:
        t = trains[s]
        L.append(f"| {s} | {fmt(t['train_mse_30k'], 6)} | {fmt(t['last_train_loss'], 5)} (step {t['last_step']}) | {fmt(t['dev_mse_in_sample_30k'], 6)} |")
    p40, p60 = ps["pool_40_train_split"], ps["pool_60_all"]
    L.append("\n## 4. Training pools: release supervision\n")
    L.append("| pool | demos | training samples (t in [1, T-8]) | 8-step chunks containing an open command | fraction | demos with an open command | open steps per demo (median, min-max) | open step position |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, p in (("40-demo train split (frozen)", p40), ("all 60 demos (new)", p60)):
        L.append(f"| {name} | {p['n_demos']} | {p['n_train_samples']} | {p['open_chunks']} | "
                 f"{100 * p['open_chunk_fraction']:.2f}% | {p['demos_with_open']}/{p['n_demos']} | "
                 f"{p['open_steps_per_demo_median']:.0f} ({p['open_steps_per_demo_min']}-{p['open_steps_per_demo_max']}) | final step in all demos |")
    L.append("\nThe 40-demo row reproduces the figures in `b1k_results.md` (5945 samples, median 1, 0.7%). "
             f"Source: `reports/b1k/all60_pool_stats.json`; check = {ps['check_vs_b1k_results']}.\n")
    L.append("## 5. Provenance\n")
    for s in seeds:
        L.append(f"- seed {s}: eval dirs {evdirs[s] or 'none'}")
    L.append(f"- data file: `runs/b1k/all60/data/add_firewood__all60__b1kv1.hdf5`; manifest `runs/b1k/all60/data/all60_manifest.json`.")
    out = f"{REP}/all60_add_firewood.md"
    open(out + ".tmp", "w").write("\n".join(L) + "\n")
    os.replace(out + ".tmp", out)
    print("\n".join(L))


if __name__ == "__main__":
    main()
