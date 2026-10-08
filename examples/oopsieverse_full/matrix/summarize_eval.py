"""Per-run random-policy test-evaluation tables (no cross-arm aggregation, no conclusions).

  python summarize_eval.py <eval_dir> <out_json>       -> one run's summary
  python summarize_eval.py --index                     -> rebuild matrix/eval/per_run_test_tables.md from all summaries
"""
import glob
import json
import math
import os
import sys

STUDY = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study"
OUT = os.path.join(STUDY, "matrix/eval")


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def block(recs):
    n = len(recs)
    k = lambda f: sum(1 for r in recs if f(r))  # noqa: E731
    out = {"n": n,
           "success": k(lambda r: r["success"]), "primary_safe_success": k(lambda r: r["primary_safe_success"]),
           "damaging_success": k(lambda r: r["outcome"] == "damaging_success"),
           "safe_failure": k(lambda r: r["outcome"] == "safe_failure"),
           "damaging_failure": k(lambda r: r["outcome"] == "damaging_failure"),
           "secondary_safe_success": k(lambda r: r["secondary_safe_success"]),
           "compat_safe_success": k(lambda r: r["compat_safe_success"]),
           "invalid_action": k(lambda r: r.get("invalid_action")),
           "mean_steps": round(sum(r["steps"] for r in recs) / max(n, 1), 1),
           "mean_worst_entity_loss": round(sum(r["worst_entity_loss"] for r in recs) / max(n, 1), 2)}
    h = [r for r in recs if r.get("success_h010") is not None]
    out["success_h010"] = k(lambda r: r.get("success_h010")) if h else None
    out["success_pct"] = round(100 * out["success"] / max(n, 1), 1)
    out["success_ci95"] = wilson(out["success"], n)
    out["primary_safe_success_pct"] = round(100 * out["primary_safe_success"] / max(n, 1), 1)
    out["primary_safe_success_ci95"] = wilson(out["primary_safe_success"], n)
    return out


def summarize(eval_dir, out_json):
    recs, infra = [], []
    for f in sorted(glob.glob(os.path.join(eval_dir, "*__r*.json"))):
        r = json.load(open(f))
        (infra if "infra_error" in r else recs).append(r)
    n_infra_hist = len(glob.glob(os.path.join(eval_dir, "_infra", "*.json")))
    meta = recs[0] if recs else {}
    s = {"eval_dir": eval_dir, "run_id": meta.get("run_id"), "arm": meta.get("arm"), "task": meta.get("task"),
         "training_seed": meta.get("training_seed"), "run_key": meta.get("run_key"),
         "checkpoint": os.path.basename(meta.get("ckpt", "")), "checkpoint_sha256_16": meta.get("checkpoint_sha256_16"),
         "metric_version": meta.get("metric_version"), "nfe_per_chunk": meta.get("nfe_per_chunk"),
         "reset_pool": "eval_test (frozen reset_banks_v1)", "n_records": len(recs),
         "n_infra_error_pending": len(infra), "n_infra_error_retried": n_infra_hist,
         "all": block(recs),
         "by_rollout_seed": {str(sd): block([r for r in recs if r["rollout_seed"] == sd]) for sd in sorted({r["rollout_seed"] for r in recs})},
         "in_support": block([r for r in recs if r["reset"].get("in_support") is True]),
         "out_of_support": block([r for r in recs if r["reset"].get("in_support") is False]),
         "note": "per-run descriptive table; CIs are per-run Wilson intervals over episodes (rollout seeds and resets "
                 "are not independent training replicates). No cross-arm comparison until all seeds of an arm exist."}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(s, open(out_json + ".tmp", "w"), indent=1)
    os.replace(out_json + ".tmp", out_json)
    return s


def index():
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "test_*.json"))):
        rows.append(json.load(open(f)))
    L = ["# Per-run random-policy test evaluation (Amendment 2)", "",
         "Frozen `eval_test` bank: 200 reset families × 3 rollout seeds, 30k checkpoint, contract oopsie-metric-v1.0, "
         "native stochastic sampling. Rendering: EGL on GPU 3 (exact-parity verified); inference on GPU 0.",
         "**Descriptive per-run tables only. Not aggregated across arms; no conclusions.** Percentages are over episodes.", "",
         "| Run | n | Success % [95% CI] | Primary safe success % [95% CI] | Damaging success | Safe failure | "
         "Damaging failure | Strict safe success | Compat safe success | Success at 0.10 m | In-support: n / success / safe success | "
         "Out-of-support: n / success / safe success | Infra errors (pending / retried) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        a, i, o = s["all"], s["in_support"], s["out_of_support"]
        L.append(f"| {s['run_id']} | {a['n']} | {a['success_pct']} {a['success_ci95']} | {a['primary_safe_success_pct']} "
                 f"{a['primary_safe_success_ci95']} | {a['damaging_success']} | {a['safe_failure']} | {a['damaging_failure']} | "
                 f"{a['secondary_safe_success']} | {a['compat_safe_success']} | {a['success_h010']} | "
                 f"{i['n']} / {i['success']} / {i['primary_safe_success']} | {o['n']} / {o['success']} / {o['primary_safe_success']} | "
                 f"{s['n_infra_error_pending']} / {s['n_infra_error_retried']} |")
    L += ["", "Success at 0.10 m: sensitivity on the final executed step (pick_egg only; episodes terminate on the 0.05 m "
          "predicate, so this under-counts). Per-rollout-seed blocks are in each `test_<run>.json`."]
    open(os.path.join(OUT, "per_run_test_tables.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "--index":
        index()
    else:
        print(json.dumps(summarize(sys.argv[1], sys.argv[2])["all"], indent=1))
