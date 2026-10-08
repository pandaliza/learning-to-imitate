"""Summarise a competence screen directory of per-episode records (rollout_eval.py) into JSON + a
markdown table. Usage: python summarize_screen.py EVAL_DIR OUT_JSON LABEL"""
import glob
import json
import os
import sys

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def main(d, out, label):
    recs, errs = [], []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(f).startswith("_"):
            continue
        r = json.load(open(f))
        (errs if "infra_error" in r else recs).append(r)
    n = len(recs)

    def cnt(key):
        return sum(bool(r.get(key)) for r in recs)

    oc = {k: sum(r["outcome"] == k for r in recs) for k in ["safe_success", "damaging_success", "safe_failure", "damaging_failure"]}
    s = {
        "label": label, "eval_dir": d, "n_episodes": n, "infra_errors": len(errs),
        "ckpt": recs[0]["ckpt"] if recs else None, "metric_version": recs[0]["metric_version"] if recs else None,
        "sampling": recs[0]["sampling"] if recs else None, "exec_horizon": recs[0]["exec_horizon"] if recs else None,
        "nfe": recs[0]["nfe"] if recs else None,
        "task_success": cnt("success"), "task_success_h010": cnt("success_h010"),
        "primary_safe_success": cnt("primary_safe_success"), "secondary_safe_success": cnt("secondary_safe_success"),
        "compat_safe_success": cnt("compat_safe_success"),
        "outcomes": oc,
        "damaging_success_primary": oc["damaging_success"],
        "any_damage_episodes": sum(r["worst_entity_loss"] > 1e-3 for r in recs),
        "ever_succeeded": sum(r["first_success_step"] is not None for r in recs),
        "invalid_action": cnt("invalid_action"),
        "rates": {k: [round(v / n, 3) if n else None, wilson(v, n)] for k, v in
                  [("task_success", cnt("success")), ("primary_safe_success", cnt("primary_safe_success")),
                   ("secondary_safe_success", cnt("secondary_safe_success")), ("compat_safe_success", cnt("compat_safe_success")),
                   ("damaging_success", oc["damaging_success"])]},
        "mean_entity_loss": {e: round(float(np.mean([r["entity_loss"][e] for r in recs])), 3) for e in (recs[0]["entity_loss"] if recs else {})},
        "median_worst_entity_loss": float(np.median([r["worst_entity_loss"] for r in recs])) if recs else None,
        "median_max_contact_force_N": float(np.median([r["max_contact_force_N"] for r in recs])) if recs else None,
        "median_max_impact_force_egg_N": float(np.median([r["max_impact_force_target_N"] for r in recs])) if recs else None,
        "steps_mean": float(np.mean([r["steps"] for r in recs])) if recs else None,
        "steps_total": int(sum(r["steps"] for r in recs)),
        "first_success_step_median": float(np.median([r["first_success_step"] for r in recs if r["first_success_step"]])) if any(r["first_success_step"] for r in recs) else None,
        "wall_s_mean": float(np.mean([r["wall_s"] for r in recs])) if recs else None,
        "reset_s_mean": float(np.mean([r["reset_s"] for r in recs])) if recs else None,
        "env_step_s_per_step": float(sum(r["env_step_s"] for r in recs) / max(1, sum(r["steps"] for r in recs))),
        "infer_s_per_chunk": float(sum(r["infer_s"] for r in recs) / max(1, sum(r["chunks"] for r in recs))),
        "episodes": [{k: r.get(k) for k in ["episode_id", "steps", "first_success_step", "success", "success_h010", "outcome",
                                             "entity_final_health", "worst_entity_loss", "max_contact_force_N",
                                             "max_impact_force_target_N", "egg_height_max", "egg_height_final", "first_damage_step", "reset"]}
                     for r in recs],
    }
    for flag in (True, False):
        R = [r for r in recs if r["reset"].get("in_support") is flag]
        oc2 = {k: sum(r["outcome"] == k for r in R) for k in ["safe_success", "damaging_success", "safe_failure", "damaging_failure"]}
        s[f"in_support={flag}"] = {"n": len(R), "task_success": sum(r["success"] for r in R),
                                   "primary_safe_success": sum(r["primary_safe_success"] for r in R),
                                   "secondary_safe_success": sum(r["secondary_safe_success"] for r in R),
                                   "compat_safe_success": sum(r["compat_safe_success"] for r in R), "outcomes": oc2}
    s["reset_obj_pos_bank_maxabs_max"] = max((r.get("reset_obj_pos_bank_maxabs") or 0) for r in recs) if recs else None
    json.dump(s, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in s.items() if k != "episodes"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
