"""Aggregate diag_rollout.py records into wipe_diag/diag_summary.json (+ per-episode table)."""
import glob
import json
import os

import numpy as np

RUNS = "/edrive1/lizad/oopsieverse_full_study/runs/wipe_diag"
OUT = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/wipe_diag"
TRAIN = {f"with/demo_{i}" for i in range(10, 15)} | {f"without/demo_{i}" for i in range(5)}


def main():
    summ, eps = {}, []
    for d in sorted(glob.glob(os.path.join(RUNS, "amix_s*", "*"))):
        if not os.path.isdir(d):
            continue
        run, cond = d.split("/")[-2], d.split("/")[-1]
        R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*__r0.json")))]
        R = [r for r in R if "infra_error" not in r]
        groups = {cond: R}
        if cond == "demo":
            key = lambda r: r["episode_id"].replace("demo__", "").replace("__", "/")
            groups = {"demo_train": [r for r in R if key(r) in TRAIN], "demo_dev": [r for r in R if key(r) not in TRAIN]}
        for g, rs in groups.items():
            if not rs:
                continue
            w = np.array([r["final_wiped"] for r in rs])
            summ[f"{run}/{g}"] = {
                "n": len(rs), "success": int(sum(r["success"] for r in rs)),
                "primary_safe_success": int(sum(r["primary_safe_success"] for r in rs)),
                "final_wiped_mean": float(w.mean()), "final_wiped_median": float(np.median(w)),
                "final_wiped_p25_p75": [float(np.percentile(w, 25)), float(np.percentile(w, 75))],
                "n_wiped0": int((w == 0).sum()), "n_ge40": int((w >= 40).sum()), "n_ge45": int((w >= 45).sum()),
                "final_wiped_sorted": sorted(w.tolist()),
                "timeouts": int(sum(r["steps"] >= 1250 for r in rs)),
                "median_wiped_at_300": float(np.median([r["wiped_at"].get("300", r["final_wiped"]) for r in rs])),
                "frac_wiped_after_300_of_final": float(np.mean([(r["final_wiped"] - r["wiped_at"].get("300", r["final_wiped"])) for r in rs])),
            }
            for r in rs:
                eps.append({k: r.get(k) for k in ("episode_id", "cond", "training_seed", "steps", "success", "final_wiped",
                                                   "first_success_step", "outcome", "worst_entity_loss", "wiped_at",
                                                   "steps_fdiff_gt100", "demo_len")} | {"group": f"{run}/{g}"})
    json.dump({"summary": summ, "episodes": eps}, open(os.path.join(OUT, "diag_summary.json"), "w"), indent=1)
    for k, v in summ.items():
        print(k, {x: v[x] for x in ("n", "success", "final_wiped_median", "final_wiped_mean", "n_wiped0", "n_ge45", "median_wiped_at_300")})


if __name__ == "__main__":
    main()
