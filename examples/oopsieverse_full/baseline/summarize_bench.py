"""Summarise bench_scaling_W* dirs: per-episode reset/step/inference times and aggregate throughput."""
import glob, json, os, sys
import numpy as np
base = sys.argv[1]; out = sys.argv[2]
res = {}
for d in sorted(glob.glob(os.path.join(base, "bench_scaling_W*")), key=lambda p: int(p.split("W")[-1]) if p.split("W")[-1].isdigit() else 0):
    if not os.path.isdir(d):
        continue
    R = [json.load(open(f)) for f in glob.glob(os.path.join(d, "*.json")) if not os.path.basename(f).startswith("_")]
    R = [r for r in R if "infra_error" not in r]
    P = [json.load(open(f)) for f in glob.glob(os.path.join(d, "_pool_*.json"))]
    if not R or not P:
        continue
    W = P[0]["workers"]; wall = P[0]["wall_s"]
    steps = sum(r["steps"] for r in R)
    res[f"W{W}"] = {
        "workers": W, "episodes": len(R), "steps_per_episode": R[0]["steps"], "pool_wall_s": round(wall, 1),
        "env_build_s_mean": round(float(np.mean([r["worker_env_build_s"] for r in R])), 1),
        "reset_s_mean": round(float(np.mean([r["reset_s"] for r in R])), 1),
        "env_step_s_per_step": round(sum(r["env_step_s"] for r in R) / steps, 4),
        "infer_s_per_chunk": round(sum(r["infer_s"] for r in R) / sum(r["chunks"] for r in R), 4),
        "episode_wall_s_mean": round(float(np.mean([r["wall_s"] for r in R])), 1),
        "per_worker_step_rate_hz": round(steps / sum(r["wall_s"] - r["reset_s"] for r in R), 2),
        "aggregate_env_steps_per_s_incl_reset_build": round(steps / wall, 2),
    }
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
