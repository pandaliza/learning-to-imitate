"""Updates/s per running training job over the last N seconds (from metrics.jsonl)."""
import glob, json, os, sys, time
win = float(sys.argv[1]) if len(sys.argv) > 1 else 900
now = time.time(); out = {}
for p in glob.glob("/edrive1/lizad/oopsieverse_full_study/runs/matrix/*/metrics.jsonl"):
    if os.path.islink(p) or now - os.path.getmtime(p) > 120:
        continue
    L = [json.loads(l) for l in open(p)]
    t0 = L[0].get("time")
    pts = [(d["step"], t0 + d["wall_s"]) for d in L if "train/loss" in d and t0 + d["wall_s"] > now - win]
    if len(pts) > 1:
        out[p.split("/")[-2]] = round((pts[-1][0] - pts[0][0]) / (pts[-1][1] - pts[0][1]), 2)
print(json.dumps({"window_s": win, "per_job": out, "total": round(sum(out.values()), 2)}))
