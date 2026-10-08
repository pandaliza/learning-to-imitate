"""Episodes/h of the current test evaluation since time T0 (all run dirs), GPU-3 memory from the latest monitor log."""
import glob, json, os, sys, time
T0 = float(sys.argv[1]); win = float(sys.argv[2]) if len(sys.argv) > 2 else None
fs = [f for f in glob.glob("/edrive1/lizad/oopsieverse_full_study/runs/matrix_eval/*/test_*/*__r*.json") if os.path.getmtime(f) > T0]
now = time.time(); t1 = T0 if win is None else max(T0, now - win)
n = sum(1 for f in fs if os.path.getmtime(f) > t1)
mons = sorted(glob.glob("/edrive1/lizad/oopsieverse_full_study/runs/matrix_eval/*/test_*/_gpu3_monitor_*.jsonl"), key=os.path.getmtime)
L = [json.loads(l) for l in open(mons[-1])] if mons else []
L = [l for l in L if l["t"] > t1]
print(json.dumps({"episodes": n, "window_min": round((now - t1) / 60, 1), "episodes_per_h": round(n / ((now - t1) / 3600), 1),
                  "gpu3_ours_peak_mib": max([l.get("peak_ours_mib", 0) for l in L] or [0]),
                  "gpu3_ours_last_mib": ([l["ours_mib"] for l in L if "ours_mib" in l] or [None])[-1],
                  "gpu3_free_min_mib": min([l["free_mib"] for l in L if "free_mib" in l] or [0]),
                  "pauses": sum(1 for l in L if l.get("event") == "pause"), "load": os.getloadavg()}))
