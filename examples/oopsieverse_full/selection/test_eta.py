"""Measured throughput / ETA of the TEST selection run (3 training seeds) -> selection/test_throughput_eta.json."""
import glob, json, os, time
import numpy as np
OVS = "/edrive1/lizad/oopsieverse_full_study"
STUDY = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study"
M, R, N_VAR, N_PRIM = 16, 4, 7, 4
now = time.time()
out = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": {}}
tot_done = tot_target = tot_prim_done = tot_prim_target = 0
mt_all, walls, steps = [], [], []
for s in (0, 1, 2):
    root = f"{OVS}/runs/selection/pick_egg/sel_test_parents/train_seed_{s}"
    st = [json.load(open(f)) for f in glob.glob(f"{root}/states/p*.json")]
    n_elig = sum(x["eligible"] for x in st) + (100 - len(st))
    files = glob.glob(f"{root}/branches/*/p*/c*_r*.json")
    prim = [f for f in files if "__primary" in f]
    mt = [os.path.getmtime(f) for f in files]
    mt_all += mt
    for f in files[-400:]:
        try:
            r = json.load(open(f)); walls.append(r["latency"]["wall_s"]); steps.append(r["continuation_steps"])
        except Exception:
            pass
    target = n_elig * M * R * N_VAR
    out["seeds"][s] = {"states": len(st), "eligible_states": sum(x["eligible"] for x in st), "done": len(files),
                       "primary_done": len(prim), "primary_target": n_elig * M * R * N_PRIM, "target_all": target,
                       "last_hour": sum(1 for t in mt if t > now - 3600)}
    tot_done += len(files); tot_target += target; tot_prim_done += len(prim); tot_prim_target += n_elig * M * R * N_PRIM
rate1 = sum(1 for t in mt_all if t > now - 3600)
rate3 = sum(1 for t in mt_all if t > now - 3 * 3600) / 3
try:
    b = [json.loads(l) for l in open(f"{OVS}/runs/selection/_gpus/budget_log.jsonl").readlines()[-3000:]]
    b = [x for x in b if x["t"] > now - 3600]
    mean_workers = float(np.mean([x["n_workers_all"] for x in b])) if b else None
except Exception:
    mean_workers = None
out.update({"done": tot_done, "target_all": tot_target, "primary_done": tot_prim_done, "primary_target": tot_prim_target,
            "continuations_last_hour": rate1, "continuations_per_hour_last_3h": rate3,
            "mean_workers_last_hour": mean_workers,
            "mean_worker_s_per_continuation_recent": float(np.mean(walls)) if walls else None,
            "mean_continuation_steps_recent": float(np.mean(steps)) if steps else None,
            "eta_hours_all_at_last_hour_rate": (tot_target - tot_done) / rate1 if rate1 else None,
            "eta_hours_all_at_3h_rate": (tot_target - tot_done) / rate3 if rate3 else None,
            "load_avg": os.getloadavg()})
json.dump(out, open(f"{STUDY}/selection/test_throughput_eta.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "seeds"}, indent=1))
