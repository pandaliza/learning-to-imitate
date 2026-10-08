"""Candidate-selection branching runner (spec §11, §14; protocol_v1) — nohup, resumable, GPU-3 budgeted.

  python run_selection.py --task pick_egg --pool sel_dev_parents --train-seed 0 [--max-workers 8] [--parents 0-49]

Design (declared 2026-10-08 before any branch outcome was produced):
  * parents: reset_banks_v1 pool entries (sel_dev_parents here), canonical reset by parent seed;
  * intermediate state (one per parent x training seed): KeyedRNG('oopsie_full_v1').derive_seed('state_collection',
    experiment='oopsie_full_v1/selection/<task>/<pool>', checkpoint_pair='train_seed_<k>', state_id=<parent index>)
    seeds numpy default_rng -> source policy ~ Uniform{amix, imean, ipath, ipathg} (balanced mixture of the four
    proposal arms, seed-k checkpoints) -> per attempt: branch chunk c ~ Uniform{1..12} (t_b = 8c in [8, 96], the
    approach phase before the earliest dev A-MIX damage onset, step 90) and a source-rollout noise seed. The source
    policy runs the evaluator protocol to t_b. Eligible iff no link lost health (strict, all 100), no success so
    far, no invalid action, episode not ended. Up to 4 attempts (source policy fixed per parent, time/noise
    redrawn); every attempt is logged (attrition by source policy); a parent with no eligible attempt is dropped.
  * candidates: M=16 per (state, arm, variant), written to candidates.npz BEFORE any continuation:
      A-MIX primary:  x0_m from the 'candidate' namespace (candidate_id=m);
      intent primary: z_m from 'candidate' (candidate_id=m), decoder x0 FIXED from 'decoder' (candidate_id=-1);
      intent joint (sensitivity, labelled): z_m from 'candidate' (the same z_m as primary) and x0_m from 'decoder'
                      (candidate_id=m).
    Keys: experiment, arm, checkpoint_pair, state_id, candidate_id. Draws: torch.Generator(seed).randn in the native
    shapes. Decoded on GPU 0 in one fixed-shape batch of 16.
  * continuation: execute the candidate's native chunk (8 steps), then the arm's own frozen policy (same
    checkpoint, native stochastic sampling) to the original timeout (650 steps from the parent reset), with tail
    replicates r=0..3 (0-1 rank, 2-3 validate). Tail noise: torch.Generator(derive_seed('continuation',
    experiment, checkpoint_pair, state_id, continuation_id=r, policy_identity='<arm>_tail')) — shared by all
    candidates (and both variants) of an arm; separate from proposal RNG, the env RNG and physics.
    The post-chunk state is snapshotted once per candidate and restored for each tail (validated: T6).
  * records: one atomic JSON per continuation (records written with os.replace), infra errors kept in _infra/ and
    retried <= 2x. Run key = hash(config, code, checkpoints, metric version, bank); resume refuses a changed key.
"""

import argparse
import glob
import hashlib
import json
import os
import signal
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from branch_worker import *  # noqa: E402,F401,F403
from branch_worker import worker_main  # noqa: E402


# ───────────────────────────── supervisor ─────────────────────────────

def scan(root, parents_idx):
    st = {"states": {}, "groups": {}}
    for p in parents_idx:
        f = state_path(root, p) + ".json"
        if os.path.exists(f):
            st["states"][p] = json.load(open(f))
    for arm, var in variants():
        for p in parents_idx:
            gd = group_dir(root, arm, var, p)
            names = set(os.listdir(gd)) if os.path.isdir(gd) else set()
            done = {n[:-5] for n in names if n.startswith("c") and n.endswith(".json") and "_r" in n}
            inf = os.listdir(f"{gd}/_infra") if "_infra" in names else []
            cnt = {}
            for n in inf:
                cnt[n.split(".")[0]] = cnt.get(n.split(".")[0], 0) + 1
            exhausted = sum(1 for k, v in cnt.items() if v > 2 and k not in done)
            st["groups"][(p, arm, var)] = {"n": len(done), "exhausted": exhausted, "infra_files": len(inf)}
    return st


def status_report(cfg, root, parents_idx, st, budget, workers, t_start, n_start, extra):
    recs = []
    for f in glob.glob(f"{root}/branches/*/p*/c*_r*.json"):
        try:
            r = json.load(open(f))
            recs.append((r["arm"], r["variant"], r["latency"]["wall_s"], r["steps"] - r["state_t_b"], os.path.getmtime(f)))
        except Exception:
            pass
    now = time.time()
    by = {}
    for arm, var in variants():
        tot = len(parents_idx) * M * R
        by[f"{arm}__{var}"] = {"done": sum(1 for x in recs if x[0] == arm and x[1] == var), "target_if_all_eligible": tot}
    elig = [p for p, s in st["states"].items() if s["eligible"]]
    att = {}
    for s in st["states"].values():
        a = att.setdefault(s["source_arm"], {"parents": 0, "eligible": 0, "attempts": 0, "attrition_reasons": {}})
        a["parents"] += 1
        a["eligible"] += int(s["eligible"])
        a["attempts"] += len(s["attempts"])
        for x in s["attempts"]:
            if x["outcome"] != "eligible":
                key = x["outcome"].split(" (")[0]
                a["attrition_reasons"][key] = a["attrition_reasons"].get(key, 0) + 1
    win = [x for x in recs if x[4] > now - 3600]
    n_drop = sum(1 for s in st["states"].values() if not s["eligible"])
    n_states_target = len(parents_idx) - n_drop          # uncollected parents assumed eligible
    n_prim = n_states_target * M * R * len(PROPOSAL_ARMS)
    n_sens = n_states_target * M * R * (len(variants()) - len(PROPOSAL_ARMS))
    done_prim = sum(v["done"] for k, v in by.items() if k.endswith("primary"))
    done_all = sum(v["done"] for v in by.values())
    worker_s = [x[2] for x in recs]
    steps = [x[3] for x in recs]
    d = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "task": cfg["task"], "pool": cfg["pool"],
        "training_seed": cfg["train_seed"], "run_key": cfg["run_key"], "root": root,
        "parents": len(parents_idx), "states_collected": len(st["states"]), "states_eligible": len(elig),
        "source_policy_attrition": att, "continuations_done": by,
        "primary_done": done_prim, "primary_target": n_prim, "sensitivity_done": done_all - done_prim,
        "sensitivity_target": n_sens, "workers_running": workers, "gpu3_budget": budget,
        "throughput": {
            "continuations_last_hour": len(win),
            "continuations_per_hour_since_start": (done_all - n_start) / max(1e-6, (now - t_start) / 3600),
            "mean_worker_s_per_continuation": float(np.mean(worker_s)) if worker_s else None,
            "mean_continuation_steps": float(np.mean(steps)) if steps else None,
            "worker_s_per_env_step": float(np.sum(worker_s) / max(1, np.sum(steps))) if steps else None},
        **extra}
    rate = d["throughput"]["continuations_last_hour"] or None
    d["eta_hours_at_last_hour_rate"] = {"primary": (n_prim - done_prim) / rate if rate else None,
                                        "all_incl_sensitivity": (n_prim + n_sens - done_all) / rate if rate else None}
    atomic_json(d, cfg["status_path"])
    return d


def supervisor(cfg, parents_idx, max_workers):
    import multiprocessing as mp
    import gpu3_budget as GB
    root = cfg["root"]
    ctx = mp.get_context("spawn")
    os.makedirs(f"{root}/states", exist_ok=True)
    log = open(f"{root}/supervisor.log", "a")

    def L(*a):
        log.write(time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a) + "\n")
        log.flush()
    SLOTS = 18
    req_q = ctx.Queue()
    resp = [ctx.Queue() for _ in range(SLOTS)]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    import harness as H
    srv = ctx.Process(target=H.inference_server, args=(run_dirs(cfg["task"], cfg["train_seed"]), CKPT, "cuda:0", req_q, resp,
                                                       f"{root}/_server_stats.json", 16))
    srv.start()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    res_q = ctx.Queue()
    workers = {}  # wid -> {"proc", "task_q", "stop", "unit", "parent", "t_spawn"}
    in_flight = set()
    t_start = time.time()
    st = scan(root, parents_idx)
    n_start = sum(v["n"] for v in st["groups"].values())
    last_status = 0
    last_spawn = 0
    grant_hist = []
    stopping = False

    def pending_units():
        units = []
        for p in parents_idx:
            if p not in st["states"]:
                units.append(("collect", p))
        order = [v for v in variants() if v[1] == "primary"] + [v for v in variants() if v[1] != "primary"]
        for arm, var in order:
            for p in parents_idx:
                s = st["states"].get(p)
                if s is None or not s["eligible"]:
                    continue
                g = st["groups"][(p, arm, var)]
                if g["n"] + g["exhausted"] < M * R:
                    units.append(("group", p, arm, var))
        # primary-arm groups parent-major so partial results stay balanced across arms
        prim = [u for u in units if u[0] == "group" and u[3] == "primary"]
        prim.sort(key=lambda u: (u[1], PROPOSAL_ARMS.index(u[2])))
        rest = [u for u in units if u[0] == "group" and u[3] != "primary"]
        rest.sort(key=lambda u: (u[1], u[2]))
        return [u for u in units if u[0] == "collect"] + prim + rest

    def assign(wid):
        w = workers[wid]
        cand = [u for u in pending_units() if u not in in_flight]
        if not cand:
            return False
        same = [u for u in cand if u[0] == "group" and u[1] == w.get("parent")]
        u = same[0] if same else cand[0]
        in_flight.add(u)
        w["unit"] = u
        w["parent"] = u[1]
        w["task_q"].put(u)
        return True

    def handle(m):
        wid = m["wid"]
        w = workers.get(wid)
        if m["event"] == "ready":
            L("worker", wid, "ready", round(m["env_build_s"], 1))
            if w and not w["stop"].is_set() and not stopping:
                assign(wid) or w["task_q"].put(None)
        elif m["event"] in ("done", "error"):
            u = tuple(m["unit"])
            in_flight.discard(u)
            if m["event"] == "error":
                L("unit error", u, m["tb"][-1500:])
            nonlocal st
            st = scan(root, parents_idx)
            if w and not w["stop"].is_set() and not stopping:
                assign(wid) or w["task_q"].put(None)
            elif w:
                w["task_q"].put(None)
        elif m["event"] == "crash":
            L("worker crash", wid, m["tb"][-2000:])
        elif m["event"] == "exit":
            if w:
                if w.get("unit"):
                    in_flight.discard(w["unit"])
                w["proc"].join(timeout=30)
                workers.pop(wid, None)
                L("worker exit", wid)

    def spawn():
        free = [i for i in range(SLOTS) if i not in workers]
        wid = free[0]
        tq = ctx.Queue()
        ev = ctx.Event()
        p = ctx.Process(target=worker_main, args=(wid, cfg, req_q, resp[wid], tq, res_q, ev))
        p.start()
        workers[wid] = {"proc": p, "task_q": tq, "stop": ev, "unit": None, "parent": None, "t_spawn": time.time()}
        L("spawn worker", wid, "pid", p.pid)

    def on_term(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, on_term)
    budget = {}
    while True:
        try:
            while True:
                handle(res_q.get(timeout=5))
        except Exception:
            pass
        if os.path.exists(f"{root}/STOP"):
            stopping = True
        pids = {w["proc"].pid for w in workers.values()}
        try:
            budget = GB.assess(pids, cpu_cap_workers=max_workers)
        except Exception as e:
            budget = {"error": repr(e), "granted_workers": 0, "over_cap": True}
        GB.publish(budget, {"holder": "selection supervisor", "supervisor_pid": os.getpid(),
                            "worker_pids": sorted(pids)})
        n = len(workers)
        grant_hist.append(budget.get("granted_workers", 0))
        del grant_hist[:-6]
        stable_grant = min(grant_hist) if len(grant_hist) >= 6 else 0  # hysteresis: grant held for ~30 s
        if budget.get("over_cap") and n and budget.get("daemon", {}).get("active"):
            L("DAEMON ACTIVE -> kill all workers", json.dumps({k: budget.get(k) for k in ("total_ours_mib", "free_mib", "daemon_mib")}))
            for wid in list(workers):
                workers[wid]["stop"].set()
                workers[wid]["proc"].kill()
            for wid in list(workers):
                workers[wid]["proc"].join(timeout=30)
                if workers[wid].get("unit"):
                    in_flight.discard(workers[wid]["unit"])
                workers.pop(wid)
        elif budget.get("over_cap") and n:
            # hard limit: free GPU-3 memory immediately (youngest first); unit is lost and re-run on resume
            wid = max(workers, key=lambda i: workers[i]["t_spawn"])
            L("OVER CAP -> kill worker", wid, json.dumps({k: budget.get(k) for k in ("total_ours_mib", "free_mib", "daemon_mib")}))
            workers[wid]["stop"].set()
            workers[wid]["proc"].kill()
            workers[wid]["proc"].join(timeout=30)
            if workers[wid].get("unit"):
                in_flight.discard(workers[wid]["unit"])
            workers.pop(wid)
        elif stopping or budget.get("granted_workers", 0) < n:
            want = 0 if stopping else budget.get("granted_workers", 0)
            active = sorted((i for i in workers if not workers[i]["stop"].is_set()), key=lambda i: -workers[i]["t_spawn"])
            for wid in active[:max(0, len([i for i in workers if not workers[i]['stop'].is_set()]) - want)]:
                workers[wid]["stop"].set()  # finish the current continuation, then exit
                workers[wid]["task_q"].put(None)  # wakes an idle worker
                L("yield worker", wid, "budget", budget.get("granted_workers"), budget.get("daemon", {}).get("reasons"))
        elif not stopping and stable_grant > n and time.time() - last_spawn > 20:
            if len(pending_units()) > len(in_flight):
                spawn()
                last_spawn = time.time()
        if time.time() - last_status > 120:
            st = scan(root, parents_idx)
            status_report(cfg, root, parents_idx, st, budget, len(workers), t_start, n_start,
                          {"stopping": stopping, "in_flight": [list(u) for u in in_flight]})
            last_status = time.time()
        if stopping and not workers:
            break
        if not workers and not pending_units() and not in_flight:
            L("all units complete")
            break
    st = scan(root, parents_idx)
    status_report(cfg, root, parents_idx, st, budget, 0, t_start, n_start, {"stopping": stopping, "finished": not pending_units()})
    req_q.put(None)
    srv.join(timeout=60)
    L("supervisor exit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_egg")
    ap.add_argument("--pool", default="sel_dev_parents")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--parents", default=None, help="e.g. 0-49")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--status", default=f"{STUDY}/selection/dev_branch_status.json")
    ap.add_argument("--root", default=None, help="override output root (verification runs)")
    a = ap.parse_args()
    assert a.pool in ("sel_dev_parents",) or os.environ.get("OV_ALLOW_TEST_PARENTS") == "1", \
        "test parents are not authorised yet"
    # CPU cap (protocol Amendment 3): <= 18 cores = workers (1 thread each) + inference server (2) + supervisor (1)
    CPU_CAP = 18
    assert a.max_workers + 3 <= CPU_CAP, f"max_workers {a.max_workers} exceeds the {CPU_CAP}-core cap"
    n_pool = json.load(open(f"{STUDY}/roster/reset_banks_v1.json"))["tasks"][a.task]["pools"][a.pool]["n"]
    if a.parents:
        lo, hi = (int(x) for x in a.parents.split("-"))
        parents_idx = list(range(lo, hi + 1))
    else:
        parents_idx = list(range(n_pool))
    cfg = {"task": a.task, "pool": a.pool, "train_seed": a.train_seed, "M": M, "R": R, "ckpt": CKPT,
           "branch_chunks": list(BRANCH_CHUNKS), "max_attempts": MAX_ATTEMPTS, "source_arms": list(SOURCE_ARMS),
           "variants": [list(v) for v in variants()], "render_gpu": 3, "infer_gpu": 0, "render_skip": True,
           "fast_contacts": True, "post_chunk_cache": True}
    key, blob = make_run_key(cfg)
    root = a.root or root_dir(a.task, a.pool, a.train_seed)
    os.makedirs(root, exist_ok=True)
    kf = f"{root}/RUN_KEY.json"
    if os.path.exists(kf):
        old = json.load(open(kf))
        if old["run_key"] != key:
            diff = {k: (old["blob"].get(k), blob.get(k)) for k in blob if old["blob"].get(k) != blob.get(k)}
            raise SystemExit(f"refusing to resume: run key changed {old['run_key']} -> {key}: {json.dumps(diff, default=str)[:3000]}")
    else:
        atomic_json({"run_key": key, "blob": blob, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, kf)
    cfg.update({"run_key": key, "root": root, "status_path": a.status})
    with open(f"{root}/supervisor.pid", "w") as f:
        f.write(str(os.getpid()))
    supervisor(cfg, parents_idx, a.max_workers)


if __name__ == "__main__":
    main()

