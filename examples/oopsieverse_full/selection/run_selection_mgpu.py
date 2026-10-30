"""Multi-GPU supervisor for the selection branching study (Amendment 6/7) — transport/scheduling only.

Identical record-producing path to run_selection.py: the same branch_worker.worker_main (wrapped by the same
run_selection.safe_worker_main SIGUSR1 path), the same unit ordering and assignment (scan/pending_units/assign copied
verbatim), the same inference_server on GPU 0 with fixed-shape batches of 16. Differences:
  * each worker renders on one GPU chosen at spawn time from the GPUs that passed the exact EGL pixel-parity check
    (gpu_budget.parity_passed_gpus(); never 0 or 2): the worker gets a copy of cfg with render_gpu=<g>. The run-key
    cfg records render_gpu="multi:parity-passed" (cross-GPU bit-identity verified with replay_records_gpu.py);
  * budget = gpu_budget (per-GPU >= 8 GB free, strict 30-min minimum, ledger shared by all seed supervisors, global
    worker cap); breach -> evict youngest worker on that GPU via SIGUSR1; over-subscription -> graceful yield;
  * --servers N inference servers (all on --infer-gpu), worker wid -> server wid % N (bit-identical: fixed batch);
  * up to 64 worker slots.

  OV_ALLOW_TEST_PARENTS=1 nice -n 10 python run_selection_mgpu.py --task pick_egg --pool sel_test_parents --train-seed 0
Stop: touch <root>/STOP (workers finish their current continuation).
"""
import argparse
import json
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from branch_worker import *  # noqa: E402,F401,F403
from run_selection import scan, status_report, safe_worker_main  # noqa: E402

SLOTS = 64


def supervisor(cfg, parents_idx, my_cap, total_cap, n_servers, infer_gpu):
    import multiprocessing as mp
    import gpu_budget as GB
    import harness as H
    root = cfg["root"]
    ctx = mp.get_context("spawn")
    os.makedirs(f"{root}/states", exist_ok=True)
    log = open(f"{root}/supervisor.log", "a")

    def L(*a):
        log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a) + "\n")
        log.flush()
    ledger = GB.Ledger(os.getpid(), label=f"{cfg['task']}/{cfg['pool']}/seed{cfg['train_seed']}")
    srv = {}

    def start_servers():
        srv["req_qs"] = [ctx.Queue() for _ in range(n_servers)]
        srv["resp"] = [ctx.Queue() for _ in range(SLOTS)]
        srv["procs"] = []
        os.environ["CUDA_VISIBLE_DEVICES"] = str(infer_gpu)
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        for s in range(n_servers):
            p = ctx.Process(target=H.inference_server, args=(
                run_dirs(cfg["task"], cfg["train_seed"]), CKPT, "cuda:0", srv["req_qs"][s], srv["resp"],
                f"{root}/_server_stats_{s}.json", 16))
            p.start()
            srv["procs"].append(p)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        L("inference servers started on GPU", infer_gpu, "pids", [p.pid for p in srv["procs"]])

    def stop_servers(kill=False):
        for p in srv["procs"]:
            if kill:
                p.kill()
        if not kill:
            for q in srv["req_qs"]:
                q.put(None)
        for p in srv["procs"]:
            p.join(timeout=60)
            if p.is_alive():
                p.kill()
    start_servers()
    server_tainted = False
    res_q = ctx.Queue()
    workers = {}
    in_flight = set()
    t_start = time.time()
    st = scan(root, parents_idx)
    n_start = sum(v["n"] for v in st["groups"].values())
    last_status, last_spawn, last_gpus = 0, 0, 0
    render_gpus = []
    stopping = False

    # ---- verbatim from run_selection.supervisor ----
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

        def cls(u):
            return 0 if u[0] == "collect" else (1 if u[3] == "primary" else 2)
        top = min(cls(u) for u in cand)
        cand = [u for u in cand if cls(u) == top]
        same = [u for u in cand if u[0] == "group" and u[1] == w.get("parent")]
        u = same[0] if same else cand[0]
        in_flight.add(u)
        w["unit"] = u
        w["parent"] = u[1]
        w["task_q"].put(u)
        return True

    def handle(m):
        nonlocal st
        wid = m["wid"]
        w = workers.get(wid)
        if m["event"] == "ready":
            L("worker", wid, "ready gpu", w["gpu"] if w else None, round(m["env_build_s"], 1))
            if w and not w["stop"].is_set() and not stopping:
                assign(wid) or w["task_q"].put(None)
        elif m["event"] in ("done", "error"):
            u = tuple(m["unit"])
            in_flight.discard(u)
            if m["event"] == "error":
                L("unit error", u, m["tb"][-1500:])
            st = scan(root, parents_idx)
            if w and not w["stop"].is_set() and not stopping:
                assign(wid) or w["task_q"].put(None)
            elif w:
                w["task_q"].put(None)
        elif m["event"] == "crash":
            L("worker crash", wid, m["tb"][-2000:])
        elif m["event"] == "exit":
            if w and w["proc"].is_alive():
                w["proc"].join(timeout=3)
            if w and w["proc"].is_alive():
                w = None
            if w:
                if w.get("unit"):
                    in_flight.discard(w["unit"])
                w["proc"].join(timeout=30)
                workers.pop(wid, None)
                L("worker exit", wid)
    # ---- end verbatim ----

    def spawn(g):
        wid = [i for i in range(SLOTS) if i not in workers][0]
        tq = ctx.Queue()
        ev = ctx.Event()
        wcfg = dict(cfg, render_gpu=g)
        p = ctx.Process(target=safe_worker_main,
                        args=(wid, wcfg, srv["req_qs"][wid % n_servers], srv["resp"][wid], tq, res_q, ev))
        p.start()
        workers[wid] = {"proc": p, "task_q": tq, "stop": ev, "unit": None, "parent": None, "t_spawn": time.time(), "gpu": g}
        L("spawn worker", wid, "pid", p.pid, "render gpu", g, "server", wid % n_servers)

    def evict(wids):
        for wid in wids:
            workers[wid]["stop"].set()
            try:
                os.kill(workers[wid]["proc"].pid, signal.SIGUSR1)
            except OSError:
                pass
        t0 = time.time()
        while time.time() - t0 < 5 and any(workers[w]["proc"].is_alive() for w in wids):
            time.sleep(0.1)
        hard = False
        for wid in wids:
            if workers[wid]["proc"].is_alive():
                workers[wid]["proc"].kill()
                hard = True
            workers[wid]["proc"].join(timeout=30)
            if workers[wid].get("unit"):
                in_flight.discard(workers[wid]["unit"])
            L("evicted worker", wid, "gpu", workers[wid]["gpu"], "hard" if hard else "safe")
            workers.pop(wid)
        return hard

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
        for wid in list(workers):
            if not workers[wid]["proc"].is_alive():
                code = workers[wid]["proc"].exitcode
                L("worker gone", wid, "exitcode", code)
                if workers[wid].get("unit"):
                    in_flight.discard(workers[wid]["unit"])
                workers.pop(wid)
                if code not in (0, 17):
                    server_tainted = True
        if os.path.exists(f"{root}/EVICT_TEST") and workers:
            os.remove(f"{root}/EVICT_TEST")
            wid = max(workers, key=lambda i: workers[i]["t_spawn"])
            t_ev = time.time()
            hard = evict([wid])
            server_tainted |= hard
            L("EVICT_TEST", wid, "hard" if hard else "safe", f"{time.time() - t_ev:.2f}s")
        if server_tainted:
            if workers:
                evict(list(workers))
            stop_servers(kill=True)
            start_servers()
            server_tainted = False
        if time.time() - last_gpus > 300:
            new = GB.parity_passed_gpus()
            if new != render_gpus:
                L("render GPUs (exact parity passed):", new)
            render_gpus = new
            last_gpus = time.time()
        mine = {w["proc"].pid: w["gpu"] for w in workers.values()}
        active = [i for i in workers if not workers[i]["stop"].is_set()]
        want = (not stopping and time.time() - last_spawn > 10 and len(pending_units()) > len(in_flight))
        try:
            budget = GB.assess(ledger, mine, render_gpus, total_cap, my_cap, want_spawn=want)
        except Exception as e:
            budget = {"error": repr(e), "spawn_gpu": None, "over": {}, "yield_gpu": None}
            L("budget error", repr(e))
        if budget.get("over"):
            for g in budget["over"]:
                on_g = [i for i in workers if workers[i]["gpu"] == int(g)]
                if on_g:
                    wid = max(on_g, key=lambda i: workers[i]["t_spawn"])
                    L("HEADROOM BREACH gpu", g, "-> evict worker", wid, json.dumps(budget["gpus"][g]))
                    server_tainted |= evict([wid])
        elif budget.get("yield_gpu") is not None or stopping:
            g = budget.get("yield_gpu")
            cand = [i for i in active if stopping or workers[i]["gpu"] == g]
            for wid in sorted(cand, key=lambda i: -workers[i]["t_spawn"])[: (len(cand) if stopping else 1)]:
                workers[wid]["stop"].set()
                workers[wid]["task_q"].put(None)
                L("yield worker", wid, "gpu", workers[wid]["gpu"], "stopping" if stopping else "over-subscribed")
        elif budget.get("spawn_gpu") is not None:
            spawn(budget["spawn_gpu"])
            last_spawn = time.time()
        if time.time() - last_status > 120:
            st = scan(root, parents_idx)
            status_report(cfg, root, parents_idx, st, {}, len(workers), t_start, n_start,
                          {"stopping": stopping, "in_flight": [list(u) for u in in_flight],
                           "workers_by_gpu": {str(g): sum(1 for w in workers.values() if w["gpu"] == g) for g in render_gpus},
                           "gpu_budget": {k: budget.get(k) for k in ("time", "n_workers_all", "n_workers_mine", "gpus", "render_gpus")},
                           "servers": n_servers})
            last_status = time.time()
        if stopping and not workers:
            break
        if not workers and not pending_units() and not in_flight:
            L("all units complete")
            break
    st = scan(root, parents_idx)
    status_report(cfg, root, parents_idx, st, {}, 0, t_start, n_start, {"stopping": stopping, "finished": not pending_units()})
    ledger.remove()
    stop_servers()
    L("supervisor exit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_egg")
    ap.add_argument("--pool", default="sel_test_parents")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--parents", default=None)
    ap.add_argument("--max-workers", type=int, default=18, help="this supervisor's worker cap")
    ap.add_argument("--total-workers", type=int, default=52, help="cap over all selection supervisors (ledger)")
    ap.add_argument("--servers", type=int, default=2)
    ap.add_argument("--infer-gpu", type=int, default=0)
    ap.add_argument("--status", default=None)
    a = ap.parse_args()
    assert a.pool in ("sel_dev_parents",) or os.environ.get("OV_ALLOW_TEST_PARENTS") == "1", "test parents not authorised"
    # CPU (user 2026-10-22: ~64 cores total): workers 1 thread each + per seed ~1 supervisor + servers
    assert a.total_workers + 3 * (1 + a.servers) <= 64
    n_pool = json.load(open(f"{STUDY}/roster/reset_banks_v1.json"))["tasks"][a.task]["pools"][a.pool]["n"]
    if a.parents:
        lo, hi = (int(x) for x in a.parents.split("-"))
        parents_idx = list(range(lo, hi + 1))
    else:
        parents_idx = list(range(n_pool))
    cfg = {"task": a.task, "pool": a.pool, "train_seed": a.train_seed, "M": M, "R": R, "ckpt": CKPT,
           "branch_chunks": list(BRANCH_CHUNKS), "max_attempts": MAX_ATTEMPTS, "source_arms": list(SOURCE_ARMS),
           "variants": [list(v) for v in variants()], "render_gpu": "multi:parity-passed", "infer_gpu": a.infer_gpu,
           "render_skip": True, "fast_contacts": True, "post_chunk_cache": True}
    key, blob = make_run_key(cfg)
    root = root_dir(a.task, a.pool, a.train_seed)
    os.makedirs(root, exist_ok=True)
    kf = f"{root}/RUN_KEY.json"
    if os.path.exists(kf):
        old = json.load(open(kf))
        if old["run_key"] != key:
            diff = {k: (old["blob"].get(k), blob.get(k)) for k in blob if old["blob"].get(k) != blob.get(k)}
            raise SystemExit(f"refusing to resume: run key changed {old['run_key']} -> {key}: {json.dumps(diff, default=str)[:3000]}")
    else:
        atomic_json({"run_key": key, "blob": blob, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, kf)
    status = a.status or f"{STUDY}/selection/test_branch_status_seed{a.train_seed}.json"
    cfg.update({"run_key": key, "root": root, "status_path": status})
    with open(f"{root}/supervisor.pid", "w") as f:
        f.write(str(os.getpid()))
    supervisor(cfg, parents_idx, a.max_workers, a.total_workers, a.servers, a.infer_gpu)


if __name__ == "__main__":
    main()
