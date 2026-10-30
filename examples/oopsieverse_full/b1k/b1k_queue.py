"""Resumable Isaac job queue for the B1K diagnostic arm.

Jobs: JSON-lines file, each {"id", "cmd": [..], "done": path-that-exists-when-complete, "log": path}.
Each job runs under nohup-equivalent (own session), `nice -n 10`, pinned to an 8-core CPU set, with
OMNIGIBSON_GPU_ID=<g> and CUDA_VISIBLE_DEVICES unset (verified Isaac launch, 2026-10-22).

Guards (checked before each launch):
  * at most --maxp concurrent jobs of this queue; launches staggered by --stagger s (one Isaac startup
    transient (~830 MiB on every GPU) at a time);
  * the active GPU must have >= --need-gib free (8 GiB headroom + ~6 GiB Isaac/torch footprint);
  * per-GPU slot cap (--per-gpu);
  * STOP file in the status dir: no new launches (running jobs finish); KILL file: terminate ours.
Status: <status_dir>/queue_status.json rewritten every poll.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time


def gpu_free_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}


def globally_used_cpu_sets(cpu_sets):
    """CPU sets currently pinned by ANY of our b1k processes (other queue instances included)."""
    used = set()
    me = os.getuid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != me:
                continue
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore")
            if "oopsieverse_full/b1k/" not in cmd or "b1k_queue.py" in cmd:
                continue
            aff = sorted(os.sched_getaffinity(int(pid)))
        except Exception:
            continue
        for c in cpu_sets:
            a, b = map(int, c.split("-"))
            if aff == list(range(a, b + 1)):
                used.add(c)
    return used


def running_elsewhere(path_frag):
    """True if any of our processes (other queue instances included) has path_frag in its command line."""
    me = os.getuid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != me:
                continue
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore")
        except Exception:
            continue
        if path_frag in cmd and "b1k_queue.py" not in cmd:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--status-dir", required=True)
    ap.add_argument("--gpus", required=True, help="comma list of working Isaac GPUs")
    ap.add_argument("--cpu-sets", default="80-87,88-95,96-103,104-111,112-119,120-127")
    ap.add_argument("--maxp", type=int, default=5)
    ap.add_argument("--per-gpu", type=int, default=2)
    ap.add_argument("--need-gib", type=float, default=14.0)
    ap.add_argument("--stagger", type=float, default=150.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--cvd-map", default="", help="g:cvd;g:cvd  CUDA_VISIBLE_DEVICES per active GPU (GPU 0 hidden)")
    a = ap.parse_args()
    os.makedirs(a.status_dir, exist_ok=True)
    gpus = [int(x) for x in a.gpus.split(",")]
    assert 0 not in gpus
    cpu_sets = a.cpu_sets.split(",")
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    running = {}  # id -> (Popen, gpu, cpus, t0)
    attempts = {j["id"]: 0 for j in jobs}
    failed = {}
    last_launch = 0.0
    env0 = dict(os.environ)
    env0.pop("CUDA_VISIBLE_DEVICES", None)
    while True:
        # reap
        for jid, (p, g, c, t0) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[jid]
                j = next(x for x in jobs if x["id"] == jid)
                if not os.path.exists(j["done"]):
                    failed[jid] = rc
                    print(time.strftime("%T"), "FAIL", jid, "rc", rc, "gpu", g, flush=True)
                else:
                    print(time.strftime("%T"), "DONE", jid, "gpu", g, f"{time.time()-t0:.0f}s", flush=True)
        if os.path.exists(os.path.join(a.status_dir, "KILL")):
            for jid, (p, *_r) in running.items():
                os.killpg(p.pid, signal.SIGTERM)
            print("KILL file: terminated", list(running), flush=True)
            break
        pending = [j for j in jobs if not os.path.exists(j["done"]) and j["id"] not in running
                   and attempts[j["id"]] <= a.retries]
        stop = os.path.exists(os.path.join(a.status_dir, "STOP"))
        free = gpu_free_mib()
        st = {"time": time.strftime("%F %T"), "running": {k: {"gpu": v[1], "cpus": v[2], "age_s": int(time.time() - v[3])}
                                                          for k, v in running.items()},
              "pending": [j["id"] for j in pending], "failed_attempts": failed, "gpu_free_mib": free, "stop": stop,
              "n_done": sum(os.path.exists(j["done"]) for j in jobs), "n_jobs": len(jobs)}
        json.dump(st, open(os.path.join(a.status_dir, "queue_status.json.tmp"), "w"), indent=1)
        os.replace(os.path.join(a.status_dir, "queue_status.json.tmp"), os.path.join(a.status_dir, "queue_status.json"))
        if not pending and not running:
            print(time.strftime("%T"), "QUEUE_COMPLETE", "done", st["n_done"], "/", len(jobs), flush=True)
            break
        if pending and not stop and len(running) < a.maxp and time.time() - last_launch > a.stagger:
            used_cpus = {v[2] for v in running.values()} | globally_used_cpu_sets(cpu_sets)
            cpus = next((c for c in cpu_sets if c not in used_cpus), None)
            cnt = {g: sum(1 for v in running.values() if v[1] == g) for g in gpus}
            cand = [g for g in gpus if cnt[g] < a.per_gpu and free.get(g, 0) >= a.need_gib * 1024]
            cvdmap = dict(x.split(":") for x in a.cvd_map.split(";") if x)
            cand = [g for g in cand if min(free[int(h)] for h in cvdmap.get(str(g), ",".join(map(str, free))).split(",")) >= 1024]
            if False:  # Isaac opens a context on EVERY GPU at startup (~830 MiB peak)
                cand = []
            cand.sort(key=lambda g: (cnt[g], -free[g]))
            pending = [j for j in pending if not running_elsewhere(j["done"])]  # done path is unique and in the cmd
            if cpus and cand and pending:
                g = cand[0]
                j = pending[0]
                attempts[j["id"]] += 1
                env = dict(env0, OMNIGIBSON_GPU_ID=str(g), B1K_JOB_ID=j["id"])
                if str(g) in cvdmap:
                    env["CUDA_VISIBLE_DEVICES"] = cvdmap[str(g)]
                env.update(j.get("env", {}))
                os.makedirs(os.path.dirname(j["log"]), exist_ok=True)
                lf = open(j["log"], "a")
                lf.write(f"\n==== {time.strftime('%F %T')} attempt {attempts[j['id']]} gpu {g} cpus {cpus}\n")
                lf.flush()
                p = subprocess.Popen(["nice", "-n", "10", "taskset", "-c", cpus] + j["cmd"], stdout=lf,
                                     stderr=subprocess.STDOUT, env=env, cwd=j.get("cwd", "/tmp"),
                                     start_new_session=True)
                running[j["id"]] = (p, g, cpus, time.time())
                last_launch = time.time()
                print(time.strftime("%T"), "START", j["id"], "gpu", g, "cpus", cpus, "attempt", attempts[j["id"]], flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
