"""Resumable launcher for the Phase C training queue (no evaluation; test resets are never touched).

  nohup python queue_runner.py --gpus 5,6,7 --jobs-per-gpu J --workers W > runs/matrix/_queue/runner.log 2>&1 &

* Queue (priority order): pick_egg primary arms {amix, asafe, imean, ipath, ipathg, auxpathg} x seeds {0,1,2}
  (seed-major), then wipe_countertop primary arms, then controls {ssmix, cmean} x seeds x {pick_egg, wipe}.
* A job is complete when <run>/DONE.json holds the current run key (train_matrix.py checks; ALREADY_DONE exit 0).
  A job is running if its <run>/LOCK is flocked (also by a previous runner instance) -> never double-launched.
* Failures: train_matrix.py ledgers the failure; the runner retries up to --max-attempts in total; each retry resumes
  from the latest step_*.pth (ledger note resume_from=...). A run-key mismatch (exit 1, REFUSE) is not retried.
* Slots: --jobs-per-gpu processes per GPU; a launch also requires >= --min-free-gb free memory on that GPU.
* Graceful stop: touch <runs>/matrix/_queue/STOP (running jobs finish; nothing new starts). Restarting the runner
  resumes the queue. Status: outputs/oopsieverse_full_study/matrix/queue_status.json (rewritten every poll).
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_matrix import RUNS, STUDY, ledger_append  # noqa: E402

PY = "/edrive1/lizad/oopsieverse_full_study/env/oopsie/bin/python"
QDIR = os.path.join(RUNS, "_queue")
STATUS = os.environ.get("QUEUE_STATUS", os.path.join(STUDY, "matrix/queue_status.json"))  # Amendment 3: 2nd runner has its own
TRAIN_GPUS = {0, 1, 4}  # Amendment 1 (GPU 0) + Amendment 3 (GPUs 1, 4: training only); never any other GPU
PRIMARY = ["amix", "asafe", "imean", "ipath", "ipathg", "auxpathg"]
CONTROLS = ["ssmix", "cmean"]
SEEDS = [0, 1, 2]
STEPS = 30000


def queue():
    q = []
    for task in ("pick_egg", "wipe_countertop"):
        for s in SEEDS:
            for a in PRIMARY:
                q.append((a, task, s))
    for task in ("pick_egg", "wipe_countertop"):
        for s in SEEDS:
            for a in CONTROLS:
                q.append((a, task, s))
    return q


def rid(a, t, s):
    return f"{a}_{t}_v1_s{s}"


def is_done(r):
    return os.path.exists(os.path.join(RUNS, r, "DONE.json"))


def is_locked(r):
    p = os.path.join(RUNS, r, "LOCK")
    if not os.path.exists(p):
        return False
    with open(p, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True


def gpu_free_gb(g):
    try:
        out = subprocess.check_output(["nvidia-smi", "-i", str(g), "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        return float(out.decode().strip()) / 1024
    except Exception:
        return 0.0


def progress(r):
    """(last logged step, wall_s at that step, updates/s over the last ~2000 steps)."""
    p = os.path.join(RUNS, r, "metrics.jsonl")
    if not os.path.exists(p):
        return None
    pts = []
    with open(p) as f:
        for line in f:
            if '"train/loss"' in line:
                d = json.loads(line)
                pts.append((d["step"], d["wall_s"]))
    if not pts:
        return None
    s1, w1 = pts[-1]
    s0, w0 = next(((s, w) for s, w in reversed(pts) if s <= s1 - 2000), pts[0])
    rate = (s1 - s0) / (w1 - w0) if w1 > w0 else None
    return {"step": s1, "wall_s": w1, "updates_per_s": rate}


def prune_ckpts(r):
    """Keep step_{10000,20000,30000} and final.pth; of the other step_* (resume points) keep only the newest, and none
    once the run is DONE."""
    d = os.path.join(RUNS, r, "ckpt")
    if not os.path.isdir(d) or os.path.islink(d):
        return
    steps = sorted(int(f[5:-4]) for f in os.listdir(d) if f.startswith("step_") and f.endswith(".pth"))
    done = is_done(r)
    other = [x for x in steps if x % 10000 != 0 or x == 0]
    newest = max(steps) if steps else None
    if newest is not None and time.time() - os.path.getmtime(os.path.join(d, f"step_{newest}.pth")) < 120:
        return  # newest may still be being written; prune later
    for x in other:
        if done or x != newest:
            try:
                os.remove(os.path.join(d, f"step_{x}.pth"))
            except FileNotFoundError:
                pass


def env():
    e = dict(os.environ)
    o = "/edrive1/lizad/oopsieverse_full_study"
    e.update({"TMPDIR": f"{o}/cache/tmp", "HF_HOME": f"{o}/cache/hf", "XDG_CACHE_HOME": f"{o}/cache/xdg",
              "TORCH_HOME": f"{o}/cache/torch", "MPLCONFIGDIR": f"{o}/cache/mpl", "NUMBA_CACHE_DIR": f"{o}/cache/numba",
              "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "PYNPUT_BACKEND": "dummy",
              "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    return e


def write_status(st):
    tmp = STATUS + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, STATUS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="0:3", help="gpu:max_jobs (Amendment 1: ONLY GPU 0 is authorised)")
    ap.add_argument("--max-jobs", type=int, default=3, help="total concurrent jobs (CPU budget ~20 cores shared with B1K)")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--min-free-gb", type=float, default=25.0,
                    help="coordinator rule: >= 25 GB free on the GPU before launching (job peak ~9 GB with --act-ckpt)")
    ap.add_argument("--job-peak-gb", type=float, default=9.0, help="expected nvidia-smi footprint of one job (measured)")
    ap.add_argument("--headroom-gb", type=float, default=8.0, help="keep >= this free per GPU at all times")
    ap.add_argument("--settle-s", type=float, default=300.0, help="a new job counts at full peak for this long")
    ap.add_argument("--poll-s", type=float, default=30.0)
    a = ap.parse_args()
    os.makedirs(QDIR, exist_ok=True)
    cap = {int(x.split(":")[0]): int(x.split(":")[1]) for x in a.slots.split(",")}
    assert set(cap) <= TRAIN_GPUS, "Amendments 1+3: only GPUs 0, 1, 4 are authorised for training"
    assert a.max_jobs * (1 + a.workers) <= 20, "CPU budget: ~20 cores (48-core cap shared with B1K Isaac processes)"
    gpus = sorted(cap)
    q = queue()
    procs = {}       # run_id -> (Popen, gpu, attempt, t0)
    attempts = {}    # run_id -> attempts launched by this runner
    refused = {}
    infra = {}       # run_id -> infrastructure retries (OOM / headroom preemption)
    preempted = set()
    seen_ckpt = set()
    t_start = time.time()
    while True:
        # reap
        for r, (p, g, att, t0) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            del procs[r]
            if rc != 0 and not is_done(r):
                log = open(os.path.join(QDIR, f"{r}.a{att}.log")).read()[-3000:]
                if "REFUSE" in log:
                    refused[r] = "run-key mismatch (refused, not retried)"
                elif rc == 86 or "out of memory" in log.lower() or r in preempted:
                    # infrastructure: shared-GPU memory. Requeue without consuming an attempt (ledgered by trainer).
                    attempts[r] -= 1
                    infra[r] = infra.get(r, 0) + 1
                    preempted.discard(r)
        # checkpoint progress -> ledger
        for (arm, task, s) in q:
            r = rid(arm, task, s)
            for step in (10000, 20000, 30000):
                ck = os.path.join(RUNS, r, "ckpt", f"step_{step}.pth")
                if (r, step) not in seen_ckpt and os.path.exists(ck):
                    seen_ckpt.add((r, step))
                    if not os.path.exists(os.path.join(QDIR, f"ckpt_{r}_{step}.mark")):
                        open(os.path.join(QDIR, f"ckpt_{r}_{step}.mark"), "w").write(str(time.time()))
                        ledger_append({"run_id": r, "event": f"checkpoint", "arm": arm, "task": task, "seed": s,
                                       "label": "final", "num_steps": STEPS, "step": step, "run_dir": os.path.join(RUNS, r),
                                       "note": "checkpoint written (queue_runner)"})
        for (arm, task, s) in q:
            prune_ckpts(rid(arm, task, s))
        stop = os.path.exists(os.path.join(QDIR, "STOP"))
        # headroom rule: if a GPU drops below headroom, stop OUR newest job on it (never anyone else's); it resumes later
        for g in gpus:
            mine = sorted([(v[3], r) for r, v in procs.items() if v[1] == g and r not in preempted])
            if mine and gpu_free_gb(g) < a.headroom_gb:
                t_l, r = mine[-1]
                procs[r][0].send_signal(15)
                preempted.add(r)
                print("PREEMPT", r, "gpu", g, "free", round(gpu_free_gb(g), 1), time.strftime("%c"), flush=True)
        # launch
        running_per_gpu = {g: sum(1 for v in procs.values() if v[1] == g) for g in gpus}
        pending = [x for x in q if not is_done(rid(*x)) and rid(*x) not in procs and not is_locked(rid(*x))
                   and rid(*x) not in refused and attempts.get(rid(*x), 0) < a.max_attempts]
        if not stop:
            for (arm, task, s) in pending:
                if len(procs) >= a.max_jobs:
                    break
                def eff_free(x):
                    # jobs launched < settle_s ago may not have reached their peak yet: count them at full peak
                    young = sum(1 for v in procs.values() if v[1] == x and time.time() - v[3] < a.settle_s)
                    return gpu_free_gb(x) - young * a.job_peak_gb
                free = [g for g in gpus if running_per_gpu[g] < cap[g] and eff_free(g) >= a.min_free_gb
                        and eff_free(g) - a.job_peak_gb >= a.headroom_gb]
                if not free:
                    break  # queue (do not fail): wait for memory
                g = max(free, key=lambda x: (eff_free(x), -running_per_gpu[x]))
                r = rid(arm, task, s)
                att = attempts.get(r, 0) + 1
                attempts[r] = att
                logp = os.path.join(QDIR, f"{r}.a{att}.log")
                e = env()
                assert g in TRAIN_GPUS
                e["CUDA_VISIBLE_DEVICES"] = str(g)
                e["MUJOCO_EGL_DEVICE_ID"] = str(g)  # training never renders
                e["EGL_DEVICE_ID"] = str(g)
                e["MATRIX_LOG"] = logp
                e["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                # save_every 2500 = resume points; non-10k checkpoints are pruned (prune_ckpts)
                cmd = [PY, os.path.join(HERE, "train_matrix.py"), "--arm", arm, "--task", task, "--seed", str(s),
                       "--num-steps", str(STEPS), "--save-every", "2500", "--label", "final", "--workers", str(a.workers),
                       "--act-ckpt"]
                p = subprocess.Popen(cmd, stdout=open(logp, "w"), stderr=subprocess.STDOUT, env=e, cwd=HERE,
                                     start_new_session=True)
                procs[r] = (p, g, att, time.time())
                running_per_gpu[g] += 1
                time.sleep(20)  # stagger dataset builds
        # status
        done = [rid(*x) for x in q if is_done(rid(*x))]
        running = {}
        for x in q:
            r = rid(*x)
            if r in procs or is_locked(r):
                pr = progress(r)
                running[r] = {"gpu": procs[r][1] if r in procs else "external", "progress": pr}
        rates = [v["progress"]["updates_per_s"] for v in running.values() if v["progress"] and v["progress"]["updates_per_s"]]
        per_job = sum(rates) / len(rates) if rates else None
        remaining_updates = sum(STEPS - (running[r]["progress"]["step"] if running.get(r, {}).get("progress") else 0)
                                for r in [rid(*x) for x in q if not is_done(rid(*x))])
        slots = max(1, min(a.max_jobs, len(running) or 1))
        eta_h = (remaining_updates / (per_job * slots) / 3600) if per_job else None
        failed = {r: attempts[r] for r in attempts if attempts[r] >= a.max_attempts and not is_done(r) and r not in procs}
        write_status({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "runner_pid": os.getpid(), "stop_flag": stop,
                      "config": vars(a), "n_total": len(q), "n_done": len(done), "n_running": len(running),
                      "n_pending": len([x for x in q if not is_done(rid(*x)) and rid(*x) not in running]),
                      "done": done, "running": running, "refused": refused, "exhausted_retries": failed,
                      "infra_retries": infra, "gpu_free_gb": {g: round(gpu_free_gb(g), 1) for g in gpus},
                      "effective_concurrency": len(procs),
                      "mean_updates_per_s_per_job": per_job, "eta_hours_at_current_rate": eta_h,
                      "runner_uptime_h": (time.time() - t_start) / 3600})
        if (stop or not [x for x in q if not is_done(rid(*x)) and rid(*x) not in refused
                         and attempts.get(rid(*x), 0) < a.max_attempts]) and not procs:
            print("QUEUE_IDLE", time.strftime("%c"), "done", len(done), "of", len(q), flush=True)
            break
        time.sleep(a.poll_s)


if __name__ == "__main__":
    main()
