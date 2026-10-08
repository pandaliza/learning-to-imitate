"""GPU-3 render budget shared with the Amendment-2 evaluation daemon (eval_runner.py, which has priority).

Rules (user, 2026-10-08):
  * our TOTAL memory on GPU 3 (eval daemon + selection workers) <= 20 GB (Amendment 4; was 10 GB);
  * >= 8 GB free on GPU 3 at all times;
  * the eval daemon has priority: never starve, pause or restart it.

The daemon cannot read our files (it is already running and is not modified), so the selection harness yields to it:
  * daemon ACTIVE   = a rollout_eval_matrix.py process exists under the daemon's process tree;
  * daemon IMMINENT = a trained run (DONE.json) is not yet fully evaluated (the daemon picks it up within its 300 s
                      poll), or a running training job is predicted to finish within LEAD_S;
  * while ACTIVE or IMMINENT we reserve the daemon's full footprint (DAEMON_WORKERS x WORKER_MIB) even before it
    allocates, so the daemon never finds our memory in its way;
  * granted selection workers = floor((CAP - reserve - margin) / WORKER_MIB), further limited by free memory
    (free - WORKER_MIB >= 8 GB headroom + margin) and by the CPU cap;
  * over-cap (measured total > CAP or free < HEADROOM) => the supervisor kills selection workers immediately
    (their unit is lost and re-run on resume; records are atomic).
Every decision is written to runs/selection/_gpu3/budget.json (+ budget_log.jsonl) so that the daemon or a human
can see what we hold.
"""

import glob
import json
import os
import subprocess
import time

CAP_MIB = 10 * 1024  # Amendment 4 authorizes 20 GB, but the coexistence change below is not applied yet
HEADROOM_MIB = 8 * 1024
WORKER_MIB = 1150          # measured 1,084 MiB per EGL worker (matrix/arm_verification.md §8) + margin
DAEMON_WORKERS = 9
MARGIN_MIB = 256
PEAK_WINDOW_S = 30 * 60     # strict headroom: plan against the lowest free memory (excluding our workers) seen recently
HEADROOM_POLICY = os.environ.get("OV_GPU3_HEADROOM_POLICY", "strict")  # strict | reactive
_FREE_HIST = []             # (t, free_mib + selection-worker mib), seeded from budget_log.jsonl
LEAD_S = 8 * 60            # stop granting when a training job is predicted to finish within 8 min
RENDER_GPU = 3
DAEMON_PID = 1017799
OVS = "/edrive1/lizad/oopsieverse_full_study"
STUDY = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study"
BUDGET_DIR = f"{OVS}/runs/selection/_gpu3"


def _children(pid):
    out, todo = set(), [pid]
    while todo:
        p = todo.pop()
        try:
            kids = open(f"/proc/{p}/task/{p}/children").read().split()
        except OSError:
            kids = []
        for k in kids:
            k = int(k)
            if k not in out:
                out.add(k)
                todo.append(k)
    return out


def _cmdline(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def _uid(pid):
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return -1


def gpu3_table():
    """{pid: MiB} of every process on GPU 3 (compute and graphics) + free MiB."""
    free = float(subprocess.check_output(["nvidia-smi", "-i", str(RENDER_GPU), "--query-gpu=memory.free",
                                          "--format=csv,noheader,nounits"], timeout=30).decode().strip())
    procs = {}
    for line in subprocess.check_output(["nvidia-smi"], timeout=30).decode().splitlines():
        parts = line.strip("| ").split()
        if len(parts) >= 7 and parts[0] == str(RENDER_GPU) and parts[3].isdigit() and parts[-1].endswith("MiB"):
            procs[int(parts[3])] = procs.get(int(parts[3]), 0.0) + float(parts[-1][:-3])
    return procs, free


def _progress(path):
    """Same estimate as matrix/queue_runner.progress: last step and updates/s over the last ~2000 steps."""
    pts = []
    with open(path) as f:
        for line in f:
            if '"train/loss"' in line:
                d = json.loads(line)
                pts.append((d["step"], d["wall_s"]))
    if not pts:
        return None
    s1, w1 = pts[-1]
    s0, w0 = next(((s, w) for s, w in reversed(pts) if s <= s1 - 2000), pts[0])
    return {"step": s1, "updates_per_s": (s1 - s0) / (w1 - w0) if w1 > w0 else None}


def daemon_state():
    alive = os.path.exists(f"/proc/{DAEMON_PID}") and "eval_runner.py" in _cmdline(DAEMON_PID)
    tree = _children(DAEMON_PID) if alive else set()
    active = any("rollout_eval_matrix.py" in _cmdline(p) for p in tree)
    reasons = []
    if active:
        reasons.append("rollout_eval_matrix running")
    imminent = False
    if alive:
        try:
            ev = json.load(open(f"{STUDY}/matrix/eval_status.json")).get("evaluated", {})
        except Exception:
            ev = {}
        for d in glob.glob(f"{OVS}/runs/matrix/*/DONE.json"):
            r = os.path.basename(os.path.dirname(d))
            if ev.get(r, {}).get("n", 0) < 600 and not ev.get(r, {}).get("note"):
                imminent = True
                reasons.append(f"{r} trained, not yet evaluated")
        # every training runner (GPU 0 queue, Amendment-3 GPU 1/4 queue, ...) plus any run whose metrics are
        # still being written: predict its finish time from the logged step rate
        running = {}
        for qf in glob.glob(f"{STUDY}/matrix/queue_status*.json"):
            try:
                for r, info in json.load(open(qf)).get("running", {}).items():
                    if info.get("progress"):
                        running[r] = info["progress"]
            except Exception as e:
                reasons.append(f"{os.path.basename(qf)} unreadable: {e}")
        for mf in glob.glob(f"{OVS}/runs/matrix/*/metrics.jsonl"):
            r = os.path.basename(os.path.dirname(mf))
            if r in running or os.path.exists(os.path.join(os.path.dirname(mf), "DONE.json")):
                continue
            if time.time() - os.path.getmtime(mf) < 900:
                pr = _progress(mf)
                if pr:
                    running[r] = pr
        for r, pr in running.items():
            step, rate = pr.get("step"), pr.get("updates_per_s")
            if step is not None and rate:
                eta = (30000 - step) / max(rate, 1e-3)
                if eta < LEAD_S:
                    imminent = True
                    reasons.append(f"{r} finishes in ~{eta / 60:.0f} min")
    return {"alive": alive, "active": active, "imminent": imminent, "tree": sorted(tree), "reasons": reasons}


def assess(our_pids, cpu_cap_workers=18):
    """Budget decision for the selection supervisor. our_pids = selection worker pids holding EGL contexts."""
    procs, free = gpu3_table()
    ds = daemon_state()
    daemon_mib = sum(v for p, v in procs.items() if p in set(ds["tree"]) | {DAEMON_PID})
    ours_mib = sum(v for p, v in procs.items() if p in our_pids)
    # any other process of this user on GPU 3 (verification tests etc.) also counts against the 10 GB cap
    uid = os.getuid()
    daemon_set = set(ds["tree"]) | {DAEMON_PID}
    other_mib = sum(v for p, v in procs.items()
                    if p not in our_pids and p not in daemon_set and _uid(p) == uid)
    reserve = max(daemon_mib, DAEMON_WORKERS * WORKER_MIB) if (ds["active"] or ds["imminent"]) else daemon_mib
    by_cap = int((CAP_MIB - reserve - other_mib - MARGIN_MIB) // WORKER_MIB)
    n_now = len([p for p in our_pids if p in procs])
    # headroom: free memory that would remain with our selection workers removed, now and (strict) at its
    # lowest over the last PEAK_WINDOW_S (other users' transient spikes, e.g. a 1-min ramp to ~0.5 GB free)
    base_now = free + ours_mib
    now_t = time.time()
    if not _FREE_HIST:
        try:
            for line in open(f"{BUDGET_DIR}/budget_log.jsonl").readlines()[-2000:]:
                x = json.loads(line)
                if "free_mib" in x and "ours_mib" in x and now_t - x["t"] < PEAK_WINDOW_S:
                    _FREE_HIST.append((x["t"], x["free_mib"] + x["ours_mib"]))
        except Exception:
            pass
    _FREE_HIST.append((now_t, base_now))
    while _FREE_HIST and now_t - _FREE_HIST[0][0] > PEAK_WINDOW_S:
        _FREE_HIST.pop(0)
    base_plan = min(b for _, b in _FREE_HIST) if HEADROOM_POLICY == "strict" else base_now
    by_free = int((base_plan - HEADROOM_MIB - MARGIN_MIB) // WORKER_MIB)
    granted = max(0, min(by_cap, by_free, cpu_cap_workers))
    total = daemon_mib + ours_mib + other_mib
    # the daemon became active while we hold workers: free GPU 3 at once, before its workers allocate (~10 s)
    over = total > CAP_MIB or free < HEADROOM_MIB or (ds["active"] and len(our_pids) > 0)
    d = {"t": time.time(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "free_mib": free, "daemon_mib": daemon_mib,
         "ours_mib": ours_mib, "other_user_procs_mib": other_mib, "total_ours_mib": total, "reserve_daemon_mib": reserve, "granted_workers": granted,
         "n_workers_holding": n_now, "over_cap": over, "headroom_policy": HEADROOM_POLICY,
         "free_without_selection_now_mib": base_now, "free_without_selection_planned_mib": base_plan,
         "peak_window_s": PEAK_WINDOW_S, "daemon": {k: v for k, v in ds.items() if k != "tree"},
         "cap_mib": CAP_MIB, "headroom_mib": HEADROOM_MIB, "worker_mib": WORKER_MIB}
    return d


def publish(d, extra=None):
    os.makedirs(BUDGET_DIR, exist_ok=True)
    if extra:
        d = {**d, **extra}
    tmp = f"{BUDGET_DIR}/budget.json.tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, f"{BUDGET_DIR}/budget.json")
    with open(f"{BUDGET_DIR}/budget_log.jsonl", "a") as f:
        f.write(json.dumps(d) + "\n")


def wait_for_slot(n=1, poll=30, max_wait=None, log=print):
    """Block until the budget grants >= n workers to a process that holds none yet (used by single-process tests)."""
    t0 = time.time()
    while True:
        d = assess(set())
        if d["granted_workers"] >= n and not d["over_cap"]:
            return d
        if max_wait and time.time() - t0 > max_wait:
            raise TimeoutError(json.dumps(d))
        log(f"gpu3 budget: waiting ({d['granted_workers']} granted; daemon {d['daemon']})")
        time.sleep(poll)


if __name__ == "__main__":
    print(json.dumps(assess(set()), indent=1))
