"""Multi-GPU EGL render budget for the selection study (Amendment 6; generalises gpu3_budget.py).

Rules (user, 2026-10-22):
  * render MuJoCo EGL only on GPUs that passed the exact pixel-parity check (PASS_FILES below); never GPUs 0 or 2;
  * on every GPU we use keep >= 8 GB free; plan against the LOWEST free memory (with our selection workers removed)
    seen over the last PEAK_WINDOW_S (strict policy: other users' jobs spike);
  * on a breach (free < 8 GB on a GPU where we hold workers) the supervisor evicts its youngest worker on that GPU
    via the SIGUSR1 safe-exit path (records are atomic; the lost unit re-runs);
  * a global worker cap shared by all selection supervisors (CPU cap: ~64 cores total incl. servers/supervisors).

Several supervisors (one per training seed) share GPUs through a ledger: each supervisor writes
runs/selection/_gpus/holders/<supervisor pid>.json = {worker pid: gpu} (+ pending reservations), under an flock.
Capacity of GPU g = floor((min_window(free_g + our_selection_mib_g) - HEADROOM - MARGIN) / WORKER_MIB);
a supervisor may spawn on g if the workers held by ALL supervisors on g are below that capacity. Every decision is
appended to runs/selection/_gpus/budget_log.jsonl.
"""

import fcntl
import json
import os
import subprocess
import time

OVS = "/edrive1/lizad/oopsieverse_full_study"
STUDY = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study"
LEDGER = f"{OVS}/runs/selection/_gpus"
HEADROOM_MIB = 8 * 1024
WORKER_MIB = 1150           # measured 1,084 MiB per EGL worker + margin (as gpu3_budget)
MARGIN_MIB = 256
PEAK_WINDOW_S = 30 * 60
NEVER_RENDER = {0, 2}       # known corrupt EGL frames
PENDING_TTL_S = 120         # a reservation for a just-spawned worker lasts until its pid is registered
_HIST = {}                  # g -> [(t, base_mib)]
_SEEDED = [False]


def parity_passed_gpus():
    """GPUs with an exact parity PASS for BOTH the plain and the ctxfix variant (selection workers use ctxfix)."""
    import glob
    ok = {}
    for f in glob.glob(f"{STUDY}/selection/verification/render_parity_egl_gpu*_run*.json") + \
            glob.glob(f"{STUDY}/matrix/verification/render_parity_egl_gpu3_*run*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        g = int(d["env"]["MUJOCO_EGL_DEVICE_ID"])
        kind = "ctxfix" if d["env"].get("CTXFIX") == "1" else "plain"
        ok.setdefault(g, {}).setdefault(kind, []).append(bool(d["pass"]) and d["step_min_seg_equal_frac"] == 1.0)
    def replay_ok(g):
        # GPU 3 rendered the dev run itself (dev replays H8); every other GPU needs a cross-GPU replay of dev records
        if g == 3:
            return True
        try:
            r = json.load(open(f"{STUDY}/selection/verification/replay_gpu{g}.json"))
        except Exception:
            return False
        return len(r) >= 6 and all(x["identical"] for x in r)
    return sorted(g for g, v in ok.items() if g not in NEVER_RENDER
                  and v.get("plain") and v.get("ctxfix") and all(v["plain"]) and all(v["ctxfix"]) and replay_ok(g))


def gpu_table():
    """({g: {pid: MiB}} for every process (compute + graphics), {g: free MiB})."""
    free = {}
    for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                                        timeout=30).decode().splitlines():
        i, f = line.split(",")
        free[int(i)] = float(f)
    procs = {g: {} for g in free}
    for line in subprocess.check_output(["nvidia-smi"], timeout=30).decode().splitlines():
        parts = line.strip("| ").split()
        if len(parts) >= 7 and parts[0].isdigit() and parts[3].isdigit() and parts[-1].endswith("MiB"):
            g, pid = int(parts[0]), int(parts[3])
            if g in procs:
                procs[g][pid] = procs[g].get(pid, 0.0) + float(parts[-1][:-3])
    return procs, free


def _alive(pid):
    return os.path.exists(f"/proc/{pid}")


class Ledger:
    def __init__(self, holder_id, label=""):
        os.makedirs(f"{LEDGER}/holders", exist_ok=True)
        self.id, self.label = str(holder_id), label
        self.path = f"{LEDGER}/holders/{self.id}.json"
        self.lockf = open(f"{LEDGER}/lock", "a")

    def __enter__(self):
        fcntl.flock(self.lockf, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        fcntl.flock(self.lockf, fcntl.LOCK_UN)

    def read_all(self):
        out = {}
        for n in os.listdir(f"{LEDGER}/holders"):
            if not n.endswith(".json"):
                continue
            try:
                d = json.load(open(f"{LEDGER}/holders/{n}"))
            except Exception:
                continue
            if not _alive(int(d["holder_pid"])):
                continue  # stale (holder died)
            out[n[:-5]] = d
        return out

    def write(self, workers, pending):
        d = {"holder_pid": os.getpid(), "label": self.label, "time": time.time(),
             "workers": {str(p): g for p, g in workers.items()}, "pending": pending}
        tmp = self.path + ".tmp"
        json.dump(d, open(tmp, "w"))
        os.replace(tmp, self.path)

    def remove(self):
        try:
            os.remove(self.path)
        except OSError:
            pass


def _seed_hist(now):
    if _SEEDED[0]:
        return
    _SEEDED[0] = True
    try:
        for line in open(f"{LEDGER}/budget_log.jsonl").readlines()[-5000:]:
            x = json.loads(line)
            if now - x["t"] < PEAK_WINDOW_S:
                for g, b in x.get("base_mib", {}).items():
                    _HIST.setdefault(int(g), []).append((x["t"], b))
    except Exception:
        pass


def assess(ledger, my_workers, render_gpus, worker_cap_total, my_cap, want_spawn=True):
    """my_workers: {pid: gpu}. Called by a supervisor with its current workers. Under the ledger lock: registers them,
    computes per-GPU capacity, and (if want_spawn) reserves one slot on the GPU with the most spare capacity.
    Returns a decision dict: spawn_gpu (reserved, or None), over {g: True} (breach on a GPU where we hold workers),
    yield_gpu (we hold the most workers on an over-subscribed GPU), per-GPU table."""
    procs, free = gpu_table()
    now = time.time()
    with ledger:
        _seed_hist(now)
        mine_pending = []
        ledger.write(my_workers, mine_pending)
        hold = ledger.read_all()
        held = {g: 0 for g in free}
        held_mine = {g: 0 for g in free}
        sel_pids = {g: set() for g in free}
        n_total = 0
        for hid, d in hold.items():
            for p, g in d["workers"].items():
                if _alive(int(p)):
                    held[g] = held.get(g, 0) + 1
                    sel_pids.setdefault(g, set()).add(int(p))
                    n_total += 1
                    if hid == ledger.id:
                        held_mine[g] += 1
            for r in d.get("pending", []):
                if now - r["t"] < PENDING_TTL_S:
                    held[r["gpu"]] = held.get(r["gpu"], 0) + 1
                    n_total += 1
        per = {}
        for g in sorted(free):
            ours = sum(v for p, v in procs.get(g, {}).items() if p in sel_pids.get(g, set()))
            base_now = free[g] + ours
            h = _HIST.setdefault(g, [])
            h.append((now, base_now))
            while h and now - h[0][0] > PEAK_WINDOW_S:
                h.pop(0)
            base_plan = min(b for _, b in h)
            cap = max(0, int((base_plan - HEADROOM_MIB - MARGIN_MIB) // WORKER_MIB)) if g in render_gpus else 0
            per[g] = {"free_mib": free[g], "ours_mib": ours, "base_now_mib": base_now, "base_plan_mib": base_plan,
                      "capacity": cap, "held_all": held.get(g, 0), "held_mine": held_mine.get(g, 0),
                      "render_ok": g in render_gpus}
        over = {g: True for g, x in per.items() if x["held_mine"] > 0 and x["free_mib"] < HEADROOM_MIB}
        yield_gpu = None
        for g, x in per.items():
            if x["held_all"] > x["capacity"] and x["held_mine"] > 0:
                # the supervisor holding the most workers on g yields (ties: lowest holder id)
                counts = sorted(((sum(1 for p, gg in d["workers"].items() if gg == g and _alive(int(p))), hid)
                                 for hid, d in hold.items()), key=lambda c: (-c[0], c[1]))
                if counts and counts[0][1] == ledger.id:
                    yield_gpu = g
                    break
        spawn_gpu = None
        n_mine = sum(held_mine.values())
        if want_spawn and not over and yield_gpu is None and n_total < worker_cap_total and n_mine < my_cap:
            spare = sorted(((x["capacity"] - x["held_all"], g) for g, x in per.items() if x["render_ok"]), reverse=True)
            if spare and spare[0][0] > 0:
                spawn_gpu = spare[0][1]
                mine_pending.append({"gpu": spawn_gpu, "t": now})
                ledger.write(my_workers, mine_pending)
        d = {"t": now, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "holder": ledger.id, "label": ledger.label,
             "render_gpus": list(render_gpus), "n_workers_all": n_total, "n_workers_mine": n_mine,
             "worker_cap_total": worker_cap_total, "my_cap": my_cap, "spawn_gpu": spawn_gpu,
             "over": over, "yield_gpu": yield_gpu, "gpus": per, "base_mib": {g: x["base_now_mib"] for g, x in per.items()},
             "headroom_mib": HEADROOM_MIB, "worker_mib": WORKER_MIB, "peak_window_s": PEAK_WINDOW_S, "policy": "strict"}
        with open(f"{LEDGER}/budget_log.jsonl", "a") as f:
            f.write(json.dumps(d) + "\n")
    return d


def single_slot_ok(g, extra_mib=WORKER_MIB):
    """For one-off verification processes (parity / replay): True if GPU g keeps >= 8 GB free (strict window) with one
    more EGL context."""
    procs, free = gpu_table()
    now = time.time()
    _seed_hist(now)
    h = _HIST.get(g, [])
    base = min([free[g]] + [b for t, b in h if now - t < PEAK_WINDOW_S])
    return base - extra_mib - MARGIN_MIB >= HEADROOM_MIB, {"free_mib": free[g], "plan_mib": base}


if __name__ == "__main__":
    print(json.dumps({"parity_passed": parity_passed_gpus(), "table": gpu_table()[1]}, indent=1))
