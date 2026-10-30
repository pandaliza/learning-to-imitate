"""GPU-gated queue for the wipe v2 matched-condition (all-60) trainings; a copy of queue_v2.py with its own job list,
paths and DONE.json-based completion. Original queue_v2 docstring follows.
 Coordinator allocation 2026-10-25 (Amendment 6): any
GPU with >= 8 GB free AFTER the job's footprint, one job of this queue per GPU; preferred 0/1/2. If a GPU holding
one of our jobs has < 8 GB free on 3 consecutive polls (~90 s), our NEWEST job on that GPU is preempted (SIGTERM; it
resumes from its newest checkpoint; checkpoints every 1000 steps, non-5k ones pruned). Jobs are tracked by pid and
by the state file, so the queue itself can be restarted without touching running jobs (they are re-adopted).
Per-job memory needs can be overridden in runs/wipe_v2/train/_queue_needs.json {"job": gb}.
Status: runs/wipe_v2/train/_queue_status.json. Stop launching: touch runs/wipe_v2/train/QUEUE_STOP."""
import glob, json, os, signal, subprocess, time

W = os.path.dirname(os.path.abspath(__file__))
R = "/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/all60"
LOGS = f"{R}/train/_logs"
STATUS = f"{R}/train/_queue_status.json"
NEEDS = f"{R}/train/_queue_needs.json"
GPUS = {0: "100-104", 1: "105-109", 2: "110-114", 3: "115-119", 4: "120-124", 5: "95-99", 6: "90-94", 7: "85-89"}
PREFER = (0, 1, 2)
COMMON = ["--task", "wipe_countertop", "--workers", "4", "--act-ckpt"]
JOBS = [  # name, args, need_gb, runs_root
    ("verify_data40_amix_s0", ["--data40", "--arm", "amix", "--seed", "0", "--num-steps", "200", "--label", "verify",
                               "--run-id", "amix_wipe_countertop_data40_s0__verify200", "--no-resume"], 12, f"{R}/verify"),
] + [(f"{a}_s{s}", ["--arm", a, "--seed", str(s), "--run-id", f"{a}_wipe_countertop_all60_s{s}", "--save-every", "1000"],
      12, f"{R}/train") for s in (0, 1, 2) for a in ("amix", "ipath")]
DEPS = {n: "verify_data40_amix_s0" for n, *_ in JOBS if n != "verify_data40_amix_s0"}


def run_dir_of(args, root):
    return os.path.join(root, args[args.index("--run-id") + 1])


def free_gb():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]).decode()
    return {int(a): int(b) / 1024 for a, b in (l.split(",") for l in out.strip().splitlines())}


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return open(f"/proc/{pid}/stat").read().split()[2] != "Z"
    except OSError:
        return False


def need_of(n, default):
    try:
        ov = json.load(open(NEEDS))
        if n in ov:
            return float(ov[n])
    except (OSError, ValueError):
        pass
    if False:
        p = f"{R}/verify/w2_wipe_countertop_v2_s0__smoke200/meta.json"
        if os.path.exists(p):
            m = json.load(open(p)).get("max_gpu_mem_gb")
            if m:
                return 1.3 * m + 2
    return default


def prune():
    for d in glob.glob(f"{R}/train/*_all60_s*/ckpt"):
        steps = sorted(int(os.path.basename(f)[5:-4]) for f in glob.glob(f"{d}/step_*.pth"))
        if not steps:
            continue
        newest = steps[-1]
        if time.time() - os.path.getmtime(f"{d}/step_{newest}.pth") < 120:
            continue
        for x in steps:
            if x % 5000 != 0 and x != newest:
                try:
                    os.remove(f"{d}/step_{x}.pth")
                except FileNotFoundError:
                    pass


def save(st, fr):
    tmp = STATUS + ".tmp"
    json.dump({"time": time.time(), "jobs": st, "free_gb": fr}, open(tmp, "w"), indent=1)
    os.replace(tmp, STATUS)


def main():
    st = {n: {"state": "pending", "attempts": 0} for n, *_ in JOBS}
    if os.path.exists(STATUS):
        old = json.load(open(STATUS))["jobs"]
        for n in st:
            if n in old:
                st[n] = old[n]
                if old[n]["state"] == "running" and not alive(old[n].get("pid", -1)):
                    st[n]["state"] = "check"   # ended while the queue was down: classify from the log below
    children = {}
    low = {}
    while True:
        for p in list(children.values()):
            p.poll()
        for n, s in st.items():
            if s["state"] not in ("running", "check") or (s["state"] == "running" and alive(s["pid"])):
                continue
            tail = open(s["log"], errors="ignore").read()[-4000:]
            if os.path.exists(os.path.join(s["run_dir"], "DONE.json")) or "ALREADY_DONE" in tail or "TRAIN_DONE" in tail:
                s["state"] = "done"
            elif "TRAIN_END stopped" in tail:
                s["state"] = "stopped"
            elif s.get("failures", 0) < 4 and ("out of memory" in tail.lower() or "SIGTERM" in tail or "refusing" in tail):
                if "SIGTERM" not in tail:
                    s["failures"] = s.get("failures", 0) + 1
                s["state"] = "pending"
            else:
                s["state"] = "failed"
            s["ended"] = time.time()
        fr = free_gb()
        run = {n: s for n, s in st.items() if s["state"] == "running"}
        for g in {s["gpu"] for s in run.values()}:
            low[g] = low.get(g, 0) + 1 if fr.get(g, 99) < 8 else 0
            if low[g] >= 3:  # ~90 s below 8 GB (lets the render budget evict first)
                _, n = max((s["started"], n) for n, s in run.items() if s["gpu"] == g)
                os.kill(run[n]["pid"], signal.SIGTERM)
                st[n]["preemptions"] = st[n].get("preemptions", 0) + 1
                low[g] = 0
        prune()
        if not os.path.exists(f"{R}/train/QUEUE_STOP"):
            busy = {s["gpu"] for s in st.values() if s["state"] == "running"}
            for n, args, need, root in JOBS:
                if st[n]["state"] != "pending" or (n in DEPS and st[DEPS[n]]["state"] != "done"):
                    continue
                need = need_of(n, need)
                cand = [g for g in GPUS if g not in busy and fr.get(g, 0) >= need + 8]
                if not cand:
                    continue
                pref = [g for g in cand if g in PREFER]
                g = max(pref or cand, key=lambda x: fr[x])
                st[n]["attempts"] += 1
                log = f"{LOGS}/{n}.a{st[n]['attempts']}.log"
                cmd = [f"{W}/launch_train.sh", str(g), GPUS[g], log, "--", *args, *COMMON, "--runs-root", root]
                p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     env=dict(os.environ, V2_SCRIPT="train_all60.py"))
                children[n] = p
                st[n].update({"state": "running", "gpu": g, "log": log, "started": time.time(), "pid": p.pid, "run_dir": run_dir_of(args, root),
                              "need_gb": need})
                busy.add(g)
                save(st, fr)
                time.sleep(90)
                fr = free_gb()
        save(st, fr)
        if all(s["state"] not in ("pending", "running", "check") for s in st.values()):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
