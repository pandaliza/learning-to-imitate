"""Amendment 2 evaluation daemon (nohup). Never touches training; renders on GPU 3, infers on GPU 0.

Phase 1 (evaluation gate): 5-reset dev smoke rollouts (eval_dev[0:5], rollout seed 0) for the 10 smoke models,
  then check_gate.py evaluation (also requires 3/3 exact GPU-3 parity runs). If the gate fails, stop.
Phase 2: random-policy test evaluation, in training-queue order, of every COMPLETED run (DONE.json):
  200 eval_test reset families x rollout seeds {0,1,2}, checkpoint step_30000.pth, contract oopsie-metric-v1.0.
  Records: runs/matrix_eval/<run_id>/test_s30000_<run_key>/<reset>__r<seed>.json (atomic, resumable; infra errors
  kept and retried <= 2x, logged separately). Per-run tables via summarize_eval.py. New runs are picked up as they
  complete. Ledger: outputs/oopsieverse_full_study/matrix/eval_ledger.csv. Stop: touch runs/matrix/_queue/STOP_EVAL.
"""
import csv
import datetime
import fcntl
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from queue_runner import queue, rid, env  # noqa: E402

PY = "/edrive1/lizad/oopsieverse_full_study/env/oopsie/bin/python"
OVS = "/edrive1/lizad/oopsieverse_full_study"
RUNS = f"{OVS}/runs/matrix"
EVAL = f"{OVS}/runs/matrix_eval"
SMOKE = f"{OVS}/runs/matrix_smoke"
STUDY = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study"
LEDGER = f"{STUDY}/matrix/eval_ledger.csv"
STATUS = f"{STUDY}/matrix/eval_status.json"
STOP = f"{RUNS}/_queue/STOP_EVAL"
SMOKE_SPECS = [("pick_egg", a) for a in ("amix", "asafe", "ssmix", "imean", "ipath", "ipathg", "cmean", "auxpathg")] + \
              [("wipe_countertop", "amix"), ("wipe_countertop", "ipathg")]
SEEDS = "0,1,2"
N_TEST = 200 * 3
FIELDS = ["time", "run_id", "event", "phase", "n_records", "n_infra_pending", "wall_s", "episodes_per_h", "out_dir", "note"]


def ledger(row):
    row = {k: row.get(k, "") for k in FIELDS}
    row["time"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(LEDGER, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, 2)
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


def count(out_dir):
    ok = inf = 0
    for f in glob.glob(os.path.join(out_dir, "*__r*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if "infra_error" in r:
            inf += 1
        else:
            ok += 1
    return ok, inf


def evaluate(run_dir, task, resets, seeds, out_dir, ckpt, limit=None, workers=8, log=None):
    cmd = [PY, os.path.join(HERE, "rollout_eval_matrix.py"), "--run-dir", run_dir, "--task", task, "--resets", resets,
           "--rollout-seeds", seeds, "--out-dir", out_dir, "--workers", str(workers), "--ckpt", ckpt]
    if limit:
        cmd += ["--limit", str(limit)]
    e = env()
    for k in ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID", "EGL_DEVICE_ID"):
        e.pop(k, None)  # set per process inside rollout_eval_matrix (GPU 0 inference / GPU 3 render)
    t0 = time.time()
    rc = subprocess.call(cmd, stdout=open(log, "a"), stderr=subprocess.STDOUT, env=e, cwd=HERE)
    return rc, time.time() - t0


def status(d):
    d["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(d, open(STATUS + ".tmp", "w"), indent=1)
    os.replace(STATUS + ".tmp", STATUS)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(f"{EVAL}/logs", exist_ok=True)
    st = {"phase": "gate", "runner_pid": os.getpid(), "workers": a.workers, "evaluated": {}, "rates_eph": []}
    # ── phase 1: evaluation gate ──
    for task, arm in SMOKE_SPECS:
        if os.path.exists(STOP):
            return
        r = f"{arm}_{task}_v1_s0__smoke500"
        out = f"{SMOKE}/eval_gpu3/{r}"
        if count(out)[0] >= 5:
            continue
        st["current"] = r
        status(st)
        rc, w = evaluate(f"{SMOKE}/{r}", task, "dev", "0", out, "final.pth", limit=5, workers=5, log=f"{EVAL}/logs/{r}.log")
        n, inf = count(out)
        ledger({"run_id": r, "event": "smoke_rollouts", "phase": "evaluation_gate", "n_records": n,
                "n_infra_pending": inf, "wall_s": round(w), "out_dir": out, "note": f"rc={rc}"})
    g = subprocess.call([PY, os.path.join(HERE, "check_gate.py"), "evaluation"], stdout=open(f"{EVAL}/logs/gate.log", "w"),
                        stderr=subprocess.STDOUT, cwd=HERE)
    gate = json.load(open(f"{STUDY}/matrix/verification/gate.json"))["evaluation_gate"]
    ledger({"run_id": "-", "event": "evaluation_gate_" + ("PASS" if g == 0 else "FAIL"), "phase": "evaluation_gate",
            "note": json.dumps({k: v for k, v in gate.items() if k != "gpu3_render_parity"})})
    st["evaluation_gate"] = gate
    if g != 0:
        st["phase"] = "stopped: evaluation gate FAILED"
        status(st)
        return
    # ── phase 2: test evaluation of completed runs, in queue order ──
    st["phase"] = "test"
    while not os.path.exists(STOP):
        did = False
        for (arm, task, s) in queue():
            r = rid(arm, task, s)
            rd = f"{RUNS}/{r}"
            if not os.path.exists(f"{rd}/DONE.json"):
                continue
            key = json.load(open(f"{rd}/DONE.json"))["run_key"]
            out = f"{EVAL}/{r}/test_s30000_{key}"
            n, inf = count(out)
            if n >= N_TEST:
                st["evaluated"][r] = {"n": n, "infra_pending": inf}
                continue
            if inf and n + inf >= N_TEST and len(glob.glob(f"{out}/_infra/*.json")) >= 2 * inf:
                st["evaluated"][r] = {"n": n, "infra_pending": inf, "note": "infra retries exhausted"}
                continue
            assert os.path.exists(f"{rd}/ckpt/step_30000.pth"), r
            st["current"] = r
            status(st)
            ledger({"run_id": r, "event": "start", "phase": "test", "n_records": n, "out_dir": out})
            rc, w = evaluate(rd, task, "test", SEEDS, out, "step_30000.pth", workers=a.workers, log=f"{EVAL}/logs/{r}.log")
            n2, inf2 = count(out)
            eph = (n2 - n) / (w / 3600) if w > 0 else None
            st["rates_eph"].append({"run": r, "episodes_per_h": eph, "task": task})
            subprocess.call([PY, os.path.join(HERE, "summarize_eval.py"), out, f"{STUDY}/matrix/eval/test_{r}.json"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, cwd=HERE)
            subprocess.call([PY, os.path.join(HERE, "summarize_eval.py"), "--index"], cwd=HERE)
            ledger({"run_id": r, "event": "completed" if n2 >= N_TEST else "partial", "phase": "test", "n_records": n2,
                    "n_infra_pending": inf2, "wall_s": round(w), "episodes_per_h": round(eph or 0, 1), "out_dir": out,
                    "note": f"rc={rc}"})
            st["evaluated"][r] = {"n": n2, "infra_pending": inf2}
            did = True
            break  # re-scan in queue order
        st.pop("current", None)
        q = queue()
        n_done_train = sum(os.path.exists(f"{RUNS}/{rid(*x)}/DONE.json") for x in q)
        st.update({"n_runs_total": len(q), "n_runs_trained": n_done_train,
                   "n_runs_evaluated": sum(1 for v in st["evaluated"].values() if v["n"] >= N_TEST)})
        status(st)
        if st["n_runs_evaluated"] >= len(q):
            break
        if not did:
            time.sleep(300)
    st["phase"] = "finished" if not os.path.exists(STOP) else "stopped (STOP_EVAL)"
    status(st)


if __name__ == "__main__":
    main()
