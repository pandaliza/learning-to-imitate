"""Throughput and ETA for the selection workload under the current caps (writes selection/throughput_eta.json + .md).

Model: wall time = remaining continuations x worker-seconds per continuation / (workers x duty), where
  * worker-seconds per continuation is measured from the dev records (wall_s, includes restore, inference wait,
    rendering, physics; the shared candidate chunk and state collection are added pro rata);
  * workers = GPU-3 cap: floor((10240 - 256) / 1150) = 8 EGL workers while the eval daemon is idle; 0 while it runs
    (its 9 workers use ~9.75 GB of the 10 GB);
  * duty = fraction of wall time GPU 3 is free of the daemon, from the eval ledger (evaluation wall per run) and the
    training queue rate (2 concurrent jobs, ~2.0 updates/s each, 30k updates), plus the 8-min pre-emptive drain.
"""
import csv
import glob
import json
import os

import numpy as np

STUDY = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study"
ROOT = "/edrive1/lizad/oopsieverse_full_study/runs/selection/pick_egg/sel_dev_parents/train_seed_0"
WORKERS = int(os.environ.get("OV_ETA_WORKERS", 8))
TAG = os.environ.get("OV_ETA_TAG", "")
M, R = 16, 4


def main():
    recs = [json.load(open(f)) for f in glob.glob(f"{ROOT}/branches/*/p*/c*_r*.json")]
    states = [json.load(open(f)) for f in glob.glob(f"{ROOT}/states/p*.json")]
    w = np.array([r["latency"]["wall_s"] + r["latency"]["candidate_chunk_s_shared"] / R for r in recs])
    steps = np.array([r["steps"] - r["state_t_b"] for r in recs])
    by_arm = {}
    for r in recs:
        by_arm.setdefault(f"{r['arm']}__{r['variant']}", []).append(r["latency"]["wall_s"])
    coll_s = float(np.mean([s["wall_s"] for s in states])) if states else 25.0
    ws = float(w.mean())
    # daemon duty from the eval ledger
    ev = [r for r in csv.DictReader(open(f"{STUDY}/matrix/eval_ledger.csv")) if r["event"] == "completed" and r["phase"] == "test"]
    ev_h = float(np.mean([float(r["wall_s"]) for r in ev])) / 3600 if ev else 1.0
    qs = [json.load(open(f)) for f in glob.glob(f"{STUDY}/matrix/queue_status*.json")]
    rates = [v["progress"]["updates_per_s"] for q in qs for v in q.get("running", {}).values()
             if v.get("progress") and v["progress"].get("updates_per_s")]
    total_rate = float(np.sum(rates)) if rates else 4.0       # updates/s summed over all training jobs (GPUs 0, 1, 4)
    runs_per_h = total_rate * 3600 / 30000
    done = set(r for q in qs for r in q.get("done", []))
    n_total = max(q["n_total"] for q in qs)
    n_pending_train = n_total - len(done)
    train_left_h = n_pending_train / max(runs_per_h, 1e-6)
    n_evaluated = sum(1 for v in json.load(open(f"{STUDY}/matrix/eval_status.json")).get("evaluated", {}).values() if v.get("n", 0) >= 600)
    n_uneval = n_total - n_evaluated
    daemon_busy_h = n_uneval * (ev_h + 8 / 60)          # every remaining evaluation + its drain
    horizon_h = max(train_left_h, daemon_busy_h)        # GPU 3 is shared until both are over
    duty = max(0.0, 1 - daemon_busy_h / horizon_h) if horizon_h > 0 else 1.0
    cycle_h = 1 / max(runs_per_h, 1e-6)
    eval_left_h = daemon_busy_h
    per_worker_h = 3600 / ws
    free_rate = WORKERS * per_worker_h
    eff_rate = free_rate * duty

    def eta(n, extra_states=0):
        work_h = (n * ws + extra_states * coll_s) / 3600 / WORKERS   # hours of free-GPU time
        # while the queue runs we get `duty`; afterwards (all trained + evaluated) duty = 1
        horizon = horizon_h
        if work_h <= horizon * duty:
            return work_h / duty, work_h
        return horizon + (work_h - horizon * duty), work_h

    dev_done = len(recs)
    dev_prim = 50 * 4 * M * R
    dev_sens = 50 * 3 * M * R
    test_prim = 100 * 3 * 4 * M * R
    test_sens = 100 * 3 * 3 * M * R
    out = {
        "measured_on": f"{len(recs)} dev continuations, {len(states)} dev states (pick_egg, training seed 0)",
        "worker_s_per_continuation": ws, "mean_continuation_steps": float(steps.mean()),
        "worker_s_per_env_step": float(w.sum() / steps.sum()),
        "worker_s_per_continuation_by_arm": {k: float(np.mean(v)) for k, v in by_arm.items()},
        "worker_s_per_state_collection": coll_s,
        "gpu3_workers_when_daemon_idle": WORKERS, "continuations_per_hour_when_free": free_rate,
        "daemon_duty": {"eval_h_per_run": ev_h, "hours_between_completed_runs": cycle_h, "training_updates_per_s_total": total_rate, "free_fraction": duty,
                        "training_runs_left": n_pending_train, "training_queue_left_h": train_left_h,
                        "runs_not_yet_evaluated": n_uneval, "daemon_busy_h_remaining": daemon_busy_h,
                        "shared_period_h": horizon_h},
        "effective_continuations_per_hour_while_queue_runs": eff_rate,
        "dev": {"done": dev_done, "primary_target": dev_prim, "sensitivity_target": dev_sens},
        "eta_wall_h": {
            "dev_primary_remaining": eta(max(0, dev_prim - dev_done), 50 - len(states))[0],
            "dev_all_remaining": eta(max(0, dev_prim + dev_sens - dev_done), 50 - len(states))[0],
            "test_primary_pick_egg_3_seeds": eta(max(0, dev_prim + dev_sens - dev_done) + test_prim, 300 + 50 - len(states))[0]
                                             - eta(max(0, dev_prim + dev_sens - dev_done), 50 - len(states))[0],
            "test_primary_plus_sensitivity_pick_egg_3_seeds": eta(max(0, dev_prim + dev_sens - dev_done) + test_prim + test_sens, 300 + 50 - len(states))[0]
                                             - eta(max(0, dev_prim + dev_sens - dev_done), 50 - len(states))[0],
            "dev_all_then_test_primary_total": eta(max(0, dev_prim + dev_sens - dev_done) + test_prim, 300 + 50 - len(states))[0],
            "dev_all_then_test_all_total": eta(max(0, dev_prim + dev_sens - dev_done) + test_prim + test_sens, 300 + 50 - len(states))[0]},
        "free_gpu_hours_needed": {"test_primary_pick_egg_3_seeds": eta(test_prim, 300)[1],
                                  "test_primary_plus_sensitivity": eta(test_prim + test_sens, 300)[1]},
        "assumptions": ["test continuations cost the same as dev ones (same policies, same branch window)",
                        "test runs after dev; seed-1/2 checkpoints exist by then (seed-2 runs are still queued)",
                        "8 GPU-3 workers whenever the daemon is idle; 0 while it evaluates; duty = 1 once the training queue and its evaluations finish",
                        "CPU: 8 workers + server ~ 10 cores, inside the 18-core cap (Amendment 3); GPU 3 memory is the binding cap",
                        "wipe_countertop (diagnostic) not included: episodes are ~2x longer (1250-step timeout); not measured here"],
    }
    json.dump(out, open(f"{STUDY}/selection/throughput_eta{TAG}.json", "w"), indent=1)
    e = out["eta_wall_h"]
    md = f"""# Selection throughput and ETA (measured {out['measured_on']})

## Measured cost

| Quantity | Value |
|---|---|
| Worker-seconds per continuation (1 chunk + own tail to 650 steps, incl. restore, render, inference wait) | {ws:.1f} s |
| Mean continuation length | {steps.mean():.0f} env steps |
| Worker-seconds per env step | {out['worker_s_per_env_step']*1000:.0f} ms |
| Per arm (s/continuation) | {', '.join(f'{k} {v:.1f}' for k, v in out['worker_s_per_continuation_by_arm'].items())} |
| State collection | {coll_s:.0f} s per parent |

## Capacity under the current caps

- **GPU 3 workers (this scenario).** {WORKERS} EGL workers while the eval daemon is idle; 0 while it evaluates.
- **Throughput.** That gives {free_rate:.0f} continuations/h when GPU 3 is free.
- **Daemon duty cycle.** Training (all queues, summed {total_rate:.1f} updates/s) completes a run every ~{cycle_h:.1f} h. Each completed run takes ~{ev_h:.2f} h to evaluate, and each evaluation is preceded by an 8-min drain. {n_uneval} runs still need evaluation (~{daemon_busy_h:.0f} h of daemon time) and training needs ~{train_left_h:.0f} h more, so GPU 3 is shared for the next ~{horizon_h:.0f} h and free about **{duty:.0%}** of that time. After that it is free 100% of the time.
- **Effective rate while the queue runs.** About {eff_rate:.0f} continuations/h.

## ETA (wall clock)

| Workload | Continuations | ETA |
|---|---|---|
| Dev primary (4 arms × 50 × 16 × 4), remaining | {max(0, dev_prim - dev_done):,} | {e['dev_primary_remaining']:.1f} h |
| Dev primary + joint sensitivity (3 intent arms), remaining | {max(0, dev_prim + dev_sens - dev_done):,} | {e['dev_all_remaining']:.1f} h |
| **Test, pick_egg, 3 training seeds, primary** (4 × 3 × 100 × 16 × 4) | {test_prim:,} | **{e['test_primary_pick_egg_3_seeds']:.0f} h ≈ {e['test_primary_pick_egg_3_seeds']/24:.1f} days** (needs {out['free_gpu_hours_needed']['test_primary_pick_egg_3_seeds']:.0f} free-GPU-3 h) |
| Test, pick_egg, primary + sensitivity | {test_prim + test_sens:,} | {e['test_primary_plus_sensitivity_pick_egg_3_seeds']:.0f} h ≈ {e['test_primary_plus_sensitivity_pick_egg_3_seeds']/24:.1f} days |

Test rows give the additional wall time once dev (incl. sensitivity) has finished. Total from now: dev then test primary {e['dev_all_then_test_primary_total']:.0f} h ({e['dev_all_then_test_primary_total']/24:.1f} d); dev then test primary + sensitivity {e['dev_all_then_test_all_total']:.0f} h ({e['dev_all_then_test_all_total']/24:.1f} d). Further assumptions: {'; '.join(out['assumptions'])}.
"""
    open(f"{STUDY}/selection/throughput_eta{TAG}.md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
