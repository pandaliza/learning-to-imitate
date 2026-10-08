"""Resource estimate for the spec section-14 matrix from measured throughput (see resource_plan.md).

Measured inputs (all on the shared host, 8x H200 at 100% util from other users, load avg 120-330):
  RoboCasa pick_egg, official A-MIX @30k, eval_dev[0:30] screen (30 workers + 1 batched GPU server):
    reset 20.9 s, worker-seconds per control step incl. inference wait 0.147 s, mean episode 340 steps
  RoboCasa wipe (throughput bench, 16 workers, untrained policy): reset 22.6 s, env step 0.115 s
  Training (official CFM, 30,001 updates, bs 64, fp32): 25,804 s = 7.17 h on one shared H200, 19.7 GB peak
  B1K add_firewood (b1k agent, bench_rollout_add_firewood.json): 3.45 steps/s, 1.8 s reset, 218 s env
    build; per process 11-17 GB RSS, ~13 cores, 1 GPU; cameras OFF in that bench (policy needs seg views).
Assumptions are explicit in TASKS below. Writes outputs/oopsieverse_full_study/baseline/resource_estimate.json.
"""
import json

OUT = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/baseline/resource_estimate.json"
TRAIN_H = 25804 / 3600  # measured, one shared H200
CKPT_GB = 0.236 * 7 + 0.05  # 7 checkpoints (state+optimizer) + logs per run

TASKS = {
    # reset_s, s_per_step (worker wall incl. inference wait), ep_len, tail_len, restore_s, workers, provenance
    "pick_egg": dict(reset_s=20.9, s_per_step=0.147, ep_len=340, tail_len=200, restore_s=1.0, workers=30,
                     note="measured @30k screen; tail_len assumed (median first success 182, branch at a pre-damage state)"),
    "wipe_countertop": dict(reset_s=22.6, s_per_step=0.147, ep_len=600, tail_len=500, restore_s=1.0, workers=30,
                            note="reset/step measured (bench); episode/tail length ASSUMED (no wipe policy yet; timeout 1250, demo median 281)"),
    "add_firewood": dict(reset_s=1.8, s_per_step=1 / 3.45 + 0.06, ep_len=250, tail_len=200, restore_s=1.0, workers=2,
                         note="B1K bench 3.45 steps/s cameras OFF + 0.06 s/step inference; 2 concurrent processes (13 cores + 1 GPU each, "
                              "<=32 CPU and <=2 GPU rule); episode length ASSUMED (demo mean 164 steps); seg rendering will slow it"),
}

PROTOCOLS = {
    "full": dict(test_parents=100, dev_parents=50, sel_resets=200),
    "R1_parents_x0.5": dict(test_parents=50, dev_parents=25, sel_resets=100),
    "R2_parents_x0.25": dict(test_parents=25, dev_parents=13, sel_resets=50),
}


def task_cost(t, p):
    ep_ws = t["reset_s"] + t["s_per_step"] * t["ep_len"]
    cont_ws = t["restore_s"] + t["s_per_step"] * t["tail_len"]
    n = {
        "training_runs": 18,
        "random_eval_episodes": 6 * 3 * 200 * 3,
        "test_continuations": 4 * 3 * p["test_parents"] * 16 * 4,
        "dev_continuations": 4 * 3 * p["dev_parents"] * 16 * 4,
        "learned_selection_episodes": 4 * 3 * p["sel_resets"] * 3 * 2,
    }
    ph = {
        "random_eval": n["random_eval_episodes"] * ep_ws / 3600,
        "test_branching": n["test_continuations"] * cont_ws / 3600,
        "dev_branching": n["dev_continuations"] * cont_ws / 3600,
        "learned_selection": n["learned_selection_episodes"] * ep_ws / 3600,
    }
    proc_h = sum(ph.values())
    cores_per_proc = 13 if t["workers"] == 2 else 1
    eval_wall = proc_h / t["workers"]
    return {
        **n, "process_h": {k: round(v, 1) for k, v in ph.items()}, "process_h_total": round(proc_h, 1),
        "cpu_core_h": round(proc_h * cores_per_proc, 1),
        "eval_wall_h": round(eval_wall, 1),
        "eval_gpu_h": round(eval_wall * (1.0 if t["workers"] == 2 else 0.8), 1),  # B1K: 1 GPU/process share; RoboCasa: 1 server GPU ~80% busy
        "train_gpu_h": round(18 * TRAIN_H, 1),
        "train_wall_h_one_gpu": round(18 * TRAIN_H, 1),
        "storage_gb": round(18 * CKPT_GB + (n["random_eval_episodes"] + n["learned_selection_episodes"]) * 15e-6
                            + (n["test_continuations"] + n["dev_continuations"]) * 4e-6
                            + 4 * 3 * (p["test_parents"] + p["dev_parents"]) * 0.5e-3, 1),
    }


def main():
    out = {"inputs": {"train_h_per_run": round(TRAIN_H, 2), "tasks": TASKS}, "protocols": {}}
    for pname, p in PROTOCOLS.items():
        per = {k: task_cost(t, p) for k, t in TASKS.items()}
        rc = ["pick_egg", "wipe_countertop"]
        tot = lambda keys, f: round(sum(per[k][f] for k in keys), 1)  # noqa: E731
        allk = list(TASKS)
        # Schedule: one training GPU runs continuously (54 runs); the second GPU hosts the eval inference server /
        # EGL rendering (RoboCasa) or one of the two B1K processes. CPU pool (<=32) is used by one task at a time.
        train_wall = tot(allk, "train_wall_h_one_gpu")
        eval_wall = tot(allk, "eval_wall_h")
        out["protocols"][pname] = {
            "per_task": per,
            "robocasa_pick_wipe": {"train_gpu_h": tot(rc, "train_gpu_h"), "eval_gpu_h": tot(rc, "eval_gpu_h"),
                                   "cpu_core_h": tot(rc, "cpu_core_h"), "eval_wall_h": tot(rc, "eval_wall_h"),
                                   "storage_gb": tot(rc, "storage_gb")},
            "all_three": {"train_gpu_h": train_wall, "eval_gpu_h": tot(allk, "eval_gpu_h"), "cpu_core_h": tot(allk, "cpu_core_h"),
                          "eval_wall_h": eval_wall, "storage_gb": tot(allk, "storage_gb"),
                          "wall_h_train_1gpu_eval_overlapped": round(max(train_wall, eval_wall) + 7.2, 1),
                          "wall_h_if_2_gpus_train_then_eval": round(train_wall / 2 + eval_wall, 1)},
        }
    json.dump(out, open(OUT, "w"), indent=1)
    for pname, v in out["protocols"].items():
        print(pname, json.dumps(v["all_three"]), json.dumps(v["robocasa_pick_wipe"]))
        for k, t in v["per_task"].items():
            print("   ", k, t["process_h"], t["eval_wall_h"], t["cpu_core_h"], t["storage_gb"])


if __name__ == "__main__":
    main()
