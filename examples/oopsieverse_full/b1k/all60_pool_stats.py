"""Training-pool statistics for the B1K add_firewood all-60 check (protocol addendum 2026-10-30, section 6).

Open-gripper command := gripper action a[-1] > 0 (demos: -1 through the carry, +1 at the final step = release).
Training samples: t in [1, T-8] (official PlaybackDataset: t from frame_stack-1 to T-action_chunk, inclusive). Chunk contains an open command iff any
a[t:t+8, -1] > 0. The 40-demo figures of b1k_results.md are reproduced first (n_train_samples 5945, median 1 open
step per demo, 0.7% of chunks) before the 60-demo pool is reported.
  python all60_pool_stats.py  ->  reports/b1k/all60_pool_stats.json (+ printed table)
"""
import csv
import json
import os
import statistics

import h5py
import numpy as np

STORE = "/edrive1/lizad/oopsieverse_full_study"
PARTS = f"{STORE}/data/b1k_playback/parts"
REP = f"{STORE}/reports/b1k"
CHUNK = 8


def load_actions(member):
    t, c, d = member.split("/")
    with h5py.File(os.path.join(PARTS, f"{t}__{c}__{d}.hdf5"), "r") as f:
        return np.asarray(f[f"data/{d}/action"], np.float64)


def pool_stats(members):
    n_samples = n_open_chunks = 0
    open_steps, first_open, open_any = [], [], 0
    for m in members:
        a = load_actions(m)
        T = len(a)
        g = a[:, -1]
        n_samples += max(T - CHUNK, 0)
        is_open = g > 0
        k = int(is_open.sum())
        open_steps.append(k)
        open_any += int(k > 0)
        first_open.append(int(np.argmax(is_open)) if k else None)
        for t in range(1, T - CHUNK + 1):
            n_open_chunks += int(is_open[t:t + CHUNK].any())
    return {"n_demos": len(members), "n_train_samples": n_samples, "open_chunks": n_open_chunks,
            "open_chunk_fraction": n_open_chunks / max(n_samples, 1),
            "demos_with_open": open_any, "open_steps_per_demo_median": statistics.median(open_steps),
            "open_steps_per_demo_min": min(open_steps), "open_steps_per_demo_max": max(open_steps),
            "first_open_step_per_demo_median": statistics.median([x for x in first_open if x is not None]) if open_any else None}


def main():
    sp = json.load(open(f"{REP}/splits_b1k_v1_add_firewood.json"))["tasks"]["add_firewood"]["splits"]["train"]["members"]
    rows = list(csv.DictReader(open(f"{REP}/replay_labels_add_firewood.csv")))
    all60 = [f"add_firewood/{r['cond']}/{r['demo']}" for r in rows]
    assert len(sp) == 40 and len(all60) == 60 and set(sp) <= set(all60)
    out = {"definition": "open := gripper a[-1] > 0; chunk = a[t:t+8], t in [1, T-8]",
           "pool_40_train_split": pool_stats(sp), "pool_60_all": pool_stats(all60)}
    p40 = out["pool_40_train_split"]
    ref = {"n_train_samples": 5945, "open_chunk_fraction_approx": 0.007, "open_steps_per_demo_median": 1}
    out["check_vs_b1k_results"] = {
        "n_train_samples_matches_5945": p40["n_train_samples"] == ref["n_train_samples"],
        "open_chunk_fraction_0.7pct": round(p40["open_chunk_fraction"], 4),
        "median_open_steps_1": p40["open_steps_per_demo_median"] == 1}
    tgt = f"{REP}/all60_pool_stats.json"
    json.dump(out, open(tgt + ".tmp", "w"), indent=1)
    os.replace(tgt + ".tmp", tgt)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
