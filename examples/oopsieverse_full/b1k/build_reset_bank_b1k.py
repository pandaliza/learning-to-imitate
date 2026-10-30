"""Build one shard of the frozen B1K reset bank (b1k_diag_protocol.md §4).

python build_reset_bank_b1k.py TASK POOL SEED0 N OUT_DIR [--determinism K]
For each seed in SEED0..SEED0+N-1: official seeded teleop reset_env; snapshot (serialized state .npy + json).
--determinism K: re-run the first K seeds at the end (after other resets) and record whether the state is
bit-identical (decides whether evaluation can rely on seed regeneration alone; the evaluator always loads the
bank state after regenerating anyway).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1k_common as C  # noqa: E402

C.kit_args_from_env()
task, pool, seed0, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
det_k = int(sys.argv[sys.argv.index("--determinism") + 1]) if "--determinism" in sys.argv else 0
os.makedirs(out, exist_ok=True)
import numpy as np  # noqa: E402

t0 = time.time()
env, cfg, mod, tb = C.build_env(task, cameras=False)
t_build = time.time() - t0
import omnigibson as og  # noqa: E402

recs = []
for s in range(seed0, seed0 + n):
    jp = os.path.join(out, f"{task}_{pool}_{s}.json")
    if os.path.exists(jp):
        recs.append(json.load(open(jp)))
        continue
    t = time.time()
    C.seeded_reset(env, cfg, mod, tb, s)
    rec, st = C.reset_snapshot(env, task)
    rec.update(task=task, pool=pool, seed=s, reset_s=time.time() - t, health=C.entity_health(env))
    np.save(os.path.join(out, f"{task}_{pool}_{s}.npy"), st)
    rec["npy_sha256"] = C.sha256_file(os.path.join(out, f"{task}_{pool}_{s}.npy"))
    json.dump(rec, open(jp + ".tmp", "w"), indent=1)
    os.replace(jp + ".tmp", jp)
    recs.append(rec)
    print("RESET", s, rec["state_sha256"][:12], f"{rec['reset_s']:.1f}s", flush=True)
det = []
for s in range(seed0, seed0 + det_k):
    C.seeded_reset(env, cfg, mod, tb, s)
    rec, st = C.reset_snapshot(env, task)
    ref = np.load(os.path.join(out, f"{task}_{pool}_{s}.npy"))
    det.append({"seed": s, "identical": rec["state_sha256"] == json.load(open(os.path.join(out, f"{task}_{pool}_{s}.json")))["state_sha256"],
                "max_abs_diff": float(np.abs(ref - st).max()) if ref.shape == st.shape else None})
    print("DET", det[-1], flush=True)
json.dump({"task": task, "pool": pool, "seed0": seed0, "n": n, "t_build_s": t_build, "determinism": det,
           "t_total_s": time.time() - t0}, open(os.path.join(out, f"SHARD_{task}_{pool}_{seed0}.json"), "w"), indent=1)
og.shutdown()
