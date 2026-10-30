"""Dev-gate evaluator for the B1K add_firewood all-60 matched-condition check (protocol addendum 2026-10-30, section 5).

Calls the UNMODIFIED eval_b1k.run_episode / apply_reset / B1KPolicy (same env, frame stack, Euler 16, chunk 8, timeout
550, +20 steps, predicates, atomic records). The only addition: a policy subclass keeps the chunks it emits, and each
record gets the EXECUTED action sequence summary (gripper command a[-1] per executed step; "open" := a[-1] > 0; first
open step counted 1-based as the env step at which the command was issued).

  python eval_b1k_all60.py --run-dir RUN --ckpt step_30000.pth --task add_firewood --pool dev --ids 0-29 \
      --rollout-seed 0 --out OUT
Output records: OUT/dev{i:03d}__r{seed}.json (atomic; resumable: finished records are skipped).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "matrix"))
import b1k_common as C  # noqa: E402
import eval_b1k as E  # noqa: E402

OPEN_THRESHOLD = 0.0  # open := a[-1] > 0 (see all60_pool_stats.py)


class RecPolicy(E.B1KPolicy):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.chunks = []

    def act(self, hist, gen):
        a = super().act(hist, gen)
        self.chunks.append(a)
        return a


def code_key():
    h = hashlib.sha256()
    for p in (__file__, E.__file__, C.__file__):
        h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


def summarize_actions(pol, steps):
    import numpy as np
    if not pol.chunks or steps <= 0:
        return {"grip_cmd": [], "open_steps": [], "n_open_steps": 0, "first_open_step": None, "n_actions_recorded": 0}
    acts = np.concatenate([np.asarray(c) for c in pol.chunks], axis=0)[:steps]
    g = acts[:, -1]
    op = np.where(g > OPEN_THRESHOLD)[0]
    return {"grip_cmd": [round(float(x), 4) for x in g], "open_steps": [int(i) + 1 for i in op],
            "n_open_steps": int(len(op)), "first_open_step": (int(op[0]) + 1) if len(op) else None,
            "n_actions_recorded": int(acts.shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="step_30000.pth")
    ap.add_argument("--task", required=True)
    ap.add_argument("--pool", required=True, choices=["dev", "test"])
    ap.add_argument("--ids", required=True)
    ap.add_argument("--rollout-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = C.kit_args_from_env()
    os.makedirs(a.out, exist_ok=True)
    if a.pool == "test":
        raise SystemExit("test pool is not used by the matched-condition gate (protocol section 4-7)")
    bank, entries = E.load_bank(a.task, a.pool)
    ids = E.parse_ids(a.ids)
    todo = [i for i in ids if not os.path.exists(os.path.join(a.out, f"{a.pool}{i:03d}__r{a.rollout_seed}.json"))]
    if not todo:
        print("NOTHING_TO_DO", flush=True)
        return
    t0 = time.time()
    env, cfg, mod, tb = C.build_env(a.task, cameras=True)
    t_build = time.time() - t0
    import torch
    torch.set_num_threads(2)
    pol = RecPolicy(a.run_dir, a.ckpt, f"cuda:{g}")
    ck_sha = C.sha256_file(os.path.join(a.run_dir, "ckpt", a.ckpt))[:16]
    for i in todo:
        out = os.path.join(a.out, f"{a.pool}{i:03d}__r{a.rollout_seed}.json")
        if os.path.exists(os.path.join(a.out, "STOP")):
            print("STOP file", flush=True)
            break
        pol.chunks = []
        try:
            rec = E.run_episode(env, cfg, mod, tb, pol, a.task, entries[i], a.rollout_seed)
            rec["executed_actions"] = summarize_actions(pol, rec["steps"])
        except Exception:
            rec = {"infra_error": traceback.format_exc()[-3000:], "reset": {"pool": a.pool, "index": i},
                   "rollout_seed": a.rollout_seed, "task": a.task}
        rec.update(run_dir=a.run_dir, arm=pol.meta.get("arm"), run_id=pol.meta.get("run_id"), ckpt=a.ckpt,
                   ckpt_sha16=ck_sha, isaac_gpu=g, t_env_build_s=t_build, eval_wrapper_key=code_key())
        json.dump(rec, open(out + ".tmp", "w"), indent=1, default=float)
        os.replace(out + ".tmp", out)
        print("EP", i, rec.get("outcome", "INFRA"), rec.get("steps"), f"{rec.get('wall_s', 0):.0f}s", flush=True)
        if "infra_error" in rec:
            print(rec["infra_error"][-800:], flush=True)
    print("EVAL_DONE", flush=True)
    import omnigibson as og
    og.shutdown()


if __name__ == "__main__":
    main()
