"""Rollout-time intent ablation on the all-60 I-PATH checkpoints (spec addendum 2026-10-29 23:31 UTC).

Entry point = eval_wipe_v2_all60.py's CLI plus

    --intent-ablation {shuffled,zero}      (required; the sampled rows are the existing all60 records)

Pool: P-layout60 only (pools/pools_layout60_v1.json, frozen). Alias on, timeout 1250, exec horizon 8.

Composition (no hashed file is edited):
  * eval_wipe_v2.py  : frozen supervisor (main, pick_infer_gpu, build_jobs, _episode, worker plumbing).
  * eval_wipe_v2_all60.py : the all60 observable refresh after restores (_refresh) and the pool-file hashes.
  * eval_wipe_v2_ablate.py : the ablation inference server (ablation_server) and the ablation worker
    (ablation_worker: per-request reset ids, provenance tallies, intent_ablation fields).
Worker chain in each process: this file's episode wrapper (pool-file fields) is installed as E._episode, then
AB.ablation_worker wraps it with the ablation fields and runs E.worker_main. In the spawned worker E.worker_main is
the frozen original (this file's main() overrides it only in the supervisor process).

Records: runs/wipe_v2/all60/ablation/<run_id>/<pool>__<mode>__step_30000__<key>/<reset>__r<seed>.json
The key hashes the all60 key source, the ablation mode, eval_wipe_v2_ablate.py, this file, and the zero token.
Smoke runs: env WV2_ALL60_ABL_ROOT_SMOKE_ONLY redirects the root (the key is unchanged).

  source examples/oopsieverse_full/baseline/envsetup.sh; cd examples/oopsieverse_full/wipe_v2
  nohup nice -n 10 $PY eval_wipe_v2_ablate_all60.py --intent-ablation shuffled --run-dir RUN --pools P-layout60 \
      --rollout-seeds 0,1,2 --max-workers 24 --total-workers 24 --render-gpus 3 --infer-gpu 3 --label NAME > LOG 2>&1 &
"""
import hashlib
import inspect
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_wipe_v2 as E  # noqa: E402
import wv2_common as C  # noqa: E402
import eval_wipe_v2_ablate as AB  # noqa: E402
import eval_wipe_v2_all60 as A60  # noqa: E402
import build_pools_all60 as BA  # noqa: E402

MODES = ("shuffled", "zero")
OLD = 'assert p in ("P-train", "P-devlayout", "P-random"), p'
NEW = 'assert p in ("P-layout60",), p'
ROOT = os.environ.get("WV2_ALL60_ABL_ROOT_SMOKE_ONLY") or os.path.join(C.V2_RUNS, "all60", "ablation")
_ORIG_RUN_SPEC = E.run_spec      # captured before any patching (this module never patches it at import)
_BASE_WORKER = E.worker_main     # frozen worker; only reached inside spawned workers (E.worker_main is patched in main)


def all60_ablation_worker(slot, gpu, specs, req_qs, resp_q, job_q, res_q):
    E.RE.apply_reset = A60._refresh
    orig = E._episode

    def episode(env, pol, job, spec, g):
        rec = orig(env, pol, job, spec, g)
        ks = spec["pools"][job["pool"]]["key_src"]
        if ks.get("pool_file_content_sha256"):
            rec["pools_file_content_sha256"] = ks["pool_file_content_sha256"]
        rec["obs_refresh_after_restore"] = job["reset"].get("state") is not None
        rec["demo_len"] = job["reset"].get("demo_len")
        rec["evaluator_file"] = "wipe_v2/eval_wipe_v2_ablate_all60.py (all60 P-layout60, intent ablation)"
        return rec
    E._episode = episode
    # AB.ablation_worker wraps E._episode (ours) with the ablation fields and calls E.worker_main (the frozen one
    # in this spawned process), so there is no recursion.
    AB.ablation_worker(slot, gpu, specs, req_qs, resp_q, job_q, res_q)


def make_run_spec(mode):
    me = C.sha256_file(os.path.abspath(__file__))
    abl = C.sha256_file(os.path.abspath(AB.__file__))
    a60 = C.sha256_file(os.path.abspath(A60.__file__))
    code = {f: C.sha256_file(os.path.join(HERE, f)) for f in ("build_pools_all60.py", "build_pool_demo.py")}

    def run_spec(run_dir, ckpt, pools_doc, pool_names):
        sp = _ORIG_RUN_SPEC(run_dir, ckpt, pools_doc, pool_names)
        meta = json.load(open(os.path.join(run_dir, "meta.json")))
        it = meta.get("intent") or {}
        assert it.get("mode") == "condition", f"{run_dir}: intent ablation needs a condition-mode intent arm"
        zt = AB.zero_token(run_dir)
        zt_sha = hashlib.sha256(json.dumps(zt["token"]).encode()).hexdigest()
        for pool, v in sp["pools"].items():
            ks = dict(v["key_src"], all60_entry_sha256=a60, pool_code_sha256=code, obs_refresh_after_restore=True,
                      intent_ablation=mode, ablation_code_sha256=abl, all60_ablation_entry_sha256=me,
                      zero_token_sha256=zt_sha)
            if pool == "P-layout60":
                ks["pool_file_content_sha256"] = BA.load_pool_doc(BA.LAYOUT60_FILE)["content_sha256"]
            key = hashlib.sha256(json.dumps(ks, sort_keys=True).encode()).hexdigest()[:12]
            v["out_dir"] = os.path.join(ROOT, sp["run_id"], f"{pool}__{mode}__{os.path.splitext(ckpt)[0]}__{key}")
            v["eval_key"], v["key_src"] = key, ks
        return sp
    return run_spec


def main():
    argv = sys.argv[1:]
    assert "--intent-ablation" in argv, "--intent-ablation {shuffled,zero} is required"
    i = argv.index("--intent-ablation")
    mode = argv[i + 1]
    del argv[i:i + 2]
    assert mode in MODES, mode
    sys.argv = [sys.argv[0]] + argv
    os.environ[AB.ENV_MODE] = mode  # inherited by the spawned server and worker processes

    # pool resets: P-layout60 from the frozen file (hash-checked by BA.layout60_resets)
    orig_pool_resets = C.pool_resets

    def pool_resets(pool, pools=None):
        if pool == "P-layout60":
            return BA.layout60_resets()
        return orig_pool_resets(pool, pools)
    C.pool_resets = pool_resets

    E.run_spec = make_run_spec(mode)
    E.inference_server = AB.ablation_server
    E.worker_main = all60_ablation_worker
    src = inspect.getsource(E.main)
    assert src.count(OLD) == 1, "eval_wipe_v2.main changed: pool check not found"
    ns = E.__dict__
    exec(compile(src.replace(OLD, NEW), E.__file__ + "::main[all60-ablate]", "exec"), ns)
    return ns["main"]()


if __name__ == "__main__":
    main()
