"""Evaluation entry point for the matched-condition baseline check (spec addendum 2026-10-27).

Same machinery and record format as eval_wipe_v2.py (frozen rollout_eval.run_episode with exec_horizon 8, 1250-step
timeout, counter-label alias, batched inference server, budgeted multi-GPU workers on the parity-passed GPUs, atomic
records, STOP file, resumable). Pools:
  P-demo60    pools/pools_demo60_v1.json   exact starts of all 60 demos; apply_reset + observable refresh
  P-layout60  pools/pools_layout60_v1.json sticky-ep_meta resets, 20 per layout x 7 layouts
  P-random    the frozen eval_dev[0:50] bank (pools_v1.json pointer)
eval_wipe_v2.py / wv2_common.py are not edited (their hashes key every existing run); like eval_wipe_v2_demo.py this
file widens eval_wipe_v2.main's pool check, resolves the new pools (sha256-verified), runs workers with the observable
refresh after state restores, and adds the pool-file content hashes, this file's sha256 and the refresh flag to the key.
Records: runs/wipe_v2/all60/eval/<run_id>/<pool>__<ckpt>__<key>/<reset>__r<seed>.json. Run dirs under
runs/wipe_v2/all60/train/ need the usual meta.json + ckpt/<ckpt> (A-MIX: plain CFMPolicy; I-PATH: meta["intent"]).

  source examples/oopsieverse_full/baseline/envsetup.sh; cd examples/oopsieverse_full/wipe_v2
  nohup nice -n 10 $PY eval_wipe_v2_all60.py --run-dir RUN --pools P-demo60,P-layout60,P-random --rollout-seeds 0,1,2 \
      --max-workers 32 --label NAME > /edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/logs/NAME.log 2>&1 &
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
import build_pools_all60 as BA  # noqa: E402

OLD = 'assert p in ("P-train", "P-devlayout", "P-random"), p'
NEW = 'assert p in ("P-demo60", "P-layout60", "P-random"), p'
ROOT = os.path.join(C.V2_RUNS, "all60", "eval")
_ORIG_APPLY = E.RE.apply_reset


def _refresh(env, reset):
    out = _ORIG_APPLY(env, reset)
    if reset.get("state") is None:
        return out
    sb = env.sim.get_state().flatten().copy()
    env._update_observables(force=True)
    out = env.get_observations()
    assert np.array_equal(sb, env.sim.get_state().flatten()), "observable refresh changed the sim state"
    return out


def all60_worker(slot, gpu, specs, req_qs, resp_q, job_q, res_q):
    E.RE.apply_reset = _refresh
    orig = E._episode

    def episode(env, pol, job, spec, g):
        rec = orig(env, pol, job, spec, g)
        ks = spec["pools"][job["pool"]]["key_src"]
        if ks.get("pool_file_content_sha256"):
            rec["pools_file_content_sha256"] = ks["pool_file_content_sha256"]
        rec["obs_refresh_after_restore"] = job["reset"].get("state") is not None
        rec["demo_len"] = job["reset"].get("demo_len")
        rec["evaluator"] = "wipe_v2/eval_wipe_v2_all60.py (counter-label alias, matched-condition check)"
        return rec
    E._episode = episode
    E.worker_main(slot, gpu, specs, req_qs, resp_q, job_q, res_q)


def main():
    files = {"P-demo60": BA.DEMO60_FILE, "P-layout60": BA.LAYOUT60_FILE}
    docs = {}

    def doc(pool):  # loaded (and hash-checked) only for the pools actually requested
        if pool not in docs:
            docs[pool] = BA.load_pool_doc(files[pool])
        return docs[pool]
    me = C.sha256_file(os.path.abspath(__file__))
    code = {f: C.sha256_file(os.path.join(HERE, f)) for f in ("build_pools_all60.py", "build_pool_demo.py")}
    orig_resets = C.pool_resets

    def pool_resets(pool, pools=None):
        if pool == "P-demo60":
            return BA.demo60_resets()
        if pool == "P-layout60":
            return BA.layout60_resets()
        return orig_resets(pool, pools)
    C.pool_resets = pool_resets
    orig_spec = E.run_spec

    def run_spec(run_dir, ckpt, pools_doc, pool_names):
        sp = orig_spec(run_dir, ckpt, pools_doc, pool_names)
        for pool, v in sp["pools"].items():
            ks = dict(v["key_src"], all60_entry_sha256=me, pool_code_sha256=code, obs_refresh_after_restore=True)
            if pool in files:
                ks["pool_file_content_sha256"] = doc(pool)["content_sha256"]
            key = hashlib.sha256(json.dumps(ks, sort_keys=True).encode()).hexdigest()[:12]
            v["out_dir"] = os.path.join(os.environ.get("WV2_ALL60_ROOT_SMOKE_ONLY") or ROOT, sp["run_id"],
                                        f"{pool}__{os.path.splitext(ckpt)[0]}__{key}")
            v["eval_key"], v["key_src"] = key, ks
        return sp
    E.run_spec = run_spec
    E.worker_main = all60_worker
    src = inspect.getsource(E.main)
    assert src.count(OLD) == 1, "eval_wipe_v2.main changed: pool check not found"
    ns = E.__dict__
    exec(compile(src.replace(OLD, NEW), E.__file__ + "::main[all60]", "exec"), ns)
    return ns["main"]()


if __name__ == "__main__":
    main()
