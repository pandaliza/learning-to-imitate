"""P-demo evaluation entry point (spec addendum 2026-10-26). Same CLI, machinery and record format as eval_wipe_v2.py
(frozen run_episode, counter-label alias, batched inference server, budgeted multi-GPU workers, atomic records, STOP
file), with the pool P-demo (pools/pools_demo_v1.json, build_pool_demo.py) added:

  * eval_wipe_v2.py / wv2_common.py are NOT edited (both are hashed into every eval key, and spawned workers re-import
    them from disk). This file (a) wraps wv2_common.pool_resets so "P-demo" resolves to build_pool_demo.load_demo_resets
    (sha256-verified demo ep_meta / model_file / states[0]); (b) runs eval_wipe_v2.main with its pool-name check
    extended by "P-demo" (the function source is re-used verbatim apart from that one tuple; checked below);
    (c) adds to the run key: pools_demo content sha256, this file's sha256, build_pool_demo.py sha256 and
    obs_refresh=True; (d) runs workers through demo_worker, which wraps rollout_eval.apply_reset with
    build_pool_demo.apply_reset_refresh (observable refresh after a state restore; sim state unchanged) and adds
    "obs_refresh_after_restore": true to every record.
Only --pools P-demo is accepted here (other pools: use eval_wipe_v2.py).

  nohup nice -n 10 $PY eval_wipe_v2_demo.py --run-dir RUN --pools P-demo --rollout-seeds 0,1,2 --max-workers 32 \
      --label NAME > LOG 2>&1 &
"""
import inspect
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_wipe_v2 as E  # noqa: E402
import wv2_common as C  # noqa: E402
import build_pool_demo as BPD  # noqa: E402

OLD = 'assert p in ("P-train", "P-devlayout", "P-random"), p'
NEW = 'assert p in ("P-demo",), p'


def demo_worker(slot, gpu, specs, req_qs, resp_q, job_q, res_q):
    E.RE.apply_reset = _refresh
    orig = E._episode

    def episode(env, pol, job, spec, g):
        rec = orig(env, pol, job, spec, g)
        rec["obs_refresh_after_restore"] = True
        rec["evaluator"] = "wipe_v2/eval_wipe_v2_demo.py (counter-label alias, P-demo, observable refresh after restore)"
        rec["demo_len"] = job["reset"].get("demo_len")
        rec["pools_file_content_sha256"] = spec["pools"][job["pool"]]["key_src"]["pools_demo_content_sha256"]
        return rec
    E._episode = episode
    E.worker_main(slot, gpu, specs, req_qs, resp_q, job_q, res_q)


_ORIG_APPLY = E.RE.apply_reset


def _refresh(env, reset):
    import numpy as np
    out = _ORIG_APPLY(env, reset)
    if reset.get("state") is None:
        return out
    sb = env.sim.get_state().flatten().copy()
    env._update_observables(force=True)
    out = env.get_observations()
    assert np.array_equal(sb, env.sim.get_state().flatten()), "observable refresh changed the sim state"
    return out


def main():
    pd = json.load(open(BPD.POOL_DEMO_FILE))
    me = C.sha256_file(os.path.abspath(__file__))
    bpd = C.sha256_file(os.path.join(HERE, "build_pool_demo.py"))
    orig_resets = C.pool_resets

    def pool_resets(pool, pools=None):
        if pool == "P-demo":
            return BPD.load_demo_resets(pd)
        return orig_resets(pool, pools)
    C.pool_resets = pool_resets
    orig_spec = E.run_spec

    def run_spec(run_dir, ckpt, pools_doc, pool_names):
        sp = orig_spec(run_dir, ckpt, pools_doc, pool_names)
        for pool, v in sp["pools"].items():
            ks = dict(v["key_src"], pools_demo_content_sha256=pd["content_sha256"], demo_entry_sha256=me,
                      build_pool_demo_sha256=bpd, obs_refresh_after_restore=True)
            key = __import__("hashlib").sha256(json.dumps(ks, sort_keys=True).encode()).hexdigest()[:12]
            v["out_dir"] = os.path.join(os.path.dirname(v["out_dir"]), f"{pool}__{os.path.splitext(ckpt)[0]}__{key}")
            v["eval_key"], v["key_src"] = key, ks
        return sp
    E.run_spec = run_spec
    E.worker_main = demo_worker
    src = inspect.getsource(E.main)
    assert src.count(OLD) == 1, "eval_wipe_v2.main changed: pool check not found"
    ns = E.__dict__
    exec(compile(src.replace(OLD, NEW), E.__file__ + "::main[P-demo]", "exec"), ns)
    return ns["main"]()


if __name__ == "__main__":
    main()
