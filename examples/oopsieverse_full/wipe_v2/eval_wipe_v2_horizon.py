"""Execution-horizon sweep entry point (spec addendum 2026-10-26: execution-horizon sweep on existing checkpoints).

Same machinery and record format as eval_wipe_v2.py (frozen rollout_eval.run_episode, counter-label alias, batched
inference server, budgeted multi-GPU workers, atomic records, STOP file), plus

    --exec-horizon H     actions executed per predicted 8-step chunk before re-observing / re-planning (1..8)

The frozen run_episode already implements receding-horizon execution: it executes chunk[:H] (each step appends the new
frame to the frame_stack-2 history), then calls policy.act again, which draws a fresh x0 ~ N(0,I) (and, for I-PATH, z
after x0) from the same per-episode generator (seed rollout_seed*1000003 + reset seed); the loop ends at the 1250-env-step
timeout or 10 steps after the first success, independent of H. eval_wipe_v2._episode passes exec_horizon=8; this entry
point's workers wrap rollout_eval.run_episode to pass H instead and to log the executed actions (smoothness proxy:
mean over steps t >= 1 of ||a_t - a_{t-1}||_2 on the raw 12-D executed action; also reported per arm-dim group).
eval_wipe_v2.py / wv2_common.py are not edited (their hashes key every existing run). P-demo resolves through
build_pool_demo (frozen pools_demo_v1.json) with the observable refresh after the state restore, as in
eval_wipe_v2_demo.py. Run key adds exec_horizon (replacing the default 8), this file's sha256 and, for P-demo, the
P-demo pool / refresh fields. Records: runs/wipe_v2/horizon/<run_id>/<pool>__<ckpt>__h<H>__<key>/ (kept apart from
runs/wipe_v2/eval so the main summaries never pick them up); every record carries exec_horizon and smoothness fields.

  nohup nice -n 10 $PY eval_wipe_v2_horizon.py --exec-horizon 4 --run-dir RUN --pools P-demo,P-devlayout \
      --rollout-seeds 0,1 --max-workers 32 --label NAME > LOG 2>&1 &
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
import build_pool_demo as BPD  # noqa: E402

OLD = 'assert p in ("P-train", "P-devlayout", "P-random"), p'
NEW = 'assert p in ("P-train", "P-devlayout", "P-random", "P-demo"), p'
HZ_ROOT = os.path.join(C.V2_RUNS, "horizon")
ENV_H = "WV2_EXEC_HORIZON"
_ORIG_APPLY = E.RE.apply_reset
_ORIG_RUN = E.RE.run_episode


def _refresh(env, reset):
    out = _ORIG_APPLY(env, reset)
    if reset.get("state") is None:
        return out
    sb = env.sim.get_state().flatten().copy()
    env._update_observables(force=True)
    out = env.get_observations()
    assert np.array_equal(sb, env.sim.get_state().flatten()), "observable refresh changed the sim state"
    return out


def _run_episode_h(env, policy, task, reset, rollout_seed, exec_horizon=8, record_chunks=False):
    H = int(os.environ[ENV_H])
    acts = []
    inner = env.step

    def step(a):
        acts.append(np.asarray(a, np.float64).copy())
        return inner(a)
    env.step = step
    try:
        rec = _ORIG_RUN(env, policy, task, reset, rollout_seed, exec_horizon=H, record_chunks=record_chunks)
    finally:
        env.step = inner
    A = np.asarray(acts)
    d = np.linalg.norm(np.diff(A, axis=0), axis=1) if len(A) > 1 else np.zeros(0)
    rec["exec_horizon"] = H
    rec["smoothness_mean_action_change_norm"] = float(d.mean()) if len(d) else None
    rec["smoothness_mean_action_change_norm_arm6"] = float(np.linalg.norm(np.diff(A[:, :6], axis=0), axis=1).mean()) if len(A) > 1 else None
    rec["n_policy_calls"] = rec.get("chunks")
    return rec


def horizon_worker(slot, gpu, specs, req_qs, resp_q, job_q, res_q):
    E.RE.apply_reset = _refresh
    E.RE.run_episode = _run_episode_h
    orig = E._episode

    def episode(env, pol, job, spec, g):
        rec = orig(env, pol, job, spec, g)
        ks = spec["pools"][job["pool"]]["key_src"]
        if job["pool"] == "P-demo":
            rec["obs_refresh_after_restore"] = True
            rec["pools_file_content_sha256"] = ks["pools_demo_content_sha256"]
            rec["demo_len"] = job["reset"].get("demo_len")
        rec["evaluator"] = f"wipe_v2/eval_wipe_v2_horizon.py (counter-label alias, exec_horizon={os.environ[ENV_H]})"
        return rec
    E._episode = episode
    E.worker_main(slot, gpu, specs, req_qs, resp_q, job_q, res_q)


def main():
    argv = sys.argv[1:]
    assert "--exec-horizon" in argv, "--exec-horizon H is required"
    i = argv.index("--exec-horizon")
    H = int(argv[i + 1])
    del argv[i:i + 2]
    assert 1 <= H <= 8, H
    sys.argv = [sys.argv[0]] + argv
    os.environ[ENV_H] = str(H)
    pd = json.load(open(BPD.POOL_DEMO_FILE))
    me = C.sha256_file(os.path.abspath(__file__))
    bpd = C.sha256_file(os.path.join(HERE, "build_pool_demo.py"))
    orig_resets = C.pool_resets

    def pool_resets(pool, pools=None):
        return BPD.load_demo_resets(pd) if pool == "P-demo" else orig_resets(pool, pools)
    C.pool_resets = pool_resets
    orig_spec = E.run_spec

    def run_spec(run_dir, ckpt, pools_doc, pool_names):
        sp = orig_spec(run_dir, ckpt, pools_doc, pool_names)
        for pool, v in sp["pools"].items():
            ks = dict(v["key_src"], exec_horizon=H, horizon_entry_sha256=me)
            if pool == "P-demo":
                ks.update(pools_demo_content_sha256=pd["content_sha256"], build_pool_demo_sha256=bpd, obs_refresh_after_restore=True)
            key = hashlib.sha256(json.dumps(ks, sort_keys=True).encode()).hexdigest()[:12]
            v["out_dir"] = os.path.join(os.environ.get("WV2_HORIZON_ROOT_SMOKE_ONLY") or HZ_ROOT, sp["run_id"],
                                        f"{pool}__{os.path.splitext(ckpt)[0]}__h{H}__{key}")
            v["eval_key"], v["key_src"] = key, ks
        return sp
    E.run_spec = run_spec
    E.worker_main = horizon_worker
    src = inspect.getsource(E.main)
    assert src.count(OLD) == 1, "eval_wipe_v2.main changed: pool check not found"
    ns = E.__dict__
    exec(compile(src.replace(OLD, NEW), E.__file__ + "::main[horizon]", "exec"), ns)
    return ns["main"]()


if __name__ == "__main__":
    main()
