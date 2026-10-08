"""Matched spatial-shift rollouts on robomimic square-mh-state, both arms, over a frozen case manifest.

Minimal extension of the native evaluator: every episode is run by the unmodified native
`collect_rollouts()` (examples/collect/collect_diversity_rollouts.py, the code behind eval_success_rate.py)
on a one-env adapter whose reset() loads the manifest's stored init state. Native pieces reused as-is:
config/checkpoint loading (`load_config`, `load_model`, which patches the config from the checkpoint), obs
stacking (MultiStepWrapper, obs_steps=2, act_steps=8), normalization, sampling, rotation-6d undo, 400-step
horizon and the success rule (any step with reward > 0 or info["success"]).

Sampling: NFE (`--num-steps`) = 9 for both arms, the primary metric of the native training-time evaluator
(train_robomimic.py: first entry of get_default_step_list("flow")) that selected and labelled both checkpoints
(success62 / success64 are mean_success_9). eval_success_rate.py's default of 1 is NOT used.

Both arms of a case run back-to-back in the same worker, so an interruption never leaves one arm missing.
Records: <out-dir>/<arm>/square_mh_state.jsonl (one line per episode, fsync'd; resume skips completed keys).

  python examples/openpi/oops/square_eval.py --manifest-dir oops_runs/square_manifest_pv1 \
      --out-dir oops_runs/square_episodes_pv1 --workers 24
"""

import argparse
import hashlib
import multiprocessing as mp
import os
import pathlib
import sys
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import oops_common as oc  # noqa: E402

CK = REPO / "checkpoints"
ARMS = {
    # arm label: (checkpoint, hydra overrides)  -- exactly the docs/checkpoints.md mapping
    "A_baseline": (CK / "square_mh_state_flow_mlp_512_h10_seed0_success62.pt", ["task=square_mh_state"]),
    "I_flow_intent": (CK / "square_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success64.pt",
                      ["task=square_mh_state_flow_intent", "+network.arch_variant=flow_intent"]),
}


class OneEnvAdapter:
    """Presents one MultiStepWrapper env with the vector-env API collect_rollouts() expects (batch dim 1).

    reset() = np.random.seed(native env seed) + robomimic reset (reproduces the native seeded reset, including
    the controller's internal references such as the null-space joint posture), then reset_to(stored state),
    which for a shifted case differs from that reset only in the nut's XY. Without the seeded reset the controller
    keeps references from a different random robot pose and trajectories diverge from native after one step
    (caught by square_parity.py)."""

    def __init__(self, env, lowdim, robo):
        self.env, self.lowdim, self.robo = env, lowdim, robo
        self.state = None
        self._obs = None

    def begin(self, state, env_seed=None):
        self.state = state
        self.env_seed = env_seed
        self.steps = 0
        self.success_step = None
        self.policy_failure = None
        self.reward_success = False

    def reset(self, **_):
        if getattr(self, "native_reset", False):
            # parity cohort only (delta=0): the untouched native seeded reset, no reset_to
            self.lowdim.seed(self.env_seed)
            obs, _ = self.env.reset()
            self._obs = obs
            return obs[None], {}
        if self.env_seed is not None:
            np.random.seed(self.env_seed)
        self.robo.reset()
        self.lowdim.init_state = self.state
        obs, _ = self.env.reset()
        self.lowdim.init_state = None
        self._obs = obs
        return obs[None], {}

    def step(self, act_env):
        act = np.asarray(act_env)[0]
        if self.policy_failure or not np.all(np.isfinite(act)):
            self.policy_failure = self.policy_failure or "nonfinite_action"
            z = np.zeros(1)
            return self._obs[None], z, np.ones(1, bool), np.zeros(1, bool), {}
        obs, r, term, trunc, info = self.env.step(act)
        self.steps = len(self.env.reward)
        self._obs = obs
        succ_info = bool(np.asarray(info["success"]).any()) if "success" in info else False
        if (r > 0 or succ_info) and self.success_step is None:
            self.success_step = self.steps  # chunk granularity (<= 8 env steps late)
        out = {"success": [np.asarray(info["success"])]} if "success" in info else {}
        return obs[None], np.array([r]), np.array([term]), np.array([trunc]), out

    def close(self):
        pass


_W = {}


def _worker_init(arm_names, num_steps, native_reset=False):
    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("MUJOCO_GL", "egl")
    from square_build_manifest import load_native, make_single_env
    _W["arms"] = {}
    for arm in arm_names:
        ckpt, ov = ARMS[arm]
        cdr, cfg = load_native(ov + ["task.num_envs=1", "optimization.device=cpu"])
        env, lowdim, robo = make_single_env(cfg.task)
        ad = OneEnvAdapter(env, lowdim, robo)
        ad.begin(None)
        cdr.setup_config_for_env(cfg, ad)
        ad.native_reset = native_reset
        dataset = cdr.make_dataset(cfg.task)
        agent, ip = cdr.load_model(str(ckpt), cfg, dataset, "cpu")
        is_fi = getattr(cfg.network, "arch_variant", "flow_action") == "flow_intent"
        if not is_fi:
            agent.config.optimization.sample_mode = "zero"  # native (default.yaml + eval_success_rate.py)
        _W["arms"][arm] = dict(cdr=cdr, cfg=cfg, ad=ad, dataset=dataset, agent=agent, ip=ip, is_fi=is_fi)
    _W["num_steps"] = num_steps


def _run_case(job):
    c, state, ps, arms = job
    out = []
    for arm in arms:
        w = _W["arms"][arm]
        seed = oc.episode_seed(c["task_name"], c["reset_id"], c["delta_m"], c["angle_idx"], ps)
        t0 = time.time()
        rec, err = None, None
        for _attempt in range(2):  # one deterministic retry of infrastructure failures
            try:
                oc.seed_everything(seed)
                w["ad"].begin(state, c["native_env_seed"])
                _, succ, _, _, _ = w["cdr"].collect_rollouts(
                    w["cfg"], w["agent"], w["dataset"], w["ad"], n_rollouts=1, device="cpu",
                    is_flow_intent=w["is_fi"], intent_predictor=w["ip"], num_steps=_W["num_steps"])
                ad = w["ad"]
                rec = {"success": bool(succ[0]) and ad.policy_failure is None, "success_step": ad.success_step,
                       "steps": ad.steps, "timeout": not bool(succ[0]) and ad.policy_failure is None,
                       "policy_failure": ad.policy_failure, "episode_seed": seed}
                err = None
                break
            except Exception:
                err = traceback.format_exc(limit=4)
        out.append((arm, c, ps, rec, err, round(time.time() - t0, 2)))
    return out


def file_sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(pathlib.Path(p).read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", default="A_baseline,I_flow_intent")
    ap.add_argument("--num-steps", type=int, default=9)
    ap.add_argument("--policy-seeds", default="0")
    ap.add_argument("--deltas", default=None)
    ap.add_argument("--reset-ids", default=None)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--native-reset", action="store_true",
                    help="parity cohort: delta=0 only, untouched native seeded reset; arms stored as <arm>@native")
    args = ap.parse_args()

    arms = args.arms.split(",")
    tag = "@native" if args.native_reset else ""
    mdir = pathlib.Path(args.manifest_dir)
    cases = oc.read_jsonl(mdir / "cases.jsonl")
    npz = np.load(mdir / "init_states.npz")
    states = npz["states"]
    assert list(npz["case_ids"]) == [c["case_id"] for c in cases]
    for c in cases:
        assert oc.state_hash(states[c["state_row"]]) == c["initialized_state_hash"], c["case_id"]
    if args.deltas:
        keep = {round(float(x), 6) for x in args.deltas.split(",")}
        cases = [c for c in cases if round(c["delta_m"], 6) in keep]
    if args.reset_ids:
        keep = {int(x) for x in args.reset_ids.split(",")}
        cases = [c for c in cases if c["reset_id"] in keep]
    seeds = [int(x) for x in args.policy_seeds.split(",")]

    ev_hash = file_sha([__file__, HERE / "oops_common.py", HERE / "square_build_manifest.py",
                        REPO / "examples/collect/collect_diversity_rollouts.py"])
    ck_hash = {a: oc.file_fingerprint(str(ARMS[a][0]), full=True) for a in arms}
    out = pathlib.Path(args.out_dir)
    done = {}
    for a in arms:
        (out / (a + tag)).mkdir(parents=True, exist_ok=True)
        done[a] = {r["result_key"] for r in oc.read_jsonl(out / (a + tag) / "square_mh_state.jsonl")}
        (out / (a + tag) / f"provenance_{ev_hash}.json").write_text(__import__("json").dumps(
            {"arm": a, "checkpoint": str(ARMS[a][0]), "checkpoint_hash": ck_hash[a], "overrides": ARMS[a][1],
             "num_steps": args.num_steps, "evaluator_hash": ev_hash, "manifest": str(mdir.resolve()),
             "device": "cpu", "native_rollout_fn": "collect_diversity_rollouts.collect_rollouts"}, indent=2))

    def rkey(a, cid, ps):
        return f"{cid}#{a}#{ck_hash[a]}#{ev_hash}#nfe{args.num_steps}#s{ps}"

    jobs = []
    for c in cases:
        for ps in seeds:
            todo = [a for a in arms if rkey(a, c["case_id"], ps) + tag not in done[a]]
            if todo:
                jobs.append((c, np.asarray(states[c["state_row"]]), ps, todo))
    if args.max_cases is not None:
        jobs = jobs[:args.max_cases]
    print(f"{len(cases)} cases x {len(seeds)} seeds; {len(jobs)} jobs to run on {args.workers} workers "
          f"(evaluator {ev_hash})", flush=True)
    if not jobs:
        return
    t0 = time.time()
    n = 0
    ctx = mp.get_context("spawn")
    if args.native_reset:
        assert all(c["delta_m"] == 0.0 for c, *_ in jobs), "--native-reset is only defined for delta=0"
    with ctx.Pool(args.workers, initializer=_worker_init, initargs=(arms, args.num_steps, args.native_reset)) as pool:
        for res in pool.imap_unordered(_run_case, jobs, chunksize=1):
            for arm, c, ps, rec, err, wall in res:
                row = {"result_key": rkey(arm, c["case_id"], ps) + tag, "cohort_id": f"{oc.EXPERIMENT}/square/{oc.CONFIG_VERSION}",
                       "case_id": c["case_id"], "arm": arm, "policy_seed": ps, "task_name": c["task_name"],
                       "reset_id": c["reset_id"], "delta_m": c["delta_m"], "angle_idx": c["angle_idx"],
                       "angle_rad": c["angle_rad"], "initialized_state_hash": c["initialized_state_hash"],
                       "validity_status": c["validity_status"], "validity_reason": c["validity_reason"],
                       "validity_version": c["validity_version"], "dist_outside_demo_box_m": c["dist_outside_demo_box_m"],
                       "checkpoint_hash": ck_hash[arm], "evaluator_hash": ev_hash, "num_steps": args.num_steps,
                       "wall_s": wall, "infrastructure_error": err}
                if rec:
                    row.update(rec)
                row["arm"] = arm + tag
                oc.append_jsonl(out / (arm + tag) / "square_mh_state.jsonl", row)
            n += 1
            if n % 25 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)} jobs  {el:.0f}s  eta {el / n * (len(jobs) - n):.0f}s", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
