"""Branch execution and seeded continuation (spec §6, §7, §11, §13).

execute_candidate(env, snapshot, candidate_chunk, h_exec) -> PostChunk
    Restore the bank snapshot (history checked against the bank), execute exactly h_exec env actions of
    the frozen candidate (stop early on official success / termination), capture the post-chunk snapshot.
run_tail(env, post, tail_policy, continuation_id, remaining_budget) -> TailOutcome
    Restore the post-chunk snapshot and continue with the arm's own frozen policy (same sample mode),
    seeded per replan by continuation(experiment, state_id, continuation_id, relative replan index,
    policy identity); never by candidate. Runs until success or until the remaining budget
    T - elapsed is spent (MID branches do not get a fresh episode; the MultiStepWrapper reward list,
    restored from the snapshot, enforces the same limit and is asserted to agree).

Transitions are deterministic given (snapshot, actions) (restoration_report.md), so the post-chunk state
is executed once per candidate, cached on disk, and reused for every continuation replicate.

Terminal handling: success inside the chunk = success for all replicates; timeout inside the chunk =
failure for all replicates; nonfinite/wrong-shape actions = policy failure (counted, termination_reason
invalid_action_*); simulator exceptions / MuJoCo instability warnings are retried once from a rebuilt
env, and if they persist the record gets error_status='unresolved' (never success/failure), which
blocks metrics.analyze.

Records: one atomic JSON file per (state, arm, candidate, continuation, tail_mode), keyed by a job key over
(experiment, config hash, checkpoint hashes, state, arm, candidate, continuation, tail mode, snapshot
schema+digest, candidate hash). Resume skips verified records and refuses on identity change.
"""

from __future__ import annotations

import json
import os
import queue
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from mip.intent_selection import harness as H
from mip.intent_selection.candidates import load_pool
from mip.intent_selection.rng import KeyedRNG

# Fields that legitimately differ between identical reruns (timings, process ids).
NONDETERMINISTIC_FIELDS = ("candidate_latency_ms", "tail_latency_ms", "tail_inference_ms_total",
                           "tail_env_ms_total", "chunk_env_ms", "wall_ms", "worker_pid",
                           "post_chunk_from_cache")


class ResumeIdentityError(RuntimeError):
    pass


@dataclass
class PostChunk:
    snapshot: object
    elapsed_before: int
    elapsed_after: int
    steps: int
    success: bool
    success_step: Optional[int]
    terminal: bool  # success or timeout inside the chunk
    reward_sum: float
    invalid_action: bool
    clipped_steps: int
    env_ms: float
    trace: dict = field(default_factory=dict)


@dataclass
class TailOutcome:
    success: bool
    termination_reason: str
    steps: int
    success_step: Optional[int]
    reward_sum: float
    replans: int
    inference_ms: float
    env_ms: float
    clipped_steps: int
    trace: dict = field(default_factory=dict)
    seeds: list = field(default_factory=list)


def continuation_seed(cfg: dict, state_id: int, continuation_id: int, replan_index: int,
                      policy_identity: str) -> int:
    return KeyedRNG(cfg["experiment_id"]).derive_seed(
        "continuation", state_id=int(state_id), continuation_id=int(continuation_id),
        replan_index=int(replan_index), policy_identity=policy_identity)


def _new_trace():
    return {"eef_pos": [], "nut_pos": [], "nut_quat": [], "reward": [], "action": []}


def _trace_add(tr, probe, action, r):
    eef, nut, nq = probe.readout()
    tr["eef_pos"].append(eef)
    tr["nut_pos"].append(nut)
    tr["nut_quat"].append(nq)
    tr["reward"].append(r)
    tr["action"].append(np.asarray(action, dtype=np.float64))


def _trace_arrays(tr):
    return {"eef_pos": np.asarray(tr["eef_pos"], np.float32).reshape(-1, 3),
            "nut_pos": np.asarray(tr["nut_pos"], np.float32).reshape(-1, 3),
            "nut_quat": np.asarray(tr["nut_quat"], np.float32).reshape(-1, 4),
            "reward": np.asarray(tr["reward"], np.float32),
            "action": np.asarray(tr["action"], np.float32).reshape(-1, 7)}


def _valid_chunk(chunk: np.ndarray, h_exec: int) -> bool:
    chunk = np.asarray(chunk)
    return chunk.ndim == 2 and chunk.shape[0] >= h_exec and chunk.shape[1] == 7 and bool(np.all(np.isfinite(chunk)))


def _clipped(a) -> bool:
    return bool(np.any(np.abs(np.asarray(a)[:6]) > 1.0))


def execute_candidate(env, snapshot, candidate_chunk: np.ndarray, h_exec: int,
                      expected_history: Optional[np.ndarray] = None) -> PostChunk:
    from mip.intent_selection.snapshots import capture_full_state, restore_full_state

    hist = restore_full_state(snapshot, env)
    if expected_history is not None and not np.array_equal(np.asarray(hist.obs), np.asarray(expected_history)):
        raise RuntimeError("restored history differs from the bank history")
    if hist.pending_actions:
        raise RuntimeError("snapshot is not at a replan boundary (pending actions)")
    probe = H.StepProbe(env)
    e0 = probe.elapsed
    tr = _new_trace()
    start = probe.readout()
    if not _valid_chunk(candidate_chunk, h_exec):
        return PostChunk(None, e0, e0, 0, False, None, True, 0.0, True, 0, 0.0, _trace_arrays(tr))
    steps, rsum, succ, sstep, term, clipped = 0, 0.0, False, None, False, 0
    t0 = time.perf_counter()
    for j in range(h_exec):
        a = candidate_chunk[j]
        clipped += _clipped(a)
        _, r, done = H.step_one(env, probe, a)
        steps += 1
        rsum += r
        _trace_add(tr, probe, a, r)
        if r > 0:
            succ, sstep, term = True, probe.elapsed, True
            break
        if done:
            term = True
            break
    env_ms = (time.perf_counter() - t0) * 1000
    post = capture_full_state(env, pending_actions=[])
    trace = _trace_arrays(tr)
    trace["eef_pos_start"] = start[0].astype(np.float32)
    trace["nut_pos_start"] = start[1].astype(np.float32)
    trace["nut_quat_start"] = start[2].astype(np.float32)
    return PostChunk(post, e0, probe.elapsed, steps, succ, sstep, term, rsum, False, clipped, env_ms, trace)


def run_tail(env, post_snapshot, tail_policy: H.ArmPolicy, cfg: dict, state_id: int,
             continuation_id: int, remaining_budget: int) -> TailOutcome:
    from mip.intent_selection.snapshots import restore_full_state

    hist = restore_full_state(post_snapshot, env)
    probe = H.StepProbe(env)
    obs = np.asarray(hist.obs)
    h = int(cfg["execution_horizon"])
    T = int(cfg["episode_horizon"])
    if probe.elapsed + remaining_budget != T:
        raise RuntimeError(f"budget mismatch: elapsed {probe.elapsed} + remaining {remaining_budget} != T {T}")
    tr = _new_trace()
    seeds: list[int] = []
    steps, k, rsum, inf_ms, env_ms, clipped = 0, 0, 0.0, 0.0, 0.0, 0
    while steps < remaining_budget:
        seed = continuation_seed(cfg, state_id, continuation_id, k, tail_policy.identity)
        seeds.append(int(seed))
        t0 = time.perf_counter()
        cand = tail_policy.adapter.sample_candidate(tail_policy.normalize_history(obs), noise_seed=seed)
        inf_ms += (time.perf_counter() - t0) * 1000
        chunk = np.asarray(cand.env_action_chunk)
        k += 1
        if not _valid_chunk(chunk, h):
            return TailOutcome(False, "invalid_action_tail", steps, None, rsum, k, inf_ms, env_ms, clipped,
                               _trace_arrays(tr), seeds)
        t0 = time.perf_counter()
        for j in range(h):
            if steps >= remaining_budget:
                break
            clipped += _clipped(chunk[j])
            obs, r, done = H.step_one(env, probe, chunk[j])
            steps += 1
            rsum += r
            _trace_add(tr, probe, chunk[j], r)
            if r > 0:
                env_ms += (time.perf_counter() - t0) * 1000
                return TailOutcome(True, "success_in_tail", steps, probe.elapsed, rsum, k, inf_ms, env_ms,
                                   clipped, _trace_arrays(tr), seeds)
            if done:
                if probe.elapsed != T:
                    raise RuntimeError(f"wrapper terminated at {probe.elapsed} != T {T}")
                env_ms += (time.perf_counter() - t0) * 1000
                return TailOutcome(False, "timeout", steps, None, rsum, k, inf_ms, env_ms, clipped,
                                   _trace_arrays(tr), seeds)
        env_ms += (time.perf_counter() - t0) * 1000
    raise RuntimeError("tail budget exhausted without wrapper termination (inconsistent horizon)")


# ----------------------------------------------------------------------------- identity / paths
def record_identity(cfg: dict, state: dict, arm: str, candidate_id: int, continuation_id: int,
                    tail_mode: str, cand_hash: str) -> dict:
    a = cfg["arms"][arm]
    tail_arm = arm  # native tails
    return {"record_version": H.RECORD_VERSION, "experiment_id": cfg["experiment_id"],
            "config_hash": H.config_hash(cfg),
            "checkpoint_hash": cfg["checkpoint_sha256"][a["checkpoint"]],
            "tail_checkpoint_hash": cfg["checkpoint_sha256"][cfg["arms"][tail_arm]["checkpoint"]],
            "tail_policy_identity": H.policy_identity(cfg, tail_arm),
            "state_id": int(state["state_id"]), "arm": arm, "candidate_id": int(candidate_id),
            "continuation_id": int(continuation_id), "tail_mode": tail_mode,
            "snapshot_version": f"{state['snapshot_schema_version']}:{state['snapshot_digest']}",
            "candidate_hash": cand_hash}


def job_key(identity: dict) -> str:
    return H.sha256_json(identity)[:32]


def record_path(cfg: dict, state_id: int, arm: str, cand: int, rep: int, tail_mode: str) -> Path:
    return H.output_dir(cfg) / "records" / str(state_id) / arm / f"c{cand:02d}_r{rep}_{tail_mode}.json"


def post_chunk_paths(cfg: dict, state_id: int, arm: str, cand: int) -> tuple[Path, Path]:
    base = H.output_dir(cfg) / "post_chunk" / str(state_id) / f"{arm}_c{cand:02d}"
    return base.with_suffix(".snap"), base.with_suffix(".json")


def trace_rel(state_id: int, arm: str, cand: int, rep: Optional[int] = None, tail_mode: str = "") -> str:
    leaf = f"c{cand:02d}_chunk.npz" if rep is None else f"c{cand:02d}_r{rep}_{tail_mode}.npz"
    return f"traces/{state_id}/{arm}/{leaf}"


def read_verified_record(path: Path, identity: dict) -> Optional[dict]:
    """Return the record if present and verified; raise on identity change; None if absent."""
    if not path.exists():
        return None
    obj = json.loads(path.read_text())
    if obj.get("identity") != identity:
        diff = {k: (obj.get("identity", {}).get(k), v) for k, v in identity.items()
                if obj.get("identity", {}).get(k) != v}
        raise ResumeIdentityError(f"refusing to resume: identity changed for {path}: {diff}")
    if obj.get("job_key") != job_key(identity) or obj.get("record_sha") != H.sha256_json(obj["record"]):
        raise ResumeIdentityError(f"record {path} failed integrity check")
    return obj["record"]


# ----------------------------------------------------------------------------- worker
_W: dict = {}


def _worker_init(cfg: dict):
    H.silence_logs()
    H.setup_inference_process(cfg)
    _W["cfg"] = cfg
    _W["policies"] = {a: H.build_arm_policy(cfg, a) for a in H.ARM_NAMES}
    _W["env"] = _container_env(cfg)
    _W["snapshots"] = {}
    _W["pools"] = {}


def _container_env(cfg):
    # A container to restore snapshots into; its own reset state is never used for outcomes.
    env = H.build_env(cfg, construction_seed=0)
    env.reset()
    return env


def _get_snapshot(state):
    from mip.intent_selection.snapshots import Snapshot

    sid = state["state_id"]
    if sid not in _W["snapshots"]:
        snap = Snapshot.deserialize(H.output_dir(_W["cfg"]) / state["snapshot_file"])
        if snap.digest() != state["snapshot_digest"]:
            raise RuntimeError(f"bank snapshot digest mismatch for state {sid}")
        _W["snapshots"][sid] = snap
    return _W["snapshots"][sid]


def _get_pool(state, arm):
    key = (state["state_id"], arm)
    if key not in _W["pools"]:
        _W["pools"][key] = load_pool(_W["cfg"], state, arm)
    return _W["pools"][key]


def _with_retry(fn, label, errs):
    """Run fn(); on a simulator/runtime exception rebuild the env and retry once (deterministic)."""
    for attempt in range(2):
        try:
            return fn(), attempt
        except (ResumeIdentityError, KeyboardInterrupt):
            raise
        except Exception as e:  # noqa: BLE001
            errs.append(f"{label} attempt {attempt}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
            try:
                _W["env"].close()
            except Exception:
                pass
            _W["env"] = _container_env(_W["cfg"])
    return None, 2


def _branch_job(job: dict) -> dict:
    """One (state, arm, candidate): chunk once (or from cache), then the pending continuations."""
    from mip.intent_selection.snapshots import Snapshot

    cfg = _W["cfg"]
    out_dir = H.output_dir(cfg)
    state, arm, cid = job["state"], job["arm"], int(job["candidate_id"])
    sid = int(state["state_id"])
    h_exec = int(cfg["execution_horizon"])
    T = int(cfg["episode_horizon"])
    pool = _get_pool(state, arm)
    chunk = np.asarray(pool["env_action_chunk"][cid])
    chash = str(pool["candidate_hash"][cid])
    t_job = time.perf_counter()
    env_steps = 0
    errs: list[str] = []

    # ---------------- chunk (cached post-chunk state)
    snap_p, info_p = post_chunk_paths(cfg, sid, arm, cid)
    chunk_ident = {"config_hash": H.config_hash(cfg), "state_id": sid, "arm": arm, "candidate_id": cid,
                   "candidate_hash": chash, "snapshot_digest": state["snapshot_digest"]}
    info = None
    from_cache = False
    if info_p.exists() and snap_p.exists():
        info = json.loads(info_p.read_text())
        if info["identity"] != chunk_ident:
            raise ResumeIdentityError(f"post-chunk cache identity changed for {info_p}")
        if info.get("post_digest") is not None:
            post_snap = Snapshot.deserialize(snap_p)
            if post_snap.digest() != info["post_digest"]:
                raise ResumeIdentityError(f"post-chunk snapshot digest mismatch {snap_p}")
        else:
            post_snap = None
        from_cache = True
    if info is None:
        snap = _get_snapshot(state)
        hist = np.asarray(state["history"], dtype=np.float64)
        pc, attempts = _with_retry(lambda: execute_candidate(_W["env"], snap, chunk, h_exec, hist),
                                   "chunk", errs)
        if pc is None:
            info = {"identity": chunk_ident, "error": True, "sim_retries": attempts, "errors": errs}
            post_snap = None
        else:
            env_steps += pc.steps * (1 + attempts)
            H.atomic_save_npz(out_dir / trace_rel(sid, arm, cid), **pc.trace)
            post_snap = pc.snapshot
            info = {"identity": chunk_ident, "error": False, "sim_retries": attempts, "errors": errs,
                    "elapsed_before": pc.elapsed_before, "elapsed_after": pc.elapsed_after,
                    "steps": pc.steps, "success": pc.success, "success_step": pc.success_step,
                    "terminal": pc.terminal, "reward_sum": pc.reward_sum,
                    "invalid_action": pc.invalid_action, "clipped_steps": pc.clipped_steps,
                    "env_ms": pc.env_ms,
                    "post_digest": None if post_snap is None else post_snap.digest(),
                    "eef_disp": (float(np.linalg.norm(pc.trace["eef_pos"][-1] - pc.trace["eef_pos_start"]))
                                 if pc.steps > 0 else 0.0),
                    "nut_disp": (float(np.linalg.norm(pc.trace["nut_pos"][-1] - pc.trace["nut_pos_start"]))
                                 if pc.steps > 0 else 0.0)}
            if post_snap is not None:
                post_snap.serialize(snap_p)
            H.atomic_write_json(info_p, info)

    # ---------------- continuations
    written = []
    policy = _W["policies"][arm]  # native tail
    for rep in job["reps"]:
        ident = record_identity(cfg, state, arm, cid, rep, job["tail_mode"], chash)
        rpath = record_path(cfg, sid, arm, cid, rep, job["tail_mode"])
        if read_verified_record(rpath, ident) is not None:
            continue
        t_rec = time.perf_counter()
        base = {
            "experiment_id": cfg["experiment_id"], "config_hash": H.config_hash(cfg),
            "checkpoint_pair_id": cfg["checkpoint_pair_id"], "arm": arm,
            "checkpoint_hash": ident["checkpoint_hash"], "split": state["split"],
            "stratum": state["stratum"], "parent_reset_id": int(state["parent_reset_id"]),
            "state_id": sid, "source_policy": state["source_policy"],
            "elapsed_steps": int(state["elapsed_steps"]), "candidate_id": cid,
            "candidate_seed": int(pool["noise_seed"][cid]), "decoder_seed": int(pool["decoder_seed"][cid]),
            "continuation_id": int(rep), "tail_mode": job["tail_mode"],
            "candidate_latency_ms": float(pool["latency_ms"][cid]), "candidate_hash": chash,
            "job_key": job_key(ident), "tail_policy_identity": ident["tail_policy_identity"],
            "tail_checkpoint_hash": ident["tail_checkpoint_hash"],
            "snapshot_digest": state["snapshot_digest"], "post_chunk_from_cache": from_cache,
            "trace_key_chunk": trace_rel(sid, arm, cid), "record_version": H.RECORD_VERSION,
            "worker_pid": os.getpid(),
        }
        if info.get("error"):
            rec = {**base, "success": False, "termination_reason": "sim_error_chunk", "steps_executed": 0,
                   "success_step": None, "return": 0.0, "tail_latency_ms": 0.0, "error_status": "unresolved",
                   "error_message": "\n".join(info["errors"])[-2000:], "sim_retries": info["sim_retries"],
                   "chunk_steps": 0, "tail_steps": 0, "tail_replans": 0, "chunk_success": False,
                   "chunk_terminal": False, "invalid_action": False, "clipped_steps": 0,
                   "trace_key_tail": None}
        else:
            chunk_part = {"chunk_steps": info["steps"], "chunk_success": info["success"],
                          "chunk_terminal": info["terminal"], "chunk_env_ms": info["env_ms"],
                          "chunk_eef_disp": info["eef_disp"], "chunk_nut_disp": info["nut_disp"]}
            if info["invalid_action"]:
                rec = {**base, **chunk_part, "success": False, "termination_reason": "invalid_action_chunk",
                       "steps_executed": 0, "success_step": None, "return": 0.0, "tail_latency_ms": 0.0,
                       "error_status": "ok", "error_message": "", "sim_retries": info["sim_retries"],
                       "tail_steps": 0, "tail_replans": 0, "invalid_action": True,
                       "clipped_steps": 0, "trace_key_tail": None}
            elif info["terminal"]:
                rec = {**base, **chunk_part, "success": bool(info["success"]),
                       "termination_reason": "success_in_chunk" if info["success"] else "timeout_in_chunk",
                       "steps_executed": info["steps"], "success_step": info["success_step"],
                       "return": info["reward_sum"], "tail_latency_ms": 0.0, "error_status": "ok",
                       "error_message": "", "sim_retries": info["sim_retries"], "tail_steps": 0,
                       "tail_replans": 0, "invalid_action": False, "clipped_steps": info["clipped_steps"],
                       "trace_key_tail": None}
            else:
                if post_snap is None:
                    post_snap = Snapshot.deserialize(snap_p)
                remaining = T - int(info["elapsed_after"])
                tail_errs: list[str] = []
                out, attempts = _with_retry(
                    lambda: run_tail(_W["env"], post_snap, policy, cfg, sid, rep, remaining), "tail", tail_errs)
                if out is None:
                    rec = {**base, **chunk_part, "success": False, "termination_reason": "sim_error_tail",
                           "steps_executed": info["steps"], "success_step": None, "return": info["reward_sum"],
                           "tail_latency_ms": 0.0, "error_status": "unresolved",
                           "error_message": "\n".join(tail_errs)[-2000:], "sim_retries": attempts,
                           "tail_steps": 0, "tail_replans": 0, "invalid_action": False,
                           "clipped_steps": info["clipped_steps"], "trace_key_tail": None}
                else:
                    env_steps += out.steps * (1 + attempts)
                    tkey = trace_rel(sid, arm, cid, rep, job["tail_mode"])
                    H.atomic_save_npz(out_dir / tkey, **out.trace)
                    rec = {**base, **chunk_part, "success": bool(out.success),
                           "termination_reason": out.termination_reason,
                           "steps_executed": info["steps"] + out.steps, "success_step": out.success_step,
                           "return": info["reward_sum"] + out.reward_sum,
                           "tail_latency_ms": out.inference_ms / max(out.replans, 1),
                           "tail_inference_ms_total": out.inference_ms, "tail_env_ms_total": out.env_ms,
                           "error_status": "ok", "error_message": "\n".join(tail_errs)[-2000:],
                           "sim_retries": max(info["sim_retries"], attempts), "tail_steps": out.steps,
                           "tail_replans": out.replans,
                           "invalid_action": out.termination_reason.startswith("invalid"),
                           "clipped_steps": info["clipped_steps"] + out.clipped_steps, "trace_key_tail": tkey,
                           "tail_seeds": out.seeds}
        rec.setdefault("tail_seeds", [])
        rec.setdefault("tail_inference_ms_total", 0.0)
        rec.setdefault("tail_env_ms_total", 0.0)
        rec.setdefault("chunk_env_ms", 0.0)
        rec.setdefault("chunk_eef_disp", float("nan"))
        rec.setdefault("chunk_nut_disp", float("nan"))
        rec["wall_ms"] = (time.perf_counter() - t_rec) * 1000
        H.atomic_write_json(rpath, {"identity": ident, "job_key": job_key(ident), "record": rec,
                                    "record_sha": H.sha256_json(rec)})
        written.append(rep)
    return {"state_id": sid, "arm": arm, "candidate_id": cid, "written": written, "env_steps": env_steps,
            "chunk_from_cache": from_cache, "wall_s": time.perf_counter() - t_job}


# ----------------------------------------------------------------------------- driver
def plan_jobs(cfg: dict, bank) -> tuple[list[dict], int]:
    """All (state, arm, candidate) jobs with their pending continuations; verifies existing records."""
    R = int(cfg["continuation_replicates"])
    M = int(cfg["candidate_pool_size"])
    jobs, n_done = [], 0
    for state in bank.states:
        for arm in H.ARM_NAMES:
            pool = load_pool(cfg, state, arm)  # must exist: candidates are frozen before outcomes
            for cid in range(M):
                for tail_mode in cfg["tail_modes"]:
                    reps = []
                    for rep in range(R):
                        ident = record_identity(cfg, state, arm, cid, rep, tail_mode, str(pool["candidate_hash"][cid]))
                        if read_verified_record(record_path(cfg, state["state_id"], arm, cid, rep, tail_mode),
                                                ident) is None:
                            reps.append(rep)
                        else:
                            n_done += 1
                    if reps:
                        jobs.append({"state": state, "arm": arm, "candidate_id": cid, "tail_mode": tail_mode,
                                     "reps": reps})
    return jobs, n_done


def evaluate(cfg: dict, bank, num_workers: Optional[int] = None, order_seed: Optional[int] = None,
             max_jobs: Optional[int] = None, log=print, env_steps_already: int = 0) -> dict:
    """Run all pending branch jobs in a spawn pool with bounded in-flight submission.

    Budget: stops submitting when counted env steps (bank + branch, including this run) reach
    runtime.max_env_steps or wall time exceeds runtime.max_wall_hours.
    """
    import multiprocessing as mp

    jobs, n_done = plan_jobs(cfg, bank)
    if order_seed is not None:
        perm = np.random.RandomState(int(order_seed)).permutation(len(jobs))
        jobs = [jobs[i] for i in perm]
    if max_jobs is not None:
        jobs = jobs[: int(max_jobs)]
    nw = max(1, min(int(num_workers or cfg["runtime"]["num_workers"]), len(jobs) or 1))
    max_steps = int(cfg["runtime"]["max_env_steps"])
    max_wall = float(cfg["runtime"]["max_wall_hours"]) * 3600
    log(f"[branch] {len(jobs)} jobs pending, {n_done} records already verified, workers={nw}, "
        f"order_seed={order_seed}")
    t0 = time.time()
    steps = int(env_steps_already)
    per_arm = {a: {"env_steps": 0, "jobs": 0, "wall_s": 0.0} for a in H.ARM_NAMES}
    results, stopped = [], None
    if jobs:
        q: queue.Queue = queue.Queue()
        with mp.get_context("spawn").Pool(nw, initializer=_worker_init, initargs=(cfg,)) as pool:
            it = iter(jobs)
            inflight = 0
            exhausted = False
            while True:
                while not exhausted and inflight < 2 * nw and stopped is None:
                    if steps >= max_steps:
                        stopped = f"env step budget reached ({steps} >= {max_steps})"
                        break
                    if time.time() - t0 > max_wall:
                        stopped = f"wall budget reached ({max_wall/3600:.2f} h)"
                        break
                    try:
                        j = next(it)
                    except StopIteration:
                        exhausted = True
                        break
                    pool.apply_async(_branch_job, (j,), callback=lambda r: q.put(("ok", r)),
                                     error_callback=lambda e: q.put(("err", e)))
                    inflight += 1
                if inflight == 0:
                    break
                kind, r = q.get()
                inflight -= 1
                if kind == "err":
                    pool.terminate()
                    raise r
                results.append(r)
                steps += r["env_steps"]
                per_arm[r["arm"]]["env_steps"] += r["env_steps"]
                per_arm[r["arm"]]["jobs"] += 1
                per_arm[r["arm"]]["wall_s"] += r["wall_s"]
                if len(results) % max(1, len(jobs) // 20) == 0 or len(results) == len(jobs):
                    log(f"[branch] {len(results)}/{len(jobs)} jobs, env_steps(total)={steps}, "
                        f"elapsed={time.time()-t0:.0f}s")
    if stopped:
        log(f"[branch] STOPPED: {stopped}")
    return {"jobs_run": len(results), "jobs_planned": len(jobs), "records_preexisting": n_done,
            "records_written": int(sum(len(r["written"]) for r in results)),
            "env_steps_this_run": steps - int(env_steps_already), "env_steps_total": steps,
            "wall_s": time.time() - t0, "workers": nw, "order_seed": order_seed, "per_arm": per_arm,
            "stopped": stopped}


def collect_records(cfg: dict, bank):
    """Load and verify every expected record; returns (DataFrame, n_missing)."""
    import pandas as pd

    rows, missing = [], 0
    R = int(cfg["continuation_replicates"])
    M = int(cfg["candidate_pool_size"])
    for state in bank.states:
        for arm in H.ARM_NAMES:
            pool = load_pool(cfg, state, arm)
            for cid in range(M):
                for tail_mode in cfg["tail_modes"]:
                    for rep in range(R):
                        ident = record_identity(cfg, state, arm, cid, rep, tail_mode, str(pool["candidate_hash"][cid]))
                        rec = read_verified_record(record_path(cfg, state["state_id"], arm, cid, rep, tail_mode), ident)
                        if rec is None:
                            missing += 1
                        else:
                            rows.append(rec)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["state_id", "arm", "candidate_id", "continuation_id", "tail_mode"]).reset_index(drop=True)
    return df, missing
