"""Frozen candidate pools (spec §6): all M candidates per (state, arm) generated and stored BEFORE any
outcome evaluation, from the stored observation history only (no env stepping).

Canonical generation: one B=1 forward pass per candidate on CPU (1 thread, matmul precision
"highest"), with keyed CPU-generator noise. Batched (B>1) flow sampling differs from B=1 by ~1e-5 even
at "highest" (rollout_audit A3(d)), so candidates are never batched: identity is independent of how
many candidates are requested per call, of worker count and of order.

File per (state, arm): <phase>/candidates/<state_id>_<arm>.npz with
  identity (json), candidate_id, noise_seed, decoder_seed, initial_noise, intent_decoded,
  normalized_action_full, normalized_action_chunk, physical_action_chunk (10-D), env_action_chunk (7-D,
  exactly what env.step receives), latency_ms, noise/intent/action/candidate hashes, obs_history_norm.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from mip.intent_selection import harness as H
from mip.intent_selection.rng import KeyedRNG

CANDIDATE_VERSION = "1.0"


def candidate_seed(cfg: dict, arm: str, state_id: int, candidate_id: int) -> int:
    return KeyedRNG(cfg["experiment_id"]).derive_seed(
        "candidate", arm=arm, checkpoint_pair=cfg["checkpoint_pair_id"], state_id=int(state_id),
        candidate_id=int(candidate_id))


def candidate_hash(initial_noise: np.ndarray, env_chunk: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    for a in (initial_noise, env_chunk):
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def generate_pool(policy: H.ArmPolicy, history: np.ndarray, seeds: list[int],
                  request_batch_size: int = 1) -> dict:
    """Generate candidates for one state. ``request_batch_size`` only groups requests; every
    candidate is still an independent B=1 forward pass (canonical, batch-size independent)."""
    on = policy.normalize_history(history)
    rows = []
    for g in range(0, len(seeds), max(1, int(request_batch_size))):
        for s in seeds[g: g + request_batch_size]:
            t0 = time.perf_counter()
            c = policy.adapter.sample_candidate(on, noise_seed=int(s))
            lat = (time.perf_counter() - t0) * 1000.0
            rows.append((c, lat))
    is_intent = policy.adapter.is_intent_arm
    noise = np.stack([c.initial_noise.cpu().numpy() for c, _ in rows])
    env = np.stack([np.asarray(c.env_action_chunk) for c, _ in rows])
    out = {
        "noise_seed": np.asarray(seeds, dtype=np.int64),
        "initial_noise": noise,
        "intent_decoded": (np.stack([c.intent_decoded.cpu().numpy() for c, _ in rows])
                           if is_intent else np.zeros((len(rows), 0), np.float32)),
        "normalized_action_full": np.stack([c.normalized_action_full.cpu().numpy() for c, _ in rows]),
        "normalized_action_chunk": np.stack([c.normalized_action_chunk.cpu().numpy() for c, _ in rows]),
        "physical_action_chunk": np.stack([c.physical_action_chunk.numpy() for c, _ in rows]),
        "env_action_chunk": env,
        "latency_ms": np.asarray([lat for _, lat in rows], dtype=np.float64),
        "noise_hash": np.asarray([c.initial_noise_hash for c, _ in rows]),
        "intent_hash": np.asarray([c.intent_hash or "" for c, _ in rows]),
        "action_hash": np.asarray([c.action_hash for c, _ in rows]),
        "candidate_hash": np.asarray([candidate_hash(noise[i], env[i]) for i in range(len(rows))]),
        "obs_history_norm": on.numpy(),
    }
    return out


def pool_identity(cfg: dict, state: dict, arm: str) -> dict:
    a = cfg["arms"][arm]
    return {"candidate_version": CANDIDATE_VERSION, "experiment_id": cfg["experiment_id"],
            "config_hash": H.config_hash(cfg), "state_id": int(state["state_id"]), "arm": arm,
            "checkpoint_sha256": cfg["checkpoint_sha256"][a["checkpoint"]],
            "policy_identity": H.policy_identity(cfg, arm),
            "snapshot_digest": state["snapshot_digest"], "history_hash": state["history_hash"],
            "M": int(cfg["candidate_pool_size"])}


def pool_path(cfg: dict, state_id: int, arm: str) -> Path:
    return H.output_dir(cfg) / "candidates" / f"{int(state_id)}_{arm}.npz"


def load_pool(cfg: dict, state: dict, arm: str) -> dict:
    z = np.load(pool_path(cfg, state["state_id"], arm), allow_pickle=False)
    ident = json.loads(str(z["identity"]))
    if ident != pool_identity(cfg, state, arm):
        raise RuntimeError(f"candidate pool identity mismatch for state {state['state_id']} {arm}")
    out = {k: z[k] for k in z.files}
    out["identity"] = ident
    for i in range(len(out["candidate_hash"])):
        if candidate_hash(out["initial_noise"][i], out["env_action_chunk"][i]) != str(out["candidate_hash"][i]):
            raise RuntimeError(f"candidate hash mismatch state {state['state_id']} {arm} cand {i}")
    return out


# ----------------------------------------------------------------------------- worker / driver
_W: dict = {}


def _worker_init(cfg: dict):
    H.silence_logs()
    H.setup_inference_process(cfg)
    _W["cfg"] = cfg
    _W["policies"] = {a: H.build_arm_policy(cfg, a) for a in H.ARM_NAMES}


def _gen_job(job):
    cfg = _W["cfg"]
    state, arm = job["state"], job["arm"]
    M = int(cfg["candidate_pool_size"])
    seeds = [candidate_seed(cfg, arm, state["state_id"], j) for j in range(M)]
    pool = generate_pool(_W["policies"][arm], np.asarray(state["history"], dtype=np.float64), seeds)
    ident = pool_identity(cfg, state, arm)
    pool["candidate_id"] = np.arange(M, dtype=np.int64)
    pool["decoder_seed"] = np.full(M, -1, dtype=np.int64)  # deterministic MLP decoder / no decoder
    H.atomic_save_npz(pool_path(cfg, state["state_id"], arm), identity=np.asarray(json.dumps(ident, sort_keys=True)),
                      **pool)
    return {"state_id": state["state_id"], "arm": arm,
            "pool_hash": H.sha256_json([str(x) for x in pool["candidate_hash"]]),
            "latency_ms_mean": float(pool["latency_ms"].mean()), "n": M}


def generate_candidates(cfg: dict, bank, num_workers: int | None = None) -> dict:
    import multiprocessing as mp

    out_dir = H.output_dir(cfg)
    jobs, done = [], []
    for s in bank.states:
        for arm in H.ARM_NAMES:
            p = pool_path(cfg, s["state_id"], arm)
            if p.exists():
                pool = load_pool(cfg, s, arm)  # raises on identity change
                done.append({"state_id": s["state_id"], "arm": arm,
                             "pool_hash": H.sha256_json([str(x) for x in pool["candidate_hash"]]),
                             "latency_ms_mean": float(pool["latency_ms"].mean()), "n": len(pool["candidate_id"])})
            else:
                jobs.append({"state": s, "arm": arm})
    t0 = time.time()
    if jobs:
        nw = min(int(num_workers or cfg["runtime"]["num_workers"]), len(jobs))
        with mp.get_context("spawn").Pool(nw, initializer=_worker_init, initargs=(cfg,)) as pool:
            done += pool.map(_gen_job, jobs)
    done.sort(key=lambda r: (r["state_id"], r["arm"]))
    man = {"candidate_version": CANDIDATE_VERSION, "config_hash": H.config_hash(cfg),
           "bank_hash": bank.bank_hash, "pools": done,
           "candidate_pool_hash": H.sha256_json([(r["state_id"], r["arm"], r["pool_hash"]) for r in done]),
           "generated_before_outcomes": True, "wall_s_this_invocation": time.time() - t0,
           "generation": "B=1 per candidate, CPU, 1 thread, highest precision, keyed CPU noise"}
    H.atomic_write_json(out_dir / "candidates" / "candidates_manifest.json", man)
    return man
