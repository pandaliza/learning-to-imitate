"""State bank collection (spec §5): RESET and MID strata from preregistered parent resets.

collect_state_bank(config) -> BankManifest

* Parent reset seeds come from a preregistered per-phase range (harness.PHASE_SEED_RANGES), disjoint
  across smoke / pilot / confirmation / reserves and from the Stage-A audit seeds.
* Every parent gets a FRESH env (a repeated reset(seed=s) on a used env is not a true reset), and the
  RESET snapshot is taken immediately after its first seeded reset (native frame padding included).
* MID: the source policy (ACTION_STOCH / INTENT) is assigned by a stable keyed hash with exact
  balance within the primary list (and separately within the reserve list); the target replan index is
  fixed BEFORE the rollout as the native boundary nearest a keyed-uniform time in
  [0.2T, 0.6T]. The source rollout runs with keyed state_collection seeds; if it succeeds or terminates
  before the target, the state is recorded as missing with the reason. Reserve parents (preregistered
  order) top up the MID stratum only, and only until the target count is reached.
* The dev/test split is assigned per parent_reset_id from the config before any branching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mip.intent_selection import harness as H
from mip.intent_selection.rng import KeyedRNG

BANK_VERSION = "1.0"


@dataclass
class BankManifest:
    path: Path
    data: dict = field(default_factory=dict)

    @property
    def states(self) -> list[dict]:
        return self.data["states"]

    @property
    def bank_hash(self) -> str:
        return self.data["bank_hash"]


# ----------------------------------------------------------------------------- preregistration
def source_assignment(cfg: dict, parents: list[int]) -> dict[int, str]:
    """Balanced source policy per parent: order parents by a keyed hash, alternate arms."""
    rng = KeyedRNG(cfg["experiment_id"])
    keyed = sorted(parents, key=lambda p: (rng.derive_seed(
        "state_collection", state_id=int(p), policy_identity="mid_source_assignment"), p))
    arms = list(H.ARM_NAMES)
    return {p: arms[i % 2] for i, p in enumerate(keyed)}


def target_replan_index(cfg: dict, parent: int) -> tuple[int, float]:
    """Nearest native replan boundary to a keyed-uniform time in [f0*T, f1*T] (fixed pre-rollout)."""
    rng = KeyedRNG(cfg["experiment_id"])
    u = rng.derive_seed("state_collection", state_id=int(parent),
                        policy_identity="mid_target_time") / float(2**31)
    f0, f1 = cfg["state_bank"]["mid_time_fraction"]
    T, h = int(cfg["episode_horizon"]), int(cfg["execution_horizon"])
    t = (f0 + (f1 - f0) * u) * T
    idx = int(np.floor(t / h + 0.5))
    lo, hi = int(np.ceil(f0 * T / h)), int(np.floor(f1 * T / h))
    return int(min(max(idx, lo), hi)), float(t)


def source_seed(cfg: dict, parent: int, replan_index: int, policy_id: str) -> int:
    return KeyedRNG(cfg["experiment_id"]).derive_seed(
        "state_collection", state_id=int(parent), replan_index=int(replan_index),
        policy_identity=f"mid_source:{policy_id}")


def preregistration(cfg: dict) -> dict:
    primary, reserve = H.parent_seed_lists(cfg)
    src = {**source_assignment(cfg, primary), **source_assignment(cfg, reserve)}
    plan = []
    for role, plist in (("primary", primary), ("reserve", reserve)):
        for p in plist:
            idx, t = target_replan_index(cfg, p)
            plan.append({"parent_reset_id": p, "role": role, "source_policy": src[p],
                         "target_replan_index": idx, "target_step": idx * int(cfg["execution_horizon"]),
                         "keyed_uniform_time": t})
    return {"primary": primary, "reserve": reserve, "plan": plan}


# ----------------------------------------------------------------------------- worker
_W: dict = {}


def _worker_init(cfg: dict):
    H.silence_logs()
    H.setup_inference_process(cfg)
    _W["cfg"] = cfg
    _W["policies"] = {a: H.build_arm_policy(cfg, a) for a in H.ARM_NAMES}


def _collect_parent(job: dict) -> dict:
    """Fresh env, first seeded reset -> RESET snapshot; optional MID source rollout."""
    from mip.intent_selection.snapshots import capture_full_state

    cfg = _W["cfg"]
    out_dir = Path(job["out_dir"])
    p = int(job["parent_reset_id"])
    t0 = time.time()
    env = H.build_env(cfg, construction_seed=p)
    # robosuite MujocoEnv.rng = default_rng(None) is OS-entropy seeded and never used by robosuite
    # (no reads anywhere in robosuite 1.5.1; stepping/reset do not touch it), but it is part of the
    # snapshot. Seed it from the keyed 'env' namespace so snapshot digests are reproducible.
    from mip.intent_selection.snapshots import find_layers

    find_layers(env)["robosuite"].rng = np.random.default_rng(
        KeyedRNG(cfg["experiment_id"]).derive_seed("env", state_id=p))
    obs, _ = env.reset()  # first reset of a fresh env, seeded by construction (seed p)
    probe = H.StepProbe(env)
    res = {"parent_reset_id": p, "role": job["role"], "states": [], "missing": [], "env_steps": 0}

    def save_state(stratum, hist, extra):
        sid = H.state_id_for(p, stratum)
        snap = capture_full_state(env, history=hist, pending_actions=[])
        f = out_dir / "bank" / "snapshots" / f"{sid}.snap"
        snap.serialize(f)
        eef, nut, nutq = probe.readout()
        res["states"].append({
            "state_id": sid, "parent_reset_id": p, "parent_role": job["role"], "stratum": stratum,
            "split": cfg["state_bank"]["split"], "elapsed_steps": probe.elapsed,
            "snapshot_file": str(f.relative_to(out_dir)), "snapshot_digest": snap.digest(),
            "snapshot_schema_version": snap.schema_version,
            "history": np.asarray(hist, dtype=np.float64).tolist(),
            "history_hash": H.hash_array(np.asarray(hist, dtype=np.float64)),
            "n_real_frames": len(probe.ms.obs), "eef_pos": eef.tolist(), "nut_pos": nut.tolist(),
            **extra})

    if "reset" in job["strata"]:
        save_state("reset", obs, {"source_policy": "none", "source_seed_namespace": None,
                                  "target_replan_index": 0})
    if "mid" in job["strata"]:
        arm = job["source_policy"]
        pol = _W["policies"][arm]
        tgt = int(job["target_replan_index"])
        h = int(cfg["execution_horizon"])
        reason = None
        k = 0
        try:
            while k < tgt:
                seed = source_seed(cfg, p, k, pol.identity)
                cand = pol.adapter.sample_candidate(pol.normalize_history(obs), noise_seed=seed)
                chunk = np.asarray(cand.env_action_chunk)
                if not np.all(np.isfinite(chunk)):
                    reason = f"source_policy_nonfinite_action_at_replan_{k}"
                    break
                for j in range(h):
                    obs, r, done = H.step_one(env, probe, chunk[j])
                    res["env_steps"] += 1
                    if r > 0:
                        reason = f"already_succeeded_at_step_{probe.elapsed}"
                        break
                    if done:
                        reason = f"terminated_at_step_{probe.elapsed}"
                        break
                if reason:
                    break
                k += 1
        except H.SimError as e:
            reason = f"sim_error_in_source_rollout: {e}"
        common = {"source_policy": arm, "source_policy_identity": pol.identity,
                  "source_seed_namespace": "state_collection(state_id=parent, replan_index=k, "
                                           "policy_identity=mid_source:<policy identity>)",
                  "target_replan_index": tgt, "target_step": tgt * h}
        if reason is None:
            assert probe.elapsed == tgt * h
            save_state("mid", obs, common)
        else:
            res["missing"].append({"parent_reset_id": p, "parent_role": job["role"],
                                   "stratum": "mid", "reason": reason,
                                   "stopped_at_step": probe.elapsed, **common})
    env.close()
    res["wall_s"] = time.time() - t0
    return res


# ----------------------------------------------------------------------------- driver
def collect_state_bank(cfg: dict, num_workers: int | None = None) -> BankManifest:
    import multiprocessing as mp

    out_dir = H.output_dir(cfg)
    path = out_dir / "bank_manifest.json"
    chash = H.config_hash(cfg)
    if path.exists():
        data = __import__("json").loads(path.read_text())
        if data["config_hash"] != chash:
            raise RuntimeError(f"existing bank at {path} has config_hash {data['config_hash']} != {chash}; "
                               "refusing to overwrite (identity change)")
        _verify_bank_hash(data)
        return BankManifest(path, data)

    pre = preregistration(cfg)
    plan = {e["parent_reset_id"]: e for e in pre["plan"]}
    strata = list(cfg["state_bank"]["strata"])
    reserves_contribute = list(cfg["state_bank"].get("reserves_contribute", ["mid"]))
    n_target_mid = int(cfg["state_bank"]["parent_resets"]) * int(cfg["state_bank"]["mid_states_per_parent"])
    nw = int(num_workers or cfg["runtime"]["num_workers"])
    t0 = time.time()

    def job(p, role, st):
        e = plan[p]
        return {"parent_reset_id": p, "role": role, "strata": st, "out_dir": str(out_dir),
                "source_policy": e["source_policy"], "target_replan_index": e["target_replan_index"]}

    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(nw, max(1, len(pre["primary"]))), initializer=_worker_init, initargs=(cfg,)) as pool:
        results += pool.map(_collect_parent, [job(p, "primary", strata) for p in pre["primary"]])
        n_mid = sum(1 for r in results for s in r["states"] if s["stratum"] == "mid")
        used_reserves = []
        ri = 0
        while "mid" in strata and n_mid < n_target_mid and ri < len(pre["reserve"]):
            # process reserves strictly in preregistered order; accept in that order
            batch = pre["reserve"][ri: ri + (n_target_mid - n_mid)]
            ri += len(batch)
            rr = pool.map(_collect_parent, [job(p, "reserve", [s for s in strata if s in reserves_contribute])
                                            for p in batch])
            for r in rr:
                if n_mid >= n_target_mid:
                    break  # never happens with batch sizing; kept for safety
                results.append(r)
                used_reserves.append(r["parent_reset_id"])
                n_mid += sum(1 for s in r["states"] if s["stratum"] == "mid")

    states = sorted([s for r in results for s in r["states"]], key=lambda s: s["state_id"])
    missing = [m for r in results for m in r["missing"]]
    src_counts = {a: sum(1 for s in states if s["stratum"] == "mid" and s["source_policy"] == a)
                  for a in H.ARM_NAMES}
    data = {
        "bank_version": BANK_VERSION,
        "experiment_id": cfg["experiment_id"], "phase": cfg["phase"], "config_hash": chash,
        "preregistration": pre,
        "split_assignment": {"unit": "parent_reset_id", "split": cfg["state_bank"]["split"],
                             "assigned_before_branching": True},
        "states": states,
        "missing": missing,
        "reserves_used": [r["parent_reset_id"] for r in results if r["role"] == "reserve"],
        "counts": {"reset": sum(1 for s in states if s["stratum"] == "reset"),
                   "mid": sum(1 for s in states if s["stratum"] == "mid"),
                   "mid_target": n_target_mid, "mid_missing": len(missing),
                   "mid_source_achieved": src_counts,
                   "mid_source_assigned_primary": {a: sum(1 for p in pre["primary"]
                                                          if plan[p]["source_policy"] == a)
                                                   for a in H.ARM_NAMES}},
        "env_steps": int(sum(r["env_steps"] for r in results)),
        "wall_s": time.time() - t0,
        "per_parent_wall_s": {r["parent_reset_id"]: r["wall_s"] for r in results},
    }
    data["bank_hash"] = H.sha256_json({k: v for k, v in data.items()
                                       if k not in ("bank_hash", "wall_s", "per_parent_wall_s")})
    H.atomic_write_json(path, data)
    return BankManifest(path, data)


def _verify_bank_hash(data: dict) -> None:
    h = H.sha256_json({k: v for k, v in data.items() if k not in ("bank_hash", "wall_s", "per_parent_wall_s")})
    if h != data["bank_hash"]:
        raise RuntimeError("bank manifest hash mismatch (modified or corrupt)")


def load_bank(cfg: dict) -> BankManifest:
    import json

    path = H.output_dir(cfg) / "bank_manifest.json"
    data = json.loads(path.read_text())
    if data["config_hash"] != H.config_hash(cfg):
        raise RuntimeError("bank config_hash differs from current config")
    _verify_bank_hash(data)
    return BankManifest(path, data)
