"""Intent-selection harness driver (spec §10-§13).

    .venv/bin/python examples/intent_selection/run_experiment.py \
        --config mip/intent_selection/configs/pilot.yaml --stage {bank,candidates,branch,analyze,all}

Outputs under outputs/intent_selection/<phase>/ (or --out-dir):
  manifest.json            spec §3 fields + `unresolved` + per-stage run log (cost, budgets)
  bank_manifest.json       state bank (hash, preregistration, missing MID states, source balance)
  bank/snapshots/*.snap    RESET / MID snapshots (schema-versioned, digest-checked)
  candidates/*.npz         frozen candidate pools (+ candidates_manifest.json)
  post_chunk/              cached post-chunk snapshots
  records/                 atomic per-job JSON records;  records.parquet (analyze)
  traces/                  EEF / nut / action traces referenced by trace_key_*
  metrics_by_state.csv, aggregate_metrics.csv, metrics_seed_*.csv (metrics.write_results_to_csv)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU inference (see mip/intent_selection/harness.py)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from mip.intent_selection import harness as H  # noqa: E402


# ----------------------------------------------------------------------------- manifest
def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for m in ("torch", "numpy", "robosuite", "mujoco", "robomimic", "gymnasium", "pandas", "scipy", "hydra"):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                mod = __import__(m)
            out[m] = getattr(mod, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001
            out[m] = f"unavailable: {e}"
    return out


def _hardware() -> dict:
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return {"hostname": platform.node(), "cpu_model": cpu, "logical_cpus": os.cpu_count(),
            "platform": platform.platform(), "inference_device": "cpu (CUDA hidden)"}


def _env_facts(cfg: dict) -> dict:
    from mip.intent_selection.snapshots import SNAPSHOT_SCHEMA_VERSION, find_layers, model_fingerprint

    env = H.build_env(cfg, construction_seed=0)
    env.reset()
    L = find_layers(env)
    rs = L["robosuite"]
    robot = rs.robots[0]
    osc = robot.composite_controller.part_controllers["right"]
    grip = robot.composite_controller.part_controllers["right_gripper"]
    try:
        xml = rs.sim.model.get_xml()
    except Exception:
        xml = rs.model.get_xml()

    def g(o, k):
        v = getattr(o, k, None)
        return np.asarray(v).tolist() if isinstance(v, np.ndarray) else (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))

    facts = {
        "env_class": type(rs).__name__, "robot": type(robot).__name__,
        "control_freq_hz": float(rs.control_freq), "model_timestep": float(rs.model_timestep),
        "robosuite_horizon": int(rs.horizon), "robosuite_ignore_done": bool(rs.ignore_done),
        "reward_shaping": bool(rs.reward_shaping), "hard_reset": bool(rs.hard_reset),
        "episode_timeout_steps": int(L["multistep"].max_episode_steps),
        "episode_timeout_source": "MultiStepWrapper.max_episode_steps (robosuite ignore_done=True, horizon 1000)",
        "env_action_dim": int(rs.action_dim),
        "controller": {"arm": type(osc).__name__, "arm_input_type": g(osc, "input_type"),
                       "arm_input_ref_frame": g(osc, "input_ref_frame"), "kp": g(osc, "kp"),
                       "kd": g(osc, "kd"), "input_max": g(osc, "input_max"), "input_min": g(osc, "input_min"),
                       "output_max": g(osc, "output_max"), "output_min": g(osc, "output_min"),
                       "gripper": type(grip).__name__},
        "controller_note": "make_env sets controller_configs.control_delta=False for abs_action, but robosuite "
                           "1.5 reads body_parts.right.control_delta (true): OSC runs in DELTA mode; training "
                           "actions are delta too (consistent). See restoration_report.md.",
        "model_xml_sha256": H.sha256_json(xml),
        "model_fingerprint": model_fingerprint(rs),
        "success_predicate": "official robosuite NutAssemblySquare._check_success (square nut on square peg); "
                             "reward_shaping=False so per-step reward>0 <=> success (native eval uses "
                             "sum(reward)>0). Branches stop at the first success step.",
        "obs_keys": list(L["lowdim"].obs_keys),
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
    }
    env.close()
    return facts


def build_manifest(cfg: dict) -> dict:
    from mip.intent_selection.rng import KeyedRNG

    hc_a = H.build_hydra_config(cfg, "action")
    hc_i = H.build_hydra_config(cfg, "intent")
    ds_a = H.get_dataset(hc_a)
    ds_i = H.get_dataset(hc_i)
    from huggingface_hub import hf_hub_download

    with contextlib.redirect_stdout(io.StringIO()):
        data_path = hf_hub_download(repo_id=hc_a.task.dataset_repo, filename=hc_a.task.dataset_filename,
                                    repo_type="dataset")
    on_a, on_i = ds_a.normalizer["obs"]["state"], ds_i.normalizer["obs"]["state"]
    an_a, an_i = ds_a.normalizer["action"], ds_i.normalizer["action"]
    norm = {
        "obs_normalizer": "MinMaxNormalizer to [-1,1] over dataset obs (N, 23)",
        "obs_min_range_hash_action_ckpt_config": H.hash_array(np.concatenate([on_a.min, on_a.range])),
        "obs_min_range_hash_intent_ckpt_config": H.hash_array(np.concatenate([on_i.min, on_i.range])),
        "action_min_range_hash_action_ckpt_config": H.hash_array(np.concatenate([an_a.min, an_a.range])),
        "action_min_range_hash_intent_ckpt_config": H.hash_array(np.concatenate([an_i.min, an_i.range])),
    }
    ck = {}
    unresolved = []
    for k in ("action", "intent"):
        sha = H.sha256_file(cfg["checkpoints"][k])
        ck[k] = {"path": cfg["checkpoints"][k], "sha256": sha, "sha256_expected": cfg["checkpoint_sha256"][k],
                 "training_step": cfg["checkpoint_training_step"][k],
                 "training_seed": cfg["checkpoint_training_seed"][k],
                 "training_seed_source": "checkpoint file name (seed0)",
                 "hydra_overrides": cfg["hydra_overrides"][k]}
        if sha != cfg["checkpoint_sha256"][k]:
            unresolved.append({"field": f"checkpoints.{k}.sha256", "affects_inference": True,
                               "note": f"file sha {sha} != expected {cfg['checkpoint_sha256'][k]}"})
    if norm["obs_min_range_hash_action_ckpt_config"] != norm["obs_min_range_hash_intent_ckpt_config"]:
        norm["note"] = "obs normalizers differ between the two checkpoint configs; each arm uses its own"
    for (a, b) in (("horizon", "prediction_horizon"), ("act_steps", "execution_horizon"), ("obs_steps", "obs_steps")):
        for hc in (hc_a, hc_i):
            if int(getattr(hc.task, a)) != int(cfg[b]):
                unresolved.append({"field": b, "affects_inference": True,
                                   "note": f"task.{a}={getattr(hc.task, a)} != {cfg[b]}"})
    env = _env_facts(cfg)
    if env["episode_timeout_steps"] != int(cfg["episode_horizon"]):
        unresolved.append({"field": "episode_horizon", "affects_inference": True, "note": "env timeout mismatch"})
    if env["env_action_dim"] != 7:
        unresolved.append({"field": "env_action_dim", "affects_inference": True, "note": str(env["env_action_dim"])})
    # Known open items that do not affect inference or action execution:
    unresolved += [
        {"field": "native_eval_original_seeds", "affects_inference": False,
         "note": "saved checkpoint SRs (0.62/0.64) are maxima over periodic evals with unknown seeds; "
                 "measured (rollout_audit): ACTION zero ~0.55-0.64, ACTION stochastic ~0.66-0.72, INTENT ~0.43"},
        {"field": "training_seed_robustness", "affects_inference": False,
         "note": "single checkpoint pair (seed 0): conclusions are checkpoint-specific"},
        {"field": "checkpoint_selection", "affects_inference": False,
         "note": "checkpoints are the provided best-eval checkpoints (selected on training-time eval, not on this bank)"},
    ]
    rng = KeyedRNG(cfg["experiment_id"])
    man = {
        "manifest_version": "1.0",
        "experiment_id": cfg["experiment_id"], "phase": cfg["phase"], "config_path": cfg["_config_path"],
        "config_hash": H.config_hash(cfg), "config": H.identity_config(cfg),
        "code": H.git_info(), "versions": _versions(), "hardware": _hardware(),
        "checkpoints": ck, "checkpoint_pair_id": cfg["checkpoint_pair_id"],
        "training_data": {"path": data_path, "sha256": H.sha256_file(data_path),
                          "hf_repo": hc_a.task.dataset_repo, "hf_filename": hc_a.task.dataset_filename},
        "observation": {"keys_in_order": env["obs_keys"], "dims": {"object": 14, "robot0_eef_pos": 3,
                        "robot0_eef_quat": 4, "robot0_gripper_qpos": 2}, "obs_dim": int(cfg["obs_dim"]),
                        "history_length": int(cfg["obs_steps"]),
                        "history_padding": "native MultiStepWrapper: first obs repeated at reset", **norm},
        "action": {"prediction_horizon": int(cfg["prediction_horizon"]),
                   "execution_horizon_H_exec": int(cfg["execution_horizon"]),
                   "execution_slice": "normalized chunk [obs_steps-1 : obs_steps-1+8] (native)",
                   "control_frequency_hz": env["control_freq_hz"], "env_action_dim": env["env_action_dim"],
                   "internal_action_representation": "10-D: pos(3) + rot6d(6) + gripper(1), MinMax-normalized "
                                                     "to [-1,1]; env action = undo_transform_action(unnormalize) "
                                                     "-> pos(3) + axis-angle(3) + gripper(1), consumed as OSC "
                                                     "DELTA commands",
                   "action_normalizer_hash_blake2b_resolved": "39231d18829e49dee86d4e54134ff0ff"},
        "intent": {"definition": "mean over intent_horizon=8 future steps of min-max-normalized obs slice "
                                 "robot0_eef_pos(3)+robot0_eef_quat(4, xyzw)",
                   "intent_dim": 7, "intent_horizon": 8, "reference_frame": "robosuite world frame (eef obs)",
                   "normalization": "obs MinMax normalizer slice [14:21]",
                   "decoder": "deterministic MLP(obs_emb, intent) -> (10, 10) normalized actions; no decoder noise",
                   "decoder_seed": "not applicable (recorded as -1)"},
        "sampler": {"ACTION_STOCH": {"noise_shape": [int(cfg["prediction_horizon"]), 10],
                                     "initial_noise": "N(0,I) (NON-NATIVE; native ACTION mode is 'zero')"},
                    "INTENT": {"noise_shape": [1, 7], "initial_noise": "N(0,I) (native)"},
                    "schedule": "Euler, t = linspace(0,1,NFE+1)", "nfe": int(cfg["nfe"]), "ema_weights": True,
                    "noise_generation": "keyed torch CPU generator per candidate / per tail replan"},
        "arms": cfg["arms"],
        "policy_identities": {a: H.policy_identity(cfg, a) for a in H.ARM_NAMES},
        "inference": {**cfg["inference"],
                      "deviations_from_native": [
                          "float32 matmul precision 'highest' instead of native 'high' (TF32 made candidates "
                          "batch-size dependent)",
                          "CPU inference (native eval used CUDA); adapter path is identical code",
                          "every policy call is B=1 (native eval batches 50 envs)"]},
        "environment": {**env, "task": cfg["task"], "fresh_env_per_parent": True,
                        "fresh_env_reason": "a repeated reset(seed=s) on a used env is not a true reset "
                                            "(seed_state_map reset_to leaves controller/gripper state)",
                        "transitions_deterministic": "bit-exact given snapshot+actions (restoration_report.md); "
                                                     "post-chunk states are cached",
                        "env_stochasticity": "none during branches (robosuite env.rng is never read)"},
        "episode": {"T": int(cfg["episode_horizon"]),
                    "branch_budget": "T - snapshot elapsed steps (MID does not get a fresh episode)"},
        "seed_namespaces": {
            "experiment_id": cfg["experiment_id"],
            "parent_reset_seeds": {"phase_ranges": H.PHASE_SEED_RANGES, "audit_seeds_excluded": H.AUDIT_SEED_RANGES,
                                   "this_phase": cfg["seed_range_phase"]},
            "state_collection": "KeyedRNG('state_collection', state_id=parent, policy_identity=mid_source_assignment"
                                " | mid_target_time | replan_index+policy_identity=mid_source:<pid>)",
            "candidate": "KeyedRNG('candidate', arm, checkpoint_pair, state_id, candidate_id)",
            "decoder": "unused (deterministic decoder)",
            "continuation": "KeyedRNG('continuation', state_id, continuation_id, replan_index (relative to tail "
                            "start), policy_identity) -- never candidate_id",
            "env": "KeyedRNG('env', state_id=parent) seeds robosuite MujocoEnv.rng at bank construction (that "
                   "generator is never read by robosuite; seeded only so snapshot digests are reproducible). "
                   "No environment stochasticity during branches.", "bootstrap": "KeyedRNG('bootstrap', state_id=0) (metrics.analyze default)",
            "example_continuation_seed": rng.derive_seed("continuation", state_id=1, continuation_id=0,
                                                         replan_index=0, policy_identity="x"),
        },
        "continuation_mode": cfg["continuation_mode"], "tail_modes": cfg["tail_modes"],
        "snapshot_schema_version": env["snapshot_schema_version"],
        "unresolved": unresolved,
    }
    return man


def _manifest_path(cfg):
    return H.output_dir(cfg) / "manifest.json"


def ensure_manifest(cfg: dict) -> dict:
    p = _manifest_path(cfg)
    fresh = build_manifest(cfg)
    blocking = [u for u in fresh["unresolved"] if u["affects_inference"]]
    if blocking:
        raise SystemExit(f"unresolved fields affect inference/execution: {blocking}")
    if p.exists():
        old = json.loads(p.read_text())
        if old["config_hash"] != fresh["config_hash"]:
            raise SystemExit(f"manifest config_hash {old['config_hash']} != {fresh['config_hash']}: refusing")
        if old["checkpoints"] != fresh["checkpoints"]:
            raise SystemExit("checkpoint identity changed: refusing")
        fresh["stages"] = old.get("stages", {})
        fresh["code_history"] = old.get("code_history", []) + (
            [old["code"]] if old.get("code") != fresh["code"] else [])
        for k in ("state_bank", "candidate_pool", "analysis_summary", "cost"):
            if k in old:
                fresh[k] = old[k]
    else:
        fresh["stages"] = {}
    H.atomic_write_json(p, fresh)
    return fresh


def log_stage(cfg: dict, stage: str, info: dict) -> None:
    p = _manifest_path(cfg)
    man = json.loads(p.read_text())
    man.setdefault("stages", {}).setdefault(stage, []).append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **info})
    H.atomic_write_json(p, man)


def update_manifest(cfg: dict, **kv) -> None:
    p = _manifest_path(cfg)
    man = json.loads(p.read_text())
    man.update(kv)
    H.atomic_write_json(p, man)


# ----------------------------------------------------------------------------- stages
def stage_bank(cfg, args, log):
    from mip.intent_selection.state_bank import collect_state_bank

    t0 = time.time()
    bank = collect_state_bank(cfg, num_workers=args.num_workers)
    d = bank.data
    log(f"[bank] reset={d['counts']['reset']} mid={d['counts']['mid']}/{d['counts']['mid_target']} "
        f"missing={d['counts']['mid_missing']} sources={d['counts']['mid_source_achieved']} "
        f"reserves_used={d['reserves_used']} hash={d['bank_hash'][:12]}")
    update_manifest(cfg, state_bank={"file": "bank_manifest.json", "bank_hash": d["bank_hash"],
                                     "split": d["split_assignment"], "counts": d["counts"],
                                     "missing": d["missing"], "reserves_used": d["reserves_used"],
                                     "env_steps": d["env_steps"]})
    log_stage(cfg, "bank", {"wall_s": time.time() - t0, "env_steps": d["env_steps"],
                            "bank_hash": d["bank_hash"], "collection_wall_s": d["wall_s"]})
    return bank


def stage_candidates(cfg, args, log):
    from mip.intent_selection.candidates import generate_candidates
    from mip.intent_selection.state_bank import load_bank

    bank = load_bank(cfg)
    t0 = time.time()
    man = generate_candidates(cfg, bank, num_workers=args.num_workers)
    lat = {a: float(np.mean([p["latency_ms_mean"] for p in man["pools"] if p["arm"] == a])) for a in H.ARM_NAMES}
    log(f"[candidates] {len(man['pools'])} pools, pool hash {man['candidate_pool_hash'][:12]}, latency ms {lat}")
    update_manifest(cfg, candidate_pool={"manifest": "candidates/candidates_manifest.json",
                                         "candidate_pool_hash": man["candidate_pool_hash"],
                                         "n_pools": len(man["pools"]), "mean_latency_ms": lat,
                                         "generated_before_outcomes": True})
    log_stage(cfg, "candidates", {"wall_s": time.time() - t0, "mean_latency_ms": lat})


def stage_branch(cfg, args, log):
    from mip.intent_selection.branch_eval import evaluate
    from mip.intent_selection.state_bank import load_bank

    bank = load_bank(cfg)
    man = json.loads(_manifest_path(cfg).read_text())
    prior = int(bank.data["env_steps"]) + sum(int(r.get("env_steps_this_run", 0))
                                              for r in man.get("stages", {}).get("branch", []))
    res = evaluate(cfg, bank, num_workers=args.num_workers, order_seed=args.order_seed,
                   max_jobs=args.max_jobs, log=log, env_steps_already=prior)
    log(f"[branch] done: {json.dumps({k: v for k, v in res.items() if k != 'per_arm'})}")
    log_stage(cfg, "branch", res)
    return res


def _cost(cfg, df, bank, man) -> dict:
    """Measured cost from records (chunk steps counted once per candidate) + stage logs."""
    out = {"per_arm": {}}
    chunk_once = df.drop_duplicates(["state_id", "arm", "candidate_id"])
    for a in H.ARM_NAMES:
        d = df[df["arm"] == a]
        c = chunk_once[chunk_once["arm"] == a]
        tail_inf = float(d.get("tail_inference_ms_total", 0).fillna(0).sum()) if "tail_inference_ms_total" in d else 0.0
        out["per_arm"][a] = {
            "continuations": int(len(d)),
            "sim_steps_chunks": int(c["chunk_steps"].fillna(0).sum()),
            "sim_steps_tails": int(d["tail_steps"].fillna(0).sum()),
            "tail_replans": int(d["tail_replans"].fillna(0).sum()),
            "inference_s_candidates": float(c["candidate_latency_ms"].sum() / 1000),
            "inference_s_tails": tail_inf / 1000,
            "env_s_tails": float(d.get("tail_env_ms_total", 0).fillna(0).sum() / 1000) if "tail_env_ms_total" in d else 0.0,
            "env_s_chunks": float(c["chunk_env_ms"].fillna(0).sum() / 1000) if "chunk_env_ms" in c else 0.0,
            "record_wall_s_sum": float(d["wall_ms"].sum() / 1000),
            "mean_remaining_budget": float((cfg["episode_horizon"] - d["elapsed_steps"]).mean()),
            "mean_steps_executed": float(d["steps_executed"].mean()),
        }
        pa = out["per_arm"][a]
        pa["sim_steps_total"] = pa["sim_steps_chunks"] + pa["sim_steps_tails"]
    br = man.get("stages", {}).get("branch", [])
    out["branch_runs"] = [{k: r.get(k) for k in ("wall_s", "workers", "jobs_run", "records_written",
                                                  "env_steps_this_run", "order_seed", "stopped")} for r in br]
    out["bank_env_steps"] = int(bank.data["env_steps"])
    out["bank_wall_s"] = float(bank.data["wall_s"])
    tot_steps = sum(v["sim_steps_total"] for v in out["per_arm"].values())
    out["branch_sim_steps_total"] = int(tot_steps)
    out["core_seconds_per_continuation"] = float(df["wall_ms"].sum() / 1000 / max(len(df), 1))
    out["core_seconds_per_sim_step"] = float(df["wall_ms"].sum() / 1000 / max(tot_steps, 1))
    return out


def own_summary(cfg, df, n_boot=2000) -> dict:
    """Independent numpy implementation of M1 / C(K) per arm x stratum + paired bootstrap (cross-check)."""
    from scipy.special import comb

    from mip.intent_selection.rng import KeyedRNG

    M = int(cfg["candidate_pool_size"])
    Ks = sorted(set([1, 4, 16, M]) & set(range(1, M + 1)))
    arms = list(H.ARM_NAMES)
    out = {"K": Ks, "strata": {}}
    seed = KeyedRNG(cfg["experiment_id"]).derive_seed("bootstrap", state_id=1)
    for stratum in ("reset", "mid"):
        d = df[(df["stratum"] == stratum) & (df["tail_mode"] == "native")]
        if not len(d):
            continue
        parents = sorted(d["parent_reset_id"].unique())
        # per parent, per arm: M1 and C(K) averaged over (state, replicate)
        per = {}
        for a in arms:
            per[a] = {}
            for p in parents:
                dp = d[(d["arm"] == a) & (d["parent_reset_id"] == p)]
                vals = {"m1": [], **{f"c{k}": [] for k in Ks}}
                for sid, ds in dp.groupby("state_id"):
                    vals["m1"].append(ds["success"].mean())
                    cs = {k: [] for k in Ks}
                    for r, dr in ds.groupby("continuation_id"):
                        m = int(dr["success"].sum())
                        assert len(dr) == M
                        for k in Ks:
                            cs[k].append(1 - comb(M - m, k, exact=True) / comb(M, k, exact=True))
                    for k in Ks:
                        vals[f"c{k}"].append(np.mean(cs[k]))
                per[a][p] = {k: float(np.mean(v)) for k, v in vals.items()}
        keys = ["m1"] + [f"c{k}" for k in Ks]
        X = {a: np.array([[per[a][p][k] for k in keys] for p in parents]) for a in arms}  # (P, nk)
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, len(parents), size=(n_boot, len(parents)))
        diff = X["INTENT"] - X["ACTION_STOCH"]
        boot = diff[idx].mean(axis=1)  # (n_boot, nk)
        st = {"n_parents": len(parents), "n_states": int(d["state_id"].nunique())}
        for a in arms:
            st[a] = {k: float(X[a][:, i].mean()) for i, k in enumerate(keys)}
            st[a]["gain_c" + str(Ks[-1]) + "_minus_c1"] = st[a][f"c{Ks[-1]}"] - st[a]["c1"]
        st["INTENT_minus_ACTION_STOCH"] = {
            k: {"mean": float(diff[:, i].mean()), "ci95": [float(np.percentile(boot[:, i], 2.5)),
                                                         float(np.percentile(boot[:, i], 97.5))]}
            for i, k in enumerate(keys)}
        # per-state success counts (diagnostic): states with any success, saturation
        sat = {}
        for a in arms:
            ps = d[d["arm"] == a].groupby("state_id")["success"].mean()
            sat[a] = {"states_all_success": int((ps == 1).sum()), "states_no_success": int((ps == 0).sum()),
                      "n_states": int(len(ps))}
        st["saturation"] = sat
        out["strata"][stratum] = st
    out["bootstrap"] = {"unit": "parent_reset_id", "resamples": n_boot, "seed": seed,
                        "note": "per-parent means; states equally weighted within parent (1 state/parent/stratum)"}
    return out


def m3_preview(cfg, df) -> dict:
    """Exploratory one-selection/one-validation preview (R=2): select on rep 0, validate on rep 1."""
    from mip.intent_selection.metrics import compute_m3_cross_validated_selector

    R = int(cfg["continuation_replicates"])
    sel, held = ([0, 1], [2, 3]) if R >= 4 else ([0], [1])
    M = int(cfg["candidate_pool_size"])
    ks = sorted(set([1, 4, 16, M]) & set(range(1, M + 1)))
    out = {"selection_replicates": sel, "heldout_replicates": held, "exploratory": R < 4, "strata": {}}
    for stratum in ("reset", "mid"):
        d = df[df["stratum"] == stratum]
        res = {}
        for a in H.ARM_NAMES:
            rows = []
            for sid in sorted(d["state_id"].unique()):
                prim, _ = compute_m3_cross_validated_selector(
                    d, a, sid, cfg["experiment_id"], cfg["checkpoint_pair_id"], selection_replicates=sel,
                    heldout_replicates=held, k_values=ks, n_subsets_k4=int(cfg["analysis"]["subset_samples_k4"]))
                rows.append(prim)
            res[a] = {k: {m: float(np.nanmean([r[k][m] for r in rows])) for m in
                          ("selected_success", "random_baseline", "gain", "tie_fraction")} for k in ks}
        out["strata"][stratum] = res
    return out


def stage_analyze(cfg, args, log):
    import pandas as pd

    from mip.intent_selection.branch_eval import collect_records
    from mip.intent_selection.metrics import OutcomeRecordValidator, analyze, write_results_to_csv
    from mip.intent_selection.state_bank import load_bank

    bank = load_bank(cfg)
    out_dir = H.output_dir(cfg)
    df, missing = collect_records(cfg, bank)
    if missing and not args.allow_incomplete:
        raise SystemExit(f"{missing} expected records missing; run --stage branch (or --allow-incomplete)")
    ok, errs, warns = OutcomeRecordValidator(df).validate()
    log(f"[analyze] {len(df)} records, missing={missing}, validator ok={ok} errors={errs} warnings={len(warns)}")
    df.to_parquet(out_dir / "records.parquet", index=False)
    man = json.loads(_manifest_path(cfg).read_text())
    tables = analyze(man, df, n_resamples=int(cfg["analysis"]["bootstrap_resamples"]),
                     continuation_replicates=int(cfg["continuation_replicates"]),
                     report_k_values=[k for k in cfg["report_k"] if k <= int(cfg["candidate_pool_size"])])
    with contextlib.redirect_stdout(io.StringIO()):
        write_results_to_csv(tables, str(out_dir))
    summ = own_summary(cfg, df, n_boot=int(cfg["analysis"]["bootstrap_resamples"]))
    m3 = m3_preview(cfg, df)
    # cross-check metrics.analyze aggregate table against the independent implementation
    agg = tables["aggregate_metrics"]
    xcheck = []
    for stratum, st in summ["strata"].items():
        for a in H.ARM_NAMES:
            row = agg[(agg["stratum"] == stratum) & (agg["arm"] == a)].iloc[0]
            for k in summ["K"]:
                col = f"c_k{k}"
                if col in row:
                    xcheck.append(abs(float(row[col]) - st[a][f"c{k}"]))
            xcheck.append(abs(float(row["m1_random_success"]) - st[a]["m1"]))
    max_dev = float(max(xcheck)) if xcheck else 0.0
    log(f"[analyze] metrics.analyze vs independent implementation: max |diff| = {max_dev:.3g}")
    counts = {
        "records": int(len(df)), "missing": int(missing),
        "error_unresolved": int((df["error_status"] == "unresolved").sum()),
        "sim_retries": int((df["sim_retries"] > 0).sum()),
        "invalid_action": int(df["invalid_action"].sum()),
        "termination_reason": {f"{a}/{s}": df[(df.arm == a) & (df.stratum == s)]["termination_reason"].value_counts().to_dict()
                               for a in H.ARM_NAMES for s in ("reset", "mid")},
        "clipped_steps_total": int(df["clipped_steps"].sum()),
        "duplicate_candidates_env_chunk_tol1e-6": _dup_rate(cfg, bank),
    }
    cost = _cost(cfg, df, bank, man)
    update_manifest(cfg, records_path="records.parquet", analysis_summary={"validator": {"ok": ok, "errors": errs, "n_warnings": len(warns)},
                                           "metrics_vs_independent_max_abs_diff": max_dev,
                                           "summary": summ, "m3_preview": m3, "counts": counts,
                                           "exploratory": cfg["phase"] != "confirmation"},
                    cost=cost)
    log_stage(cfg, "analyze", {"records": len(df), "missing": missing, "validator_ok": ok})
    # optional report.py (written by another agent)
    rp = REPO / "mip" / "intent_selection" / "report.py"
    if rp.exists() and not args.no_report:
        try:
            import importlib

            rep = importlib.import_module("mip.intent_selection.report")
            if hasattr(rep, "make_report"):
                with contextlib.redirect_stdout(io.StringIO()):
                    r = rep.make_report(out_dir)
                log(f"[analyze] report.py make_report({out_dir}) done: {str(r)[:300]}")
            else:
                log("[analyze] report.py has no make_report(); skipped")
        except Exception as e:  # noqa: BLE001
            log(f"[analyze] report.py failed: {type(e).__name__}: {e}")
    print(json.dumps({"summary": summ, "m3_preview": m3, "counts": counts, "cost": cost}, indent=1, default=str))
    return df


def _dup_rate(cfg, bank) -> dict:
    from mip.intent_selection.candidates import load_pool

    out = {}
    for a in H.ARM_NAMES:
        n_pairs = n_dup = 0
        for s in bank.states:
            c = load_pool(cfg, s, a)["env_action_chunk"].reshape(int(cfg["candidate_pool_size"]), -1)
            d = np.abs(c[:, None] - c[None]).max(-1)
            iu = np.triu_indices(len(c), 1)
            n_pairs += len(iu[0])
            n_dup += int((d[iu] < 1e-6).sum())
        out[a] = {"pairs": n_pairs, "duplicates": n_dup}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["bank", "candidates", "branch", "analyze", "all"])
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--order-seed", type=int, default=None, help="permute branch job order")
    ap.add_argument("--max-jobs", type=int, default=None, help="run at most N branch jobs (interruption test)")
    ap.add_argument("--out-dir", default=None, help="override output dir (identity unaffected)")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)

    ov = {"output_dir": args.out_dir} if args.out_dir else None
    cfg = H.load_config(args.config, ov)
    if args.num_workers:
        cfg["runtime"]["num_workers"] = args.num_workers
    H.silence_logs()
    H.setup_inference_process(cfg)
    out_dir = H.output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "run_log.txt", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"=== stage={args.stage} config={args.config} out={out_dir} config_hash={H.config_hash(cfg)}")
    ensure_manifest(cfg)
    stages = ["bank", "candidates", "branch", "analyze"] if args.stage == "all" else [args.stage]
    for st in stages:
        {"bank": stage_bank, "candidates": stage_candidates, "branch": stage_branch,
         "analyze": stage_analyze}[st](cfg, args, log)


if __name__ == "__main__":
    main()
