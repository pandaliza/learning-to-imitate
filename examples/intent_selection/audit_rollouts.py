"""Rollout audit for the intent-selection experiment (spec §4 A1, A3 + native reproduction).

Usage (each step appends to outputs/intent_selection/audit/rollout_audit.json):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/intent_selection/audit_rollouts.py step0
    ... a1 | spread | a3 | quat | md

step0 : native eval path (mirrors train_robomimic.eval) for ACTION-zero, ACTION-stochastic
        and INTENT, 25 episodes each, env seeds 0..24, horizon 400, NFE 9. Also stores the
        5 pilot states (ACTION-zero rollout, env seed 0, t = 0/40/80/120/160).
a1    : lockstep native-wrapper vs adapter rollouts, identical env seeds and injected noise.
spread: 16 stochastic candidates at the 5 pilot states (ACTION-stochastic, INTENT).
a3    : intent causality audit at the 5 pilot states.
quat  : training-time intent quaternion representation check (dataset, intent_type=mean).
md    : write rollout_audit.md tables from the JSON.
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import CKPT, REPO, build_config, get_dataset, load_agent, make_single_env, strict_load_report  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_float32_matmul_precision("high")  # as in train_robomimic.py

from mip.intent_selection.adapters import DeterministicNoiseSampler, PolicyAdapter, keyed_normal  # noqa: E402

OUT_DIR = Path(REPO) / "outputs/intent_selection/audit"
OUT_JSON = OUT_DIR / "rollout_audit.json"
PILOT_NPZ = OUT_DIR / "rollout_audit_pilot_states.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EP = 25
NFE = 9
PILOT_T = [0, 40, 80, 120, 160]
KEYS = DeterministicNoiseSampler()


def load_json():
    if OUT_JSON.exists():
        return json.loads(OUT_JSON.read_text())
    return {}


def save_section(name, data):
    d = load_json()
    d[name] = data
    OUT_JSON.write_text(json.dumps(d, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"[saved] {name} -> {OUT_JSON}")


# ─────────────────────────────────────────────────────────────── Step 0 / Step 1
def native_vec_rollout(arm, sample_mode, n_ep=N_EP, seed=0, record_pilot=False, torch_seed=0):
    """Mirror of examples/train/train_robomimic.py::eval for state obs (no viz extras)."""
    from mip.envs.robomimic.robomimic_env import make_vec_env

    cfg = build_config(arm, device=DEVICE, num_steps=NFE, extra=[f"task.num_envs={n_ep}"])
    agent = load_agent(arm, cfg)
    ds = get_dataset(cfg)
    envs = make_vec_env(cfg.task, seed=seed)  # env idx i is seeded seed+i at first reset
    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)
    obs, _ = envs.reset()
    ep_reward = np.zeros(n_ep)
    first_success_t = np.full(n_ep, -1)
    pilot = {}
    t = 0
    t0 = time.time()
    while t < cfg.task.max_episode_steps:
        obs_n = ds.normalizer["obs"]["state"].normalize(obs.astype(np.float32))
        obs_tensor = torch.tensor(obs_n, device=DEVICE, dtype=torch.float32)
        if record_pilot and t in PILOT_T:
            pilot[t] = {"obs_norm": obs_n[0].copy(), "obs_raw": obs[0].astype(np.float32).copy()}
        act_0 = torch.randn((n_ep, cfg.task.horizon, cfg.task.act_dim), device=DEVICE)
        if arm == "intent":
            act_normed = agent.sample(obs=obs_tensor, use_ema=True, num_steps=NFE)
        else:
            act_normed = agent.sample(act_0=act_0, obs={"state": obs_tensor}, num_steps=NFE,
                                      use_ema=True, sample_mode=sample_mode)
        act = ds.normalizer["action"].unnormalize(act_normed.detach().cpu().numpy())
        s0 = cfg.task.obs_steps - 1
        act = act[:, s0:s0 + cfg.task.act_steps]
        act = ds.undo_transform_action(act)
        obs, reward, terminated, truncated, info = envs.step(act)
        ep_reward += reward
        t += cfg.task.act_steps
        newly = (reward > 0) & (first_success_t < 0)
        first_success_t[newly] = t
    envs.close()
    succ = (ep_reward > 0).astype(int)
    print(f"[{arm}/{sample_mode}] success {succ.mean():.3f} ({succ.sum()}/{n_ep}) in {time.time()-t0:.0f}s")
    res = {
        "arm": arm, "sample_mode": sample_mode, "nfe": NFE, "n_episodes": n_ep,
        "env_seeds": list(range(seed, seed + n_ep)), "torch_seed": torch_seed,
        "episode_horizon": int(cfg.task.max_episode_steps),
        "success_rate": float(succ.mean()), "n_success": int(succ.sum()),
        "per_episode_success": succ.tolist(), "per_episode_reward_sum": ep_reward.tolist(),
        "first_success_chunk_end_step": first_success_t.tolist(),
        "network": {"num_layers": int(cfg.network.num_layers), "emb_dim": int(cfg.network.emb_dim),
                    "arch_variant": str(getattr(cfg.network, "arch_variant", "flow_action"))},
        "config_sample_mode": str(cfg.optimization.sample_mode),
    }
    return res, pilot


def step0():
    out = {}
    cfg = build_config("action", device=DEVICE)
    out["strict_load"] = {}
    for arm in ["action", "intent"]:
        c = build_config(arm, device=DEVICE)
        a = load_agent(arm, c)
        rep, ts = strict_load_report(arm, a)
        out["strict_load"][arm] = {"overrides": f"task={'square_mh_state' if arm=='action' else 'square_mh_state_flow_intent'} "
                                   f"network={'mlp' if arm=='action' else 'mlp_flow_intent'}",
                                   "num_layers": int(c.network.num_layers), "emb_dim": int(c.network.emb_dim),
                                   "modules": rep, "saved_best_metrics": ts.get("best_metrics"),
                                   "training_step": ts.get("n_gradient_step")}
    r, pilot = native_vec_rollout("action", "zero", record_pilot=True)
    out["action_zero_native"] = r
    np.savez(PILOT_NPZ, **{f"obs_norm_t{t}": pilot[t]["obs_norm"] for t in PILOT_T},
             **{f"obs_raw_t{t}": pilot[t]["obs_raw"] for t in PILOT_T})
    out["pilot_states"] = {"source": "ACTION zero-mode native rollout, env seed 0", "t": PILOT_T,
                           "file": str(PILOT_NPZ)}
    r, _ = native_vec_rollout("action", "stochastic")
    out["action_stochastic"] = r
    r, _ = native_vec_rollout("intent", None)
    out["intent_native"] = r
    save_section("step0_native", out)


# ───────────────────────────────────────────────────────────────────── helpers
def make_adapter(arm, sample_mode=None):
    cfg = build_config(arm, device=DEVICE, num_steps=NFE)
    ds = get_dataset(cfg)
    ad = PolicyAdapter(CKPT[arm], cfg, arm_name=arm, device=DEVICE,
                       action_normalizer=ds.normalizer["action"],
                       undo_transform_action=ds.undo_transform_action,
                       action_sample_mode=sample_mode, num_steps=NFE)
    return ad, cfg, ds


def load_pilot():
    z = np.load(PILOT_NPZ)
    return [torch.tensor(z[f"obs_norm_t{t}"], dtype=torch.float32) for t in PILOT_T]


def chunk_spread(chunks):
    """chunks: (K, act_steps, D) -> mean per-dim std across K, max pairwise L2 (flattened)."""
    c = np.asarray(chunks, dtype=np.float64)
    K = c.shape[0]
    flat = c.reshape(K, -1)
    d = np.linalg.norm(flat[:, None] - flat[None], axis=-1)
    return {"mean_per_dim_std": float(c.std(axis=0).mean()), "max_pairwise_l2": float(d.max()),
            "mean_pairwise_l2": float(d[np.triu_indices(K, 1)].mean()),
            "per_dim_std_mean_over_t": c.std(axis=0).mean(axis=0).tolist()}


def spread():
    pilot = load_pilot()
    out = {}
    for arm, mode in [("action", "stochastic"), ("action", "zero"), ("intent", None)]:
        ad, cfg, ds = make_adapter(arm, mode)
        rows = []
        for si, obs in enumerate(pilot):
            seeds = [KEYS.get_noise_seed("audit_spread", arm, "sq_mh_seed0", si, k) for k in range(16)]
            cands = ad.sample_batch(obs, seeds)
            phys10 = np.stack([c.physical_action_chunk.numpy() for c in cands])
            env7 = np.stack([c.env_action_chunk for c in cands])
            norm = np.stack([c.normalized_action_chunk.cpu().numpy() for c in cands])
            rows.append({"state_t": PILOT_T[si], "physical10": chunk_spread(phys10),
                         "env7": chunk_spread(env7), "normalized": chunk_spread(norm),
                         "pos_only_max_pairwise_l2_m": chunk_spread(phys10[..., :3])["max_pairwise_l2"]})
        label = f"{arm}_{mode}" if arm == "action" else "intent"
        out[label] = {"per_state": rows,
                      "avg_mean_per_dim_std_physical10": float(np.mean([r["physical10"]["mean_per_dim_std"] for r in rows])),
                      "avg_max_pairwise_l2_physical10": float(np.mean([r["physical10"]["max_pairwise_l2"] for r in rows])),
                      "avg_mean_per_dim_std_env7": float(np.mean([r["env7"]["mean_per_dim_std"] for r in rows])),
                      "avg_max_pairwise_l2_env7": float(np.mean([r["env7"]["max_pairwise_l2"] for r in rows]))}
        print(label, {k: v for k, v in out[label].items() if k != "per_state"})
    save_section("step1_spread", out)


# ─────────────────────────────────────────────────────────────────────── A1
class BCPolicy:
    """Verbatim logic of examples/eval/eval_best_of_n.py::BCPolicy (lines 67-87), state obs."""

    def __init__(self, agent, config, device):
        self.agent, self.config, self.device = agent, config, device
        self._s = config.task.obs_steps - 1

    def act(self, obs_tensor):
        B = obs_tensor.shape[0]
        act_0 = torch.randn(B, self.config.task.horizon, self.config.task.act_dim, device=self.device)
        with torch.no_grad():
            act_normed = self.agent.sample(act_0=act_0, obs={"state": obs_tensor}, use_ema=True)
        return act_normed[:, self._s: self._s + self.config.task.act_steps].cpu().numpy()


class FlowIntentPolicy:
    """Verbatim logic of examples/eval/eval_best_of_n.py::FlowIntentPolicy (lines 90-101)."""

    def __init__(self, agent, config):
        self.agent, self.config = agent, config
        self._s = config.task.obs_steps - 1

    def act(self, obs_tensor):
        with torch.no_grad():
            act_normed = self.agent.sample(obs=obs_tensor, use_ema=True)
        return act_normed[:, self._s: self._s + self.config.task.act_steps].cpu().numpy()


@contextmanager
def inject_randn(noise):
    """Make the next torch.randn call return `noise` (shape-checked); any extra call errors."""
    orig = torch.randn
    calls = {"n": 0}

    def fake(*args, **kwargs):
        shape = tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)) else tuple(args)
        assert shape == tuple(noise.shape), f"randn shape {shape} != injected {tuple(noise.shape)}"
        calls["n"] += 1
        assert calls["n"] == 1, "native policy called torch.randn more than once"
        return noise.clone().to(kwargs.get("device", noise.device))

    torch.randn = fake
    try:
        yield calls
    finally:
        torch.randn = orig


def a1_arm(arm, sample_mode, n_ep, seeds):
    cfg = build_config(arm, device=DEVICE, num_steps=NFE,
                       extra=[f"optimization.sample_mode={sample_mode}"] if sample_mode else [])
    ds = get_dataset(cfg)
    agent = load_agent(arm, cfg)
    native = FlowIntentPolicy(agent, cfg) if arm == "intent" else BCPolicy(agent, cfg, DEVICE)
    ad = PolicyAdapter(CKPT[arm], cfg, arm_name=arm, device=DEVICE,
                       action_normalizer=ds.normalizer["action"],
                       undo_transform_action=ds.undo_transform_action,
                       action_sample_mode=sample_mode, num_steps=NFE)
    obs_norm_fn = ds.normalizer["obs"]["state"].normalize
    eps = []
    for e, seed in enumerate(seeds[:n_ep]):
        env_n, env_a = make_single_env(cfg, seed), make_single_env(cfg, seed)
        obs_n, _ = env_n.reset()
        obs_a, _ = env_a.reset()
        max_d = {"normalized": 0.0, "physical10": 0.0, "env7": 0.0, "obs": float(np.abs(obs_n - obs_a).max()),
                 "intent_noise_calls": 0}
        rew_n = rew_a = 0.0
        t = 0
        k = 0
        while t < cfg.task.max_episode_steps:
            noise = keyed_normal(ad.noise_shape(), KEYS.get_noise_seed("audit_a1", arm, str(sample_mode), e, k), DEVICE)
            on = torch.tensor(obs_norm_fn(obs_n.astype(np.float32)), device=DEVICE)[None]
            oa = torch.tensor(obs_norm_fn(obs_a.astype(np.float32)), device=DEVICE)
            with inject_randn(noise[None]) as calls:
                nat_norm = native.act(on)[0]  # (act_steps, 10) normalized
            max_d["intent_noise_calls"] += calls["n"]
            cand = ad.sample_candidate(oa, initial_noise=noise)
            nat_phys10 = ds.normalizer["action"].unnormalize(nat_norm)
            nat_env7 = ds.undo_transform_action(nat_phys10)
            max_d["normalized"] = max(max_d["normalized"], float(np.abs(nat_norm - cand.normalized_action_chunk.cpu().numpy()).max()))
            max_d["physical10"] = max(max_d["physical10"], float(np.abs(nat_phys10 - cand.physical_action_chunk.numpy()).max()))
            max_d["env7"] = max(max_d["env7"], float(np.abs(nat_env7 - cand.env_action_chunk).max()))
            # step one physical action at a time so every per-step obs is compared
            # (MultiStepWrapper semantics for history/termination are unchanged).
            for j in range(nat_env7.shape[0]):
                obs_n, rn, _, _, _ = env_n.step(nat_env7[j:j + 1])
                obs_a, ra, _, _, _ = env_a.step(cand.env_action_chunk[j:j + 1])
                rew_n += rn
                rew_a += ra
                max_d["obs"] = max(max_d["obs"], float(np.abs(obs_n[-1] - obs_a[-1]).max()))
            t += cfg.task.act_steps
            k += 1
        sn, sa = int(rew_n > 0), int(rew_a > 0)
        eps.append({"env_seed": seed, "native_success": sn, "adapter_success": sa, "n_replans": k, **max_d})
        print(f"[A1 {arm}/{sample_mode}] ep{e} seed{seed} succ {sn}/{sa} maxdiff norm {max_d['normalized']:.2e} "
              f"phys {max_d['physical10']:.2e} env7 {max_d['env7']:.2e} obs {max_d['obs']:.2e}")
        env_n.close()
        env_a.close()
    agg = {k: float(max(ep[k] for ep in eps)) for k in ["normalized", "physical10", "env7", "obs"]}
    passed = all(ep["native_success"] == ep["adapter_success"] for ep in eps) and agg["obs"] <= 1e-6
    return {"arm": arm, "sample_mode": sample_mode, "episodes": eps, "max_abs_diff": agg,
            "native_sr": float(np.mean([ep["native_success"] for ep in eps])),
            "adapter_sr": float(np.mean([ep["adapter_success"] for ep in eps])),
            "success_identical": all(ep["native_success"] == ep["adapter_success"] for ep in eps),
            "pass": bool(passed)}


def a1():
    seeds = list(range(10))
    out = {"tolerance_obs": 1e-6, "noise": "keyed CPU generator per (arm, mode, episode, replan); "
           "injected into native wrapper by patching torch.randn, passed explicitly to adapter",
           "native_wrapper": "eval_best_of_n.py BCPolicy/FlowIntentPolicy logic (copied; that module imports LIBERO); "
           "config.optimization.num_steps=9; env action = undo_transform_action(unnormalize(.)) as train_robomimic.eval"}
    out["action_zero"] = a1_arm("action", "zero", 10, seeds)
    out["intent"] = a1_arm("intent", None, 10, seeds)
    out["action_stochastic"] = a1_arm("action", "stochastic", 5, seeds)
    save_section("a1_parity", out)


# ─────────────────────────────────────────────────────────────────────── A3
def a3():
    from scipy.stats import pearsonr, spearmanr
    pilot = load_pilot()
    ad, cfg, ds = make_adapter("intent")
    on = ds.normalizer["obs"]["state"]
    i0, i1 = ds.intent_start, ds.intent_end
    imin, irange = on.min[i0:i1], on.range[i0:i1]

    def phys_intent(x):
        return (np.asarray(x) + 1) / 2 * irange + imin

    out = {"intent_slice_in_obs": [int(i0), int(i1)], "intent_layout": "eef_pos(3) + eef_quat(4, xyzw robosuite), "
           "mean over 8 future steps of min-max-normalized obs", "states": []}
    all_pairs_i, all_pairs_a = [], []
    for si, obs in enumerate(pilot):
        seeds = [KEYS.get_noise_seed("audit_a3", "intent", "sq_mh_seed0", si, k) for k in range(16)]
        singles = ad.sample_batch(obs, seeds, vectorized=False)
        batched = ad.sample_batch(obs, seeds, vectorized=True)
        I = np.stack([c.intent_decoded.cpu().numpy() for c in singles])  # normalized
        Ip = phys_intent(I)
        A = np.stack([c.physical_action_chunk.numpy() for c in singles])
        An = np.stack([c.normalized_action_chunk.cpu().numpy() for c in singles])
        # (d) batch vs singles; noise distinct
        d_int = float(np.abs(I - np.stack([c.intent_decoded.cpu().numpy() for c in batched])).max())
        d_act = float(np.abs(An - np.stack([c.normalized_action_chunk.cpu().numpy() for c in batched])).max())
        noise_hashes = {c.initial_noise_hash for c in singles}
        # native batched path with the same noise (FlowIntentAgent.sample_with_intent_noise)
        noise = torch.stack([c.initial_noise for c in singles])  # (16,1,7)
        obs_b = obs[None].expand(16, *obs.shape).contiguous().to(DEVICE)
        nat = ad.agent.sample_with_intent_noise(obs_b, noise, use_ema=True, num_steps=NFE)
        d_native = float(np.abs(nat[:, 1:9].cpu().numpy() - An).max())
        # (b) pairwise distances
        iu = np.triu_indices(16, 1)
        DI = np.linalg.norm(I[:, None] - I[None], axis=-1)[iu]
        DA = np.linalg.norm((A[:, None] - A[None]).reshape(16, 16, -1), axis=-1)[iu]
        DAn = np.linalg.norm((An[:, None] - An[None]).reshape(16, 16, -1), axis=-1)[iu]
        all_pairs_i.append(DI)
        all_pairs_a.append(DAn)
        # (c) manipulation: PC1 5th/95th percentile intents, obs fixed
        Ic = I - I.mean(0)
        pc1 = np.linalg.svd(Ic, full_matrices=False)[2][0]
        proj = Ic @ pc1
        lo, hi = np.percentile(proj, 5), np.percentile(proj, 95)
        ilo, ihi = int(np.argmin(np.abs(proj - lo))), int(np.argmin(np.abs(proj - hi)))
        dec = ad.decode_given_intent(obs, torch.tensor(I[[ilo, ihi]], dtype=torch.float32)).cpu().numpy()
        dec_ph = ds.normalizer["action"].unnormalize(dec)[:, 1:9]
        dec_rep = ad.decode_given_intent(obs, torch.tensor(I[[ilo, ilo]], dtype=torch.float32)).cpu().numpy()
        # swap test: obs held fixed, intent from another pilot state
        out["states"].append({
            "state_t": PILOT_T[si],
            "intents_normalized": I.tolist(), "intents_physical": Ip.tolist(),
            "action_chunks_physical10": A.tolist(),
            "a_intent_spread": {
                "norm_per_dim_std": I.std(0).tolist(), "norm_mean_std": float(I.std(0).mean()),
                "phys_pos_std_m": Ip[:, :3].std(0).tolist(), "phys_quat_std": Ip[:, 3:].std(0).tolist(),
                "max_pairwise_l2_norm": float(DI.max()),
                "max_pairwise_pos_l2_m": float(np.linalg.norm(Ip[:, None, :3] - Ip[None, :, :3], axis=-1).max()),
                "decoded_quat_norm_min_max": [float(np.linalg.norm(Ip[:, 3:], axis=1).min()),
                                              float(np.linalg.norm(Ip[:, 3:], axis=1).max())]},
            "b_corr": {"pearson_phys": float(pearsonr(DI, DA)[0]), "spearman_phys": float(spearmanr(DI, DA)[0]),
                       "pearson_norm": float(pearsonr(DI, DAn)[0]), "spearman_norm": float(spearmanr(DI, DAn)[0])},
            "c_manipulation": {
                "idx_p5": ilo, "idx_p95": ihi,
                "intent_p5_phys": Ip[ilo].tolist(), "intent_p95_phys": Ip[ihi].tolist(),
                "intent_l2_norm": float(np.linalg.norm(I[ilo] - I[ihi])),
                "intent_pos_l2_m": float(np.linalg.norm(Ip[ilo, :3] - Ip[ihi, :3])),
                "action_chunk_l2_phys10": float(np.linalg.norm(dec_ph[0] - dec_ph[1])),
                "action_pos_maxabs_m": float(np.abs(dec_ph[0, :, :3] - dec_ph[1, :, :3]).max()),
                "action_final_pos_p5": dec_ph[0, -1, :3].tolist(), "action_final_pos_p95": dec_ph[1, -1, :3].tolist(),
                "same_intent_twice_maxabs": float(np.abs(dec_rep[0] - dec_rep[1]).max())},
            "d_batch_vs_single": {"intent_maxabs": d_int, "action_norm_maxabs": d_act,
                                  "adapter_vs_native_batched_sample_with_intent_noise_maxabs": d_native,
                                  "distinct_noise_hashes": len(noise_hashes)},
        })
        s = out["states"][-1]
        print(f"[A3] t={PILOT_T[si]} intent std {s['a_intent_spread']['norm_mean_std']:.4f} "
              f"corr {s['b_corr']} manip actL2 {s['c_manipulation']['action_chunk_l2_phys10']:.4f} "
              f"batch {s['d_batch_vs_single']}")
    DI, DA = np.concatenate(all_pairs_i), np.concatenate(all_pairs_a)
    out["b_corr_pooled_norm"] = {"pearson": float(pearsonr(DI, DA)[0]), "spearman": float(spearmanr(DI, DA)[0]),
                                 "n_pairs": int(len(DI))}
    # obs-swap: decode pilot-state-0 intents under state-2 obs; action must change with obs too
    out["e_leakage"] = {
        "inference_inputs": "PolicyAdapter._sample(obs_in, noise): encoder(obs_in) -> intent ODE from noise -> "
                            "action_decoder(obs_emb, intent). obs_in = MultiStepWrapper._get_obs(n_obs_steps=2): "
                            "the current and previous observation only (mip/env_utils.py).",
        "ground_truth_intent": "Only built in RobomimicDataset.sample_to_data (state[obs_steps:obs_steps+intent_horizon]) "
                               "and consumed by FlowIntentAgent.update(intent_gt=...). FlowIntentAgent.sample / "
                               "sample_with_intent_noise and the adapter take no intent_gt argument.",
        "future_obs": "No env/dataset future access in adapter or FlowIntentAgent.sample; the only dataset object "
                      "used at inference is the normalizer (min/max statistics) and undo_transform_action.",
    }
    save_section("a3_intent_causality", out)


# ─────────────────────────────────────────────────────────── quat representation
def quat():
    cfg = build_config("intent", device=DEVICE)
    ds = get_dataset(cfg)
    obs = ds.replay_buffer["obs"][:]
    ends = ds.replay_buffer.episode_ends[:]
    q = obs[:, 17:21].astype(np.float64)
    starts = np.concatenate([[0], ends[:-1]])
    flips, windows, mean_norms, hemis = 0, 0, [], 0
    H = 8
    for s, e in zip(starts, ends):
        qq = q[s:e]
        dots = np.sum(qq[1:] * qq[:-1], axis=1)
        flips += int((dots < 0).sum())
        for i in range(0, len(qq) - H):
            w = qq[i + 1:i + 1 + H]
            m = w.mean(0)
            mean_norms.append(np.linalg.norm(m))
            windows += 1
            if np.any(w @ w[0] < 0):
                hemis += 1
    mean_norms = np.array(mean_norms)
    out = {"n_frames": int(len(q)), "n_episodes": int(len(ends)),
           "consecutive_sign_flips": flips,
           "windows": windows, "windows_with_mixed_hemisphere": hemis,
           "mean_quat_norm_quantiles(min,1%,50%,max)": [float(mean_norms.min()), float(np.quantile(mean_norms, 0.01)),
                                                       float(np.median(mean_norms)), float(mean_norms.max())],
           "raw_quat_norm_min_max": [float(np.linalg.norm(q, axis=1).min()), float(np.linalg.norm(q, axis=1).max())],
           "per_component_min_max": [q.min(0).tolist(), q.max(0).tolist()],
           "note": "intent_type=mean: per-dim arithmetic mean of min-max-normalized eef_quat over obs[t+1..t+8] "
                   "(robomimic_dataset.py sample_to_data). Equals normalize(arithmetic mean of raw quats) since "
                   "min-max is affine per dim. No hemisphere alignment, no renormalization."}
    print(out)
    save_section("quat_representation", out)


# ─────────────────────────────────────────────────────────────────────── md
def md():
    d = load_json()
    L = ["# Rollout audit (square-mh-state, NFE 9, EMA)", ""]
    s0 = d.get("step0_native", {})
    if s0:
        L += ["## Success rates (25 episodes, env seeds 0-24, horizon 400)", "",
              "| Arm | Path | Sample mode | Success | Saved (50 ep) |", "|---|---|---|---|---|"]
        for k, saved in [("action_zero_native", 0.62), ("action_stochastic", "n/a (non-native)"), ("intent_native", 0.64)]:
            r = s0[k]
            L.append(f"| {r['arm'].upper()} | native agent.sample | {r['sample_mode']} | "
                     f"{r['success_rate']:.2f} ({r['n_success']}/{r['n_episodes']}) | {saved} |")
        L += ["", "| Arm | Config | Strict load |", "|---|---|---|"]
        for arm, v in s0["strict_load"].items():
            ok = all(not m["missing"] and not m["unexpected"] for m in v["modules"].values())
            L.append(f"| {arm} | {v['overrides']} (num_layers={v['num_layers']}, emb={v['emb_dim']}) | {'OK' if ok else 'FAIL'} |")
        L.append("")
    sp = d.get("step1_spread", {})
    if sp:
        L += ["## Candidate spread (16 candidates, 5 pilot states)", "",
              "| Arm | State t | mean per-dim std (phys 10-D) | max pairwise L2 (phys 10-D) | max pairwise L2 (env 7-D) | max pairwise pos L2 (m) |",
              "|---|---|---|---|---|---|"]
        for arm, v in sp.items():
            for r in v["per_state"]:
                L.append(f"| {arm} | {r['state_t']} | {r['physical10']['mean_per_dim_std']:.4g} | "
                         f"{r['physical10']['max_pairwise_l2']:.4g} | {r['env7']['max_pairwise_l2']:.4g} | "
                         f"{r['pos_only_max_pairwise_l2_m']:.4g} |")
        L.append("")
    a1d = d.get("a1_parity", {})
    if a1d:
        L += ["## A1 parity (native wrapper vs adapter, lockstep, injected noise)", "",
              "| Arm | Episodes | max abs diff normalized | physical 10-D | env 7-D | per-step obs | native SR | adapter SR | Verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k in ["action_zero", "action_stochastic", "intent"]:
            if k in a1d:
                r = a1d[k]
                m = r["max_abs_diff"]
                L.append(f"| {k} | {len(r['episodes'])} | {m['normalized']:.3g} | {m['physical10']:.3g} | {m['env7']:.3g} | "
                         f"{m['obs']:.3g} | {r['native_sr']:.2f} | {r['adapter_sr']:.2f} | {'PASS' if r['pass'] else 'FAIL'} |")
        L.append("")
    a3d = d.get("a3_intent_causality", {})
    if a3d:
        L += ["## A3 intent causality (INTENT, 16 keyed intents per state)", "",
              "| State t | intent mean std (norm) | max pairwise pos L2 (m) | quat norm range | Pearson / Spearman (intent dist vs action dist, norm) | manip: intent pos L2 (m) | manip: action L2 (phys) | manip: max pos diff (m) | batch vs single max abs | adapter vs native batched |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s in a3d["states"]:
            a, b, c, dd = s["a_intent_spread"], s["b_corr"], s["c_manipulation"], s["d_batch_vs_single"]
            L.append(f"| {s['state_t']} | {a['norm_mean_std']:.4f} | {a['max_pairwise_pos_l2_m']:.4f} | "
                     f"{a['decoded_quat_norm_min_max'][0]:.3f}-{a['decoded_quat_norm_min_max'][1]:.3f} | "
                     f"{b['pearson_norm']:.3f} / {b['spearman_norm']:.3f} | {c['intent_pos_l2_m']:.4f} | "
                     f"{c['action_chunk_l2_phys10']:.4f} | {c['action_pos_maxabs_m']:.4f} | "
                     f"{max(dd['intent_maxabs'], dd['action_norm_maxabs']):.2g} | "
                     f"{dd['adapter_vs_native_batched_sample_with_intent_noise_maxabs']:.2g} |")
        p = a3d["b_corr_pooled_norm"]
        L += ["", f"Pooled (n={p['n_pairs']} pairs): Pearson {p['pearson']:.3f}, Spearman {p['spearman']:.3f}", ""]
    pc = d.get("a3_d_precision_check", {})
    if pc:
        L += ["## A3(d) batch-16 vs 16 singles, by matmul precision (max abs diff over 5 states)", "",
              "| Device / precision | intent | action (normalized) |", "|---|---|---|"]
        for k, rows in pc.items():
            L.append(f"| {k} | {max(r['intent_maxabs'] for r in rows):.2g} | {max(r['action_maxabs'] for r in rows):.2g} |")
        L.append("")
    dg = d.get("step0_diagnostic_extra_seeds", {})
    if dg:
        L += ["## Success-rate diagnostic (extra seeds)", "", "| Run | Success |", "|---|---|"]
        for k, v in dg.items():
            if isinstance(v, float):
                L.append(f"| {k} | {v:.2f} |")
        h = dg.get("saved_eval_history_mean_success_9", {})
        for arm, xs in h.items():
            L.append(f"| saved eval history {arm} (40 ep each): mean / max | {np.mean(xs):.2f} / {max(xs):.2f} |")
        L.append("")
    q = d.get("quat_representation", {})
    if q:
        L += ["## Training intent quaternion representation (intent_type=mean)", "",
              "| Quantity | Value |", "|---|---|",
              f"| frames / episodes | {q['n_frames']} / {q['n_episodes']} |",
              f"| consecutive-frame sign flips (q·q_prev<0) | {q['consecutive_sign_flips']} |",
              f"| 8-step windows with mixed hemisphere | {q['windows_with_mixed_hemisphere']} / {q['windows']} |",
              f"| norm of averaged quat (min, 1%, median, max) | {', '.join(f'{x:.4f}' for x in q['mean_quat_norm_quantiles(min,1%,50%,max)'])} |",
              ""]
    (OUT_DIR / "rollout_audit.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    {"step0": step0, "spread": spread, "a1": a1, "a3": a3, "quat": quat, "md": md}[sys.argv[1]]()
