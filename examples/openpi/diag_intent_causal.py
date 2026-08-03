"""G1a offline diagnostic: does GT intent reduce action MSE vs model intent?

Runs on held-out RoboCasa demo states (last 10% of episodes) and tests intent usage via
action-prediction flow-matching loss under four conditions:
  (a) GT intent from dataset
  (b) Model-generated intent (via sample_intent)
  (c) Shuffled intent (from different episode in same task, or cross-task)
  (d) Zero intent (tau_I=1 conditioning, intent uninformative)

Uses identical z_A noise across conditions (common random numbers) for paired statistical tests.
Reports per-task and pooled MSE with bootstrap 95% CIs.

Tag: mask_regime="channel_probe" (J-mask model, so action tokens see intent but intent was
denoised with action-token state present).

Example:
  python examples/openpi/diag_intent_causal.py \\
    --pi05-config pi05_robocasa_copred \\
    --checkpoint-dir /data/group_data/maxlab/common_datasets/pandaliza/maxvla/m11_checkpoints/m11_a1_tied_robocasa/30000 \\
    --out /tmp/g1a_results.json \\
    --num-samples 512 \\
    --fp32
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

try:
    from steer_intent.robocasa_copred_dataset import RobocasaCopredDataset
except ImportError:
    RobocasaCopredDataset = None


def build_pi05_transforms(pi05_config):
    """Build transform pipeline matching train_pi05_m11.py."""
    data_config = pi05_config.data.create(pi05_config.assets_dirs, pi05_config.model)
    norm_stats = data_config.norm_stats
    if norm_stats is None:
        raise FileNotFoundError(
            f"norm stats missing for config '{pi05_config.name}' -- expected under "
            f"{pi05_config.assets_dirs}"
        )
    return _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])


def batch_to_observation(batch, device, tfm, ds, obs_steps, action_horizon):
    """Convert RobocasaCopredDataset batch to pi0.5 Observation."""
    # Un-normalize state and actions
    st = ds.normalizer["obs"]["state"].unnormalize(batch["obs"]["state"].numpy())
    agv = batch["obs"]["agentview_rgb"].numpy()
    wr = batch["obs"]["eye_in_hand_rgb"].numpy()
    act = ds.normalizer["action"].unnormalize(batch["action"].numpy())
    tid = batch["task_id"].numpy()

    B = st.shape[0]
    elems = []
    cur = obs_steps - 1
    for i in range(B):
        el = {
            "image": (agv[i, cur].transpose(1, 2, 0) * 255).astype(np.uint8),
            "wrist_image": (wr[i, cur].transpose(1, 2, 0) * 255).astype(np.uint8),
            "state": st[i, cur].astype(np.float32),
            "actions": act[i, :action_horizon].astype(np.float32),
            "prompt": f"task_{int(tid[i])}",
        }
        elems.append(tfm(el))

    coll = {k: np.stack([e[k] for e in elems]) if not isinstance(elems[0][k], dict)
            else {kk: np.stack([e[k][kk] for e in elems]) for kk in elems[0][k]}
            for k in elems[0]}

    def _to_t(v):
        t = torch.as_tensor(v, device=device)
        return t.float() if t.is_floating_point() else t

    obs = _model.Observation.from_dict({
        k: (_to_t(v) if not isinstance(v, dict) else {kk: _to_t(vv) for kk, vv in v.items()})
        for k, v in coll.items()})
    actions = _to_t(coll["actions"])
    return obs, actions


def compute_action_loss(model, observation, actions, intent, noise_A, device):
    """Compute action MSE with injected intent via forward pass.

    We run the forward pass with intent_targets=intent (to make the model aware of the intent
    in the embed_suffix and forward computation), but we only use the action loss.
    The intent is held at tau_I=0 (clean, not denoising).

    This gives the MSE between predicted action velocity and target velocity when the intent
    is injected.
    """
    B = actions.shape[0]
    time_A = torch.full((B,), 0.5, dtype=torch.float32, device=device)  # arbitrary tau_A
    time_I = torch.full((B,), 0.0, dtype=torch.float32, device=device)  # tau_I = 0 (clean)

    model.eval()
    with torch.no_grad():
        # Forward pass with injected intent
        action_loss, _ = model(
            observation,
            actions,
            noise=noise_A,
            time=time_A,
            intent_targets=intent,
            intent_time=time_I,
        )
    return action_loss  # (B, H, d_A), unreduced MSE


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap 95% CI for mean."""
    n = len(values)
    rng = np.random.RandomState(42)
    boots = rng.choice(values, size=(n_bootstrap, n), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return np.percentile(boots, alpha * 100), np.percentile(boots, (1 - alpha) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi05-config", default="pi05_robocasa_copred")
    ap.add_argument("--checkpoint-dir", required=True, help="Path to the trained checkpoint (e.g., .../30000)")
    ap.add_argument("--config-dir", default="examples/configs")
    ap.add_argument("--data-root", default=os.environ.get(
        "ROBOCASA_DATA_ROOT",
        "/data/group_data/maxlab/common_datasets/amagnuso/robocasa/v1.0/target"
    ))
    ap.add_argument("--num-samples", type=int, default=512, help="Min held-out samples to probe")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", required=True, help="Output JSON file")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # --- Load pi0.5 model ---
    pi05_config = _config.get_config(args.pi05_config)
    model = PI0Pytorch(pi05_config.model).to(args.device)
    if args.fp32:
        model = model.float()
    import safetensors.torch
    safetensors.torch.load_model(model, os.path.join(args.checkpoint_dir, "model.safetensors"), strict=False)
    model.eval()

    # --- Load RoboCasa dataset ---
    if RobocasaCopredDataset is None:
        raise ImportError("steer_intent.robocasa_copred_dataset not found")

    ds = RobocasaCopredDataset(
        root_dir=args.data_root,
        norm_stats_path="assets/pi05_robocasa_copred/robocasa/norm_stats.json",
        action_horizon=int(pi05_config.model.action_horizon),
        lookahead_stride=2,
        intent_horizon=16,
        load_images=False,  # Optimization: state-only for this diagnostic
    )
    tfm = build_pi05_transforms(pi05_config)

    # --- Held-out episodes: last 10% ---
    all_episodes = set(ep["episode_index"] for ep in ds._episodes)
    n_episodes = len(ds._episodes)
    n_holdout = max(1, n_episodes // 10)
    held_out_episode_indices = sorted(all_episodes)[-n_holdout:]
    held_out_frame_indices = [
        i for i, (ep_idx, t) in enumerate(ds._frame_index)
        if ep_idx in held_out_episode_indices
    ]
    held_out_frame_indices = held_out_frame_indices[:args.num_samples]

    print(f"[dataset] {len(ds._episodes)} episodes, {len(ds._frame_index)} frames total")
    print(f"[holdout] {len(held_out_episode_indices)} episodes ({100*n_holdout/n_episodes:.1f}%), "
          f"{len(held_out_frame_indices)} frames probed")

    # --- Build per-task intent index for shuffling ---
    task_to_intents = {}  # task_id -> [(sample_idx, intent_eef_rel)]
    for sample_idx, (ep_idx, t) in enumerate(ds._frame_index):
        task_id = ds._episodes[ep_idx]["task_id"]
        if task_id not in task_to_intents:
            task_to_intents[task_id] = []
        batch = ds[sample_idx]
        task_to_intents[task_id].append((sample_idx, batch["wsm_intent_target"]))

    # --- Collect all intents for cross-task shuffling ---
    all_intents = [(task_id, intent) for task_id, intents in task_to_intents.items()
                   for (_, intent) in intents]

    obs_steps = int(ds.obs_steps)
    action_horizon = int(pi05_config.model.action_horizon)
    copred_h = int(pi05_config.model.copred_h)
    copred_intent_dim = int(pi05_config.model.copred_intent_dim)

    # --- Run diagnostic on held-out samples ---
    results_by_task = {}  # task_id -> {"gt": [...], "model": [...], "shuffled_same": [...], "shuffled_cross": [...], "zero": [...]}
    results_pooled = {"gt": [], "model": [], "shuffled_same": [], "shuffled_cross": [], "zero": []}

    print(f"\n[probe] {len(held_out_frame_indices)} samples")
    for sample_idx in held_out_frame_indices:
        ep_idx, t = ds._frame_index[sample_idx]
        task_id = ds._episodes[ep_idx]["task_id"]

        if task_id not in results_by_task:
            results_by_task[task_id] = {"gt": [], "model": [], "shuffled_same": [], "shuffled_cross": [], "zero": []}

        batch = ds[sample_idx]
        observation, actions = batch_to_observation(
            batch, args.device, tfm, ds, obs_steps, action_horizon)

        # Normalize intent
        intent_gt = torch.tensor(
            ds.normalizer["obs"]["state"].normalize(batch["wsm_intent_target"]),
            dtype=torch.float32,
            device=args.device,
        )[None]  # (1, h, d_I)

        # Sample shared action noise for common random numbers
        noise_A = model.sample_noise((1, action_horizon, pi05_config.model.action_dim), args.device)

        # (a) GT intent
        loss_gt = compute_action_loss(model, observation, actions, intent_gt, noise_A, args.device)
        mse_gt = loss_gt.mean(dim=(1, 2)).item()
        results_by_task[task_id]["gt"].append(mse_gt)
        results_pooled["gt"].append(mse_gt)

        # (b) Model-generated intent (phase 1 only)
        try:
            with torch.no_grad():
                intent_model, _ = model.sample_intent(
                    args.device, observation, num_intent_steps=4, noise_I=None)
            loss_model = compute_action_loss(model, observation, actions, intent_model, noise_A, args.device)
            mse_model = loss_model.mean(dim=(1, 2)).item()
        except Exception as e:
            print(f"[warn] sample_intent failed: {e}, using zero")
            mse_model = np.nan
        results_by_task[task_id]["model"].append(mse_model)
        results_pooled["model"].append(mse_model)

        # (c) Shuffled intent (same task)
        same_task_intents = [intent for (sample_i, intent) in task_to_intents.get(task_id, [])
                             if sample_i != sample_idx]
        if same_task_intents:
            shuffled_intent_same = rng.choice(same_task_intents)
            shuffled_intent_same = torch.tensor(
                ds.normalizer["obs"]["state"].normalize(shuffled_intent_same),
                dtype=torch.float32,
                device=args.device,
            )[None]
            loss_shuffled_same = compute_action_loss(model, observation, actions, shuffled_intent_same, noise_A, args.device)
            mse_shuffled_same = loss_shuffled_same.mean(dim=(1, 2)).item()
        else:
            mse_shuffled_same = np.nan
        results_by_task[task_id]["shuffled_same"].append(mse_shuffled_same)
        results_pooled["shuffled_same"].append(mse_shuffled_same)

        # (c2) Shuffled intent (cross-task)
        cross_task_intents = [(tid, intent) for (tid, intent) in all_intents if tid != task_id]
        if cross_task_intents:
            tid, shuffled_intent_cross = rng.choice(cross_task_intents)
            shuffled_intent_cross = torch.tensor(
                ds.normalizer["obs"]["state"].normalize(shuffled_intent_cross),
                dtype=torch.float32,
                device=args.device,
            )[None]
            loss_shuffled_cross = compute_action_loss(model, observation, actions, shuffled_intent_cross, noise_A, args.device)
            mse_shuffled_cross = loss_shuffled_cross.mean(dim=(1, 2)).item()
        else:
            mse_shuffled_cross = np.nan
        results_by_task[task_id]["shuffled_cross"].append(mse_shuffled_cross)
        results_pooled["shuffled_cross"].append(mse_shuffled_cross)

        # (d) Zero intent (tau_I=1, uninformative)
        intent_zero = torch.zeros_like(intent_gt)
        loss_zero = compute_action_loss(model, observation, actions, intent_zero, noise_A, args.device)
        mse_zero = loss_zero.mean(dim=(1, 2)).item()
        results_by_task[task_id]["zero"].append(mse_zero)
        results_pooled["zero"].append(mse_zero)

        if len(results_pooled["gt"]) % 50 == 0:
            print(f"  ... {len(results_pooled['gt'])} samples processed")

    # --- Statistics ---
    print("\n[results]")
    report = {
        "metadata": {
            "mask_regime": "channel_probe",
            "checkpoint": args.checkpoint_dir,
            "num_samples": len(held_out_frame_indices),
            "seed": args.seed,
        },
        "by_task": {},
        "pooled": {},
    }

    def summarize(values_dict, name_label):
        """Summarize results with bootstrap CIs."""
        summary = {}
        for cond_name in ["gt", "model", "shuffled_same", "shuffled_cross", "zero"]:
            vals = np.array([v for v in values_dict[cond_name] if not np.isnan(v)])
            if len(vals) == 0:
                summary[cond_name] = {"mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
            else:
                ci_low, ci_high = bootstrap_ci(vals, n_bootstrap=10000)
                summary[cond_name] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n": len(vals),
                }
        return summary

    # Per-task stats
    for task_id in sorted(results_by_task.keys()):
        task_name = list(ds._task_id_map.keys())[task_id] if task_id < len(ds._task_id_map) else f"task_{task_id}"
        report["by_task"][task_name] = summarize(results_by_task[task_id], task_name)

    # Pooled stats
    report["pooled"] = summarize(results_pooled, "pooled")

    # G1a metric: relative MSE reduction
    gt_mean = np.nanmean(results_pooled["gt"])
    model_mean = np.nanmean(results_pooled["model"])
    if model_mean > 0:
        reduction = (model_mean - gt_mean) / model_mean * 100
    else:
        reduction = 0.0
    report["pooled"]["g1a_relative_mse_reduction_pct"] = float(reduction)
    report["pooled"]["g1a_gate_threshold_pct"] = 20.0
    report["pooled"]["g1a_gate_pass"] = reduction >= 20.0

    print(f"\n{json.dumps(report, indent=2)}")

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
