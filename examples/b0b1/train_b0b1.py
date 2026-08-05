"""Training script for B0/B1 from-scratch generative policies.

B0: action-only flow matching, standard loss.
B1: intent + action co-prediction with decoupled noise levels + stratified corners.

Flow matching conventions:
  x_tau = tau*eps + (1-tau)*x0
  target velocity: u = eps - x0
  network predicts: v
  loss: ||v - u||^2 per-dimension normalized

Stratified noise (B1 only):
  P(tau_I=0) = 0.25  (clean intent conditioning)
  P(tau_I=1) = 0.10  (intent uninformative)
  Otherwise: independent Beta(1.5,1) draws
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler

from b0b1.dit import create_dit
from b0b1.feature_dataset import make_feature_based_dataset
from b0b1.sample import SampleContext, sample_b0, sample_b1

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")


class NoiseSchedule:
    """Sampler for Beta(1.5,1) noise schedule with optional stratification."""

    def __init__(self, stratify_intent: bool = True):
        self.stratify_intent = stratify_intent

    def sample_tau(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample tau ~ Beta(1.5, 1) in [0, 1].

        Beta(1.5, 1) = 1.5 * (1-u)^0.5 where u ~ U[0,1]
        """
        u = torch.rand(batch_size, device=device)
        tau = 1.0 - (1.0 - u) ** (1.0 / 1.5)  # Inverse CDF
        tau = torch.clamp(tau, 0.001, 0.999)  # Clamp to valid range
        return tau

    def sample_tau_i_stratified(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample tau_I with stratification (B1 only).

        P(tau_I=0) = 0.25 (clean intent conditioning)
        P(tau_I=1) = 0.10 (intent uninformative)
        Otherwise: Beta(1.5,1)
        """
        if not self.stratify_intent:
            return self.sample_tau(batch_size, device)

        tau_i = torch.zeros(batch_size, device=device)
        probs = torch.rand(batch_size)

        # P(tau_I=0) = 0.25
        mask_clean = probs < 0.25
        tau_i[mask_clean] = 0.0

        # P(tau_I=1) = 0.10
        mask_noisy = (probs >= 0.25) & (probs < 0.35)
        tau_i[mask_noisy] = 1.0

        # Otherwise: Beta(1.5, 1)
        mask_other = ~mask_clean & ~mask_noisy
        if mask_other.any():
            u = torch.rand(mask_other.sum(), device=device)
            tau_i[mask_other] = 1.0 - (1.0 - u) ** (1.0 / 1.5)
            tau_i[mask_other] = torch.clamp(tau_i[mask_other], 0.001, 0.999)

        return tau_i


def compute_flow_loss_b0(
    model: torch.nn.Module,
    x0_action: torch.Tensor,
    tau_action: torch.Tensor,
    vision_features: torch.Tensor,
    lang_features: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """Compute flow matching loss for B0.

    L = ||v - (eps - x0)||^2 / (H * d_A)
    """
    # Sample noise
    eps = torch.randn_like(x0_action)

    # Construct noised sample: x_tau = tau*eps + (1-tau)*x0
    x_tau = tau_action.view(-1, 1, 1) * eps + (1 - tau_action.view(-1, 1, 1)) * x0_action

    # Target velocity
    u_action = eps - x0_action

    # Forward pass
    v_action, _ = model(
        x_action=x_tau,
        x_intent=None,
        tau_action=tau_action,
        tau_intent=None,
        vision_features=vision_features,
        lang_features=lang_features,
        state=state,
    )

    # Per-dimension loss
    loss = torch.mean((v_action - u_action) ** 2)
    return loss


def compute_flow_loss_b1(
    model: torch.nn.Module,
    x0_action: torch.Tensor,
    x0_intent: torch.Tensor,
    tau_action: torch.Tensor,
    tau_intent: torch.Tensor,
    vision_features: torch.Tensor,
    lang_features: torch.Tensor,
    state: torch.Tensor,
    w_action: float = 1.0,
    w_intent: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute flow matching loss for B1 with decoupled noise levels.

    L = w_A * ||v_A - u_A||^2 / (H * d_A) + w_I * ||v_I - u_I||^2 / (h * d_I)

    When tau_I=0 (stratified clean intent), intent loss is masked (already conditioned).
    """
    # Sample noise
    eps_action = torch.randn_like(x0_action)
    eps_intent = torch.randn_like(x0_intent)

    # Construct noised samples
    x_tau_action = tau_action.view(-1, 1, 1) * eps_action + (1 - tau_action.view(-1, 1, 1)) * x0_action
    x_tau_intent = tau_intent.view(-1, 1, 1) * eps_intent + (1 - tau_intent.view(-1, 1, 1)) * x0_intent

    # Target velocities
    u_action = eps_action - x0_action
    u_intent = eps_intent - x0_intent

    # Forward pass
    v_action, v_intent = model(
        x_action=x_tau_action,
        x_intent=x_tau_intent,
        tau_action=tau_action,
        tau_intent=tau_intent,
        vision_features=vision_features,
        lang_features=lang_features,
        state=state,
    )

    # Per-dimension normalized losses
    loss_action = w_action * torch.mean((v_action - u_action) ** 2)

    # Intent loss: compute for all samples, but with masking for tau_I=0
    # When tau_I=0, intent is already at its denoised value (no update needed)
    # We still compute loss but it should be zero contribution
    mask_intent = tau_intent > 0.01  # Only count loss for noisy intent
    if mask_intent.sum() > 0:
        loss_intent = w_intent * torch.mean((v_intent[mask_intent] - u_intent[mask_intent]) ** 2)
    else:
        # All intent tokens are clean (tau_I=0), no loss to compute
        loss_intent = torch.zeros(1, device=x0_action.device, dtype=x0_action.dtype)[0]

    total_loss = loss_action + loss_intent

    return total_loss, loss_action, loss_intent


def main():
    parser = argparse.ArgumentParser(description="Train B0/B1 generative policies")
    parser.add_argument("--arm", type=str, choices=["b0", "b1"], default="b0", help="b0 or b1")
    parser.add_argument("--data-root", type=str, help="RoboCasa /target directory")
    parser.add_argument(
        "--feature-cache",
        type=str,
        help="Feature cache directory (/data/group_data/.../b0b1_features/<variant>)",
    )
    parser.add_argument("--norm-stats", type=str, help="Path to norm_stats.json")
    parser.add_argument("--depth", type=int, default=6, help="DiT depth (blocks)")
    parser.add_argument("--dim", type=int, default=512, help="DiT embedding dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="DiT num heads")
    parser.add_argument("--vision-dim", type=int, default=768, help="Vision feature dimension")
    parser.add_argument("--num-vision-tokens", type=int, default=1, help="1 (pooled) or 16 (grid16)")
    parser.add_argument("--lang-dim", type=int, default=768, help="Language embedding dimension")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--steps", type=int, default=200000, help="Total training steps")
    parser.add_argument("--ckpt-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--ckpt-interval", type=int, default=25000, help="Checkpoint every N steps")
    parser.add_argument("--log-interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Paths
    data_root = Path(args.data_root) if args.data_root else Path("/data/group_data/maxlab/common_datasets/robocasa-test/target")
    feature_cache = Path(args.feature_cache) if args.feature_cache else Path(
        "/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features/pooled"
    )
    norm_stats = Path(args.norm_stats) if args.norm_stats else Path(
        "/home/ldahiya/max_vla/much-ado-about-noising/assets/pi05_robocasa_copred/robocasa/norm_stats.json"
    )
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(
        "/data/group_data/maxlab/common_datasets/pandaliza/b0b1_checkpoints"
    )

    logger.info(f"[B{args.arm.upper()}] Training starts | depth={args.depth} dim={args.dim} bs={args.batch_size}")
    logger.info(f"Data: {data_root}")
    logger.info(f"Features: {feature_cache}")
    logger.info(f"Norm stats: {norm_stats}")
    logger.info(f"Checkpoints: {ckpt_dir}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = make_feature_based_dataset(
        root_dir=data_root,
        feature_cache_dir=feature_cache,
        normalize=True,
        norm_stats_path=norm_stats,
    )
    logger.info(f"Dataset size: {len(dataset)} samples")

    # Create dataloader
    sampler = RandomSampler(dataset, replacement=False, num_samples=args.steps * args.batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    logger.info(f"Creating DiT model (arm={args.arm})...")
    model = create_dit(
        arm=args.arm,
        depth=args.depth,
        dim=args.dim,
        num_heads=args.num_heads,
        vision_dim=args.vision_dim,
        num_vision_tokens=args.num_vision_tokens,
        lang_dim=args.lang_dim,
    ).to(device)
    num_params = model.count_parameters()
    logger.info(f"DiT created | {num_params:,} trainable parameters")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # Noise schedule
    noise_schedule = NoiseSchedule(stratify_intent=(args.arm == "b1"))

    # Training loop
    model.train()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    losses = {"total": [], "action": [], "intent": []} if args.arm == "b1" else {"total": []}

    for step, batch in enumerate(dataloader):
        if step >= args.steps:
            break

        # Move batch to device
        obs_state = batch["obs"]["state"].to(device)
        vision = batch["obs"]["vision"].to(device)
        lang = batch["lang"].to(device)
        x0_action = batch["action"].to(device)

        # Extract dimensions
        B = x0_action.shape[0]
        H = x0_action.shape[1]

        # Sample noise levels
        tau_action = noise_schedule.sample_tau(B, device)

        if args.arm == "b0":
            # B0: standard flow matching
            loss = compute_flow_loss_b0(
                model,
                x0_action=x0_action,
                tau_action=tau_action,
                vision_features=vision,
                lang_features=lang,
                state=obs_state,
            )
            losses["total"].append(loss.item())

        else:
            # B1: intent + action with stratified noise
            x0_intent = batch["wsm_intent_target"].to(device)
            tau_intent = noise_schedule.sample_tau_i_stratified(B, device)

            loss, loss_action, loss_intent = compute_flow_loss_b1(
                model,
                x0_action=x0_action,
                x0_intent=x0_intent,
                tau_action=tau_action,
                tau_intent=tau_intent,
                vision_features=vision,
                lang_features=lang,
                state=obs_state,
                w_action=1.0,
                w_intent=1.0,
            )
            losses["total"].append(loss.item())
            losses["action"].append(loss_action.item())
            losses["intent"].append(loss_intent.item())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        # Logging
        if (step + 1) % args.log_interval == 0:
            if args.arm == "b0":
                logger.info(f"[Step {step+1}/{args.steps}] Loss: {loss.item():.4f}")
            else:
                logger.info(
                    f"[Step {step+1}/{args.steps}] Total: {loss.item():.4f} "
                    f"Action: {loss_action.item():.4f} Intent: {loss_intent.item():.4f}"
                )

        # Checkpointing
        if (step + 1) % args.ckpt_interval == 0:
            ckpt_path = ckpt_dir / args.arm / f"step_{step+1:06d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "losses": losses,
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    final_ckpt = ckpt_dir / args.arm / "final.pt"
    final_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": args.steps,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "losses": losses,
        },
        final_ckpt,
    )
    logger.info(f"Final checkpoint saved: {final_ckpt}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
