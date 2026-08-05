"""Sampling / inference schedules for B0/B1.

B0: standard Euler integration from tau=1 -> 0 over K_A steps.
B1: intent-first schedule with two phases:
  1. Intent denoising (K_I steps): tau_I 1 -> 0, action tokens at noise
  2. Action denoising (K_A steps): tau_A 1 -> 0, conditioned on clean intent
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


class SampleContext:
    """Context for sampling: observation, language, model."""

    def __init__(
        self,
        vision_features: torch.Tensor,
        lang_features: torch.Tensor,
        state: torch.Tensor,
        model: torch.nn.Module,
        device: torch.device = "cuda",
    ):
        """Initialize sampling context.

        Args:
            vision_features: (B, num_vision, vision_dim) or (B, num_vision, 16, vision_dim)
            lang_features: (B, lang_dim)
            state: (B, obs_steps, state_dim)
            model: DiT model
            device: torch device
        """
        self.vision_features = vision_features.to(device)
        self.lang_features = lang_features.to(device)
        self.state = state.to(device)
        self.model = model.to(device)
        self.device = device
        self.batch_size = vision_features.shape[0]


def sample_b0(
    ctx: SampleContext,
    action_dim: int = 12,
    action_horizon: int = 10,
    steps: int = 10,
    z_action: torch.Tensor | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample action chunk from B0 model using Euler integration.

    Args:
        ctx: SampleContext with observation features
        action_dim: action dimension (default 12)
        action_horizon: number of action steps (default 10)
        steps: number of Euler steps
        z_action: (B, H, action_dim) initial noise. If None, sample from N(0,I)
        seed: random seed for reproducibility

    Returns:
        action: (B, H, action_dim) sampled action chunk
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Initialize action noise
    if z_action is None:
        z_action = torch.randn(
            ctx.batch_size,
            action_horizon,
            action_dim,
            device=ctx.device,
        )
    else:
        z_action = z_action.to(ctx.device)

    # Euler integration: tau 1 -> 0
    x = z_action.clone()
    dt = 1.0 / steps

    ctx.model.eval()
    with torch.no_grad():
        for step in range(steps):
            tau = torch.tensor(
                1.0 - step * dt,
                dtype=torch.float32,
                device=ctx.device,
            ).expand(ctx.batch_size)

            # Predict velocity
            v_action, _ = ctx.model(
                x_action=x,
                x_intent=None,
                tau_action=tau,
                tau_intent=None,
                vision_features=ctx.vision_features,
                lang_features=ctx.lang_features,
                state=ctx.state,
            )

            # Clip velocity to prevent divergence
            v_action = torch.clamp(v_action, -10.0, 10.0)

            # Euler step: x = x + v * dt
            x = x + v_action * (-dt)

    return x


def sample_b1(
    ctx: SampleContext,
    action_dim: int = 12,
    action_horizon: int = 10,
    intent_dim: int = 7,
    intent_horizon: int = 8,
    steps_intent: int = 4,
    steps_action: int = 10,
    z_intent: torch.Tensor | None = None,
    z_action: torch.Tensor | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample intent + action from B1 model using intent-first schedule.

    Two-phase sampling:
      Phase 1: Denoise intent (tau_I 1 -> 0) while actions stay at noise
      Phase 2: Denoise actions (tau_A 1 -> 0) conditioned on clean intent

    Args:
        ctx: SampleContext with observation features
        action_dim: action dimension
        action_horizon: number of action steps
        intent_dim: intent dimension
        intent_horizon: number of intent waypoints
        steps_intent: Euler steps for intent phase
        steps_action: Euler steps for action phase
        z_intent: (B, h, intent_dim) initial noise. If None, sample from N(0,I)
        z_action: (B, H, action_dim) initial noise. If None, sample from N(0,I)
        seed: random seed for reproducibility

    Returns:
        (intent, action): sampled intent and action chunks
          intent: (B, h, intent_dim)
          action: (B, H, action_dim)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Initialize intent and action noise
    if z_intent is None:
        z_intent = torch.randn(
            ctx.batch_size,
            intent_horizon,
            intent_dim,
            device=ctx.device,
        )
    else:
        z_intent = z_intent.to(ctx.device)

    if z_action is None:
        z_action = torch.randn(
            ctx.batch_size,
            action_horizon,
            action_dim,
            device=ctx.device,
        )
    else:
        z_action = z_action.to(ctx.device)

    # Phase 1: Denoise intent (tau_I 1 -> 0)
    intent = z_intent.clone()
    action = z_action.clone()
    dt_intent = 1.0 / steps_intent

    ctx.model.eval()
    with torch.no_grad():
        for step in range(steps_intent):
            tau_i = torch.tensor(
                1.0 - step * dt_intent,
                dtype=torch.float32,
                device=ctx.device,
            ).expand(ctx.batch_size)

            # tau_A = 1 during intent denoising (actions stay at noise)
            tau_a = torch.ones_like(tau_i)

            # Predict velocities
            v_action, v_intent = ctx.model(
                x_action=action,
                x_intent=intent,
                tau_action=tau_a,
                tau_intent=tau_i,
                vision_features=ctx.vision_features,
                lang_features=ctx.lang_features,
                state=ctx.state,
            )

            # Clip velocities to prevent divergence
            v_intent = torch.clamp(v_intent, -10.0, 10.0)

            # Euler step for intent only
            intent = intent + v_intent * (-dt_intent)

    # Clamp intent at tau_I = 0 (clean intent)
    intent = intent.detach().clone()

    # Phase 2: Denoise actions (tau_A 1 -> 0) conditioned on clean intent
    tau_intent_clean = torch.zeros(ctx.batch_size, device=ctx.device)
    dt_action = 1.0 / steps_action

    with torch.no_grad():
        for step in range(steps_action):
            tau_a = torch.tensor(
                1.0 - step * dt_action,
                dtype=torch.float32,
                device=ctx.device,
            ).expand(ctx.batch_size)

            # Predict velocities (intent at clean conditioning)
            v_action, _ = ctx.model(
                x_action=action,
                x_intent=intent,
                tau_action=tau_a,
                tau_intent=tau_intent_clean,
                vision_features=ctx.vision_features,
                lang_features=ctx.lang_features,
                state=ctx.state,
            )

            # Clip velocity to prevent divergence
            v_action = torch.clamp(v_action, -10.0, 10.0)

            # Euler step for action only
            action = action + v_action * (-dt_action)

    return intent, action
