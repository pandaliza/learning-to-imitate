"""DiT (Diffusion Transformer) for B0/B1 flow matching.

Generative policy backbone: 6-8 transformer blocks with per-token adaLN-zero conditioning.
Supports B0 (action-only) and B1 (intent + action with T-mask).

Architecture:
  - Context tokens: cached VL features (base + wrist) + encoded proprioception
  - Suffix: B0 = [A_1..A_H]; B1 = [I_1..I_8 | A_1..A_H]
  - Per-token conditioning: tau_I/tau_A via sinusoidal embeddings + MLP
  - T-mask for B1: intent tokens attend to {context, intent}; action tokens attend to {context, intent, action}
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position / time embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed scalar or batch of scalars as sinusoidal features.

        Args:
            x: (B,) or scalar tensor, values in [0, 1]

        Returns:
            (B, dim) sinusoidal embedding
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with learnable scale/shift, gated residual.

    Per-token variant: accepts (B, S, D) conditioning and applies per-token.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 3),
        )
        # Initialize mlp to output near-zero
        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply adaLN-zero conditioning.

        Args:
            x: (B, S, D) hidden states
            cond: (B, D) or (B, S, D) conditioning
                  If 2-D, broadcast over sequence; if 3-D, per-token

        Returns:
            (B, S, D) modulated and gated output
        """
        # Normalize
        normed = self.norm(x)

        # Get conditioning
        if cond.ndim == 2:
            # (B, D) -> expand to (B, 1, 3D) then broadcast
            out = self.mlp(cond)  # (B, 3D)
            out = out.unsqueeze(1)  # (B, 1, 3D)
            # Broadcast to (B, S, 3D)
            out = out.expand(-1, x.shape[1], -1)
        else:
            # (B, S, D) -> (B, S, 3D)
            out = self.mlp(cond)

        scale, shift, gate = out.chunk(3, dim=-1)

        # Apply
        return normed * (1 + scale) + shift + gate * x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP, adaLN-zero conditioning."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, attn_drop: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.norm1 = AdaLNZero(dim)
        self.norm2 = AdaLNZero(dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with per-token adaLN-zero conditioning.

        Args:
            x: (B, S, D) hidden states
            cond: (B, D) or (B, S, D) conditioning (tau embeddings)
            attn_mask: (S, S) or None, True = masked

        Returns:
            (B, S, D) output
        """
        # Self-attention with adaLN-zero
        normed = self.norm1(x, cond)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out

        # MLP with adaLN-zero
        normed = self.norm2(x, cond)
        mlp_out = self.mlp(normed)
        x = x + mlp_out

        return x


class ProprioEncoder(nn.Module):
    """Small MLP encoder for proprioception (16D state)."""

    def __init__(self, state_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state.

        Args:
            state: (B, obs_steps, state_dim)

        Returns:
            (B, output_dim) encoded state
        """
        # Pool over obs_steps
        state_pooled = state.mean(dim=1)  # (B, state_dim)
        return self.mlp(state_pooled)


class DiT(nn.Module):
    """Diffusion Transformer for B0/B1 generative policies.

    Attributes:
        arm: "b0" (action-only) or "b1" (intent + action with T-mask)
        depth: number of transformer blocks
        dim: embedding dimension
        num_heads: number of attention heads
        action_dim: action dimension (12 for RoboCasa)
        action_horizon: number of action tokens (10)
        intent_dim: intent dimension (7 for RoboCasa)
        intent_horizon: number of intent tokens (8 for h=8, Δ=2)
        vision_dim: dimension of cached vision features (e.g., 768 for PaliGemma)
        num_vision_tokens: pooled (1) or grid16 (16)
        state_dim: proprioception dimension (16)
    """

    def __init__(
        self,
        arm: Literal["b0", "b1"] = "b0",
        depth: int = 6,
        dim: int = 512,
        num_heads: int = 8,
        action_dim: int = 12,
        action_horizon: int = 10,
        intent_dim: int = 7,
        intent_horizon: int = 8,
        vision_dim: int = 768,
        num_vision_tokens: int = 1,
        state_dim: int = 16,
        lang_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.arm = arm
        self.depth = depth
        self.dim = dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.intent_dim = intent_dim
        self.intent_horizon = intent_horizon
        self.vision_dim = vision_dim
        self.num_vision_tokens = num_vision_tokens
        self.state_dim = state_dim
        self.lang_dim = lang_dim

        # Input projections
        self.vision_proj = nn.Linear(vision_dim, dim)
        self.lang_proj = nn.Linear(lang_dim, dim)
        self.state_encoder = ProprioEncoder(state_dim, 256, dim)

        # Suffix projections
        if arm == "b1":
            self.intent_in_proj = nn.Linear(intent_dim, dim)
            self.intent_out_proj = nn.Linear(dim, intent_dim)
            # Zero-init out projection so intent branch starts silent
            nn.init.constant_(self.intent_out_proj.weight, 0)
            nn.init.constant_(self.intent_out_proj.bias, 0)

        self.action_in_proj = nn.Linear(action_dim, dim)
        self.action_out_proj = nn.Linear(dim, action_dim)

        # Time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio=4.0, attn_drop=dropout)
            for _ in range(depth)
        ])

    def _make_causal_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create block-causal attention mask for T-mask (B1 only).

        Pattern: [1] (context) + [0]*7 (intent tokens) + [1] + [0]*9 (action tokens)
        Result:
          - Context row: attends to all
          - Intent rows: attend to {context, intent}
          - Action rows: attend to {context, intent, action}

        Returns:
            (seq_len, seq_len) boolean mask (True = masked out)
        """
        if self.arm != "b1":
            return None

        # This assumes: context tokens + intent tokens + action tokens
        # seq_len = num_context + intent_horizon + action_horizon
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Find split points
        # Context tokens get no masking (attend to all, but we only care about attention within suffix)
        # Intent tokens: attend to context + intent only
        # Action tokens: attend to context + intent + action

        # For simplicity in masking the suffix:
        # Assume context comes first, then intent, then action
        # We need to know the number of context tokens, which depends on num_vision_tokens + 1 (lang)
        # This is tricky; we'll handle it per-layer

        # Actually, the cleaner approach: the mask is created at forward time when we know seq structure
        return mask

    def _get_attention_mask_for_b1(
        self,
        batch_size: int,
        num_context: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create T-mask for B1: intent tokens attend to {context, intent}; action to {context, intent, action}.

        PyTorch convention: True = mask out (don't attend), False = attend

        Args:
            batch_size: batch size
            num_context: number of context tokens (vision + lang + state)
            device: torch device

        Returns:
            (S, S) attention mask where True = mask out, False = attend
        """
        if self.arm != "b0":
            # T-mask: context [0:num_context] + intent [num_context:num_context+intent_horizon] + action [num_context+intent_horizon:]
            num_intent = self.intent_horizon
            num_action = self.action_horizon
            total = num_context + num_intent + num_action

            # Build mask: (S, S) where False = attend, True = mask out
            mask = torch.zeros(total, total, dtype=torch.bool, device=device)

            # Intent tokens: MASK OUT action tokens (set to True)
            intent_start = num_context
            intent_end = intent_start + num_intent
            action_start = intent_end
            action_end = total

            # Intent rows: mask action columns (set to True to mask out)
            mask[intent_start:intent_end, action_start:action_end] = True

            return mask

        return None

    def forward(
        self,
        x_action: torch.Tensor,
        x_intent: torch.Tensor | None,
        tau_action: torch.Tensor,
        tau_intent: torch.Tensor | None,
        vision_features: torch.Tensor,
        lang_features: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through the DiT.

        Args:
            x_action: (B, H, action_dim) noised action
            x_intent: (B, h, intent_dim) or None, noised intent
            tau_action: (B,) noise level for actions
            tau_intent: (B,) or None, noise level for intent
            vision_features: (B, num_cams*num_vision_tokens, vision_dim) cached VL features
            lang_features: (B, lang_dim) instruction embedding
            state: (B, obs_steps, state_dim) proprioception

        Returns:
            (v_action, v_intent): predicted velocities
              v_action: (B, H, action_dim)
              v_intent: (B, h, intent_dim) or None
        """
        B = x_action.shape[0]
        device = x_action.device

        # Project inputs
        vision_proj = self.vision_proj(vision_features)  # (B, num_vision, dim)
        lang_proj = self.lang_proj(lang_features).unsqueeze(1)  # (B, 1, dim)
        state_proj = self.state_encoder(state).unsqueeze(1)  # (B, 1, dim)

        # Context tokens: vision + lang + state
        context = torch.cat([vision_proj, lang_proj, state_proj], dim=1)  # (B, num_context, dim)
        num_context = context.shape[1]

        # Condition embeddings (per-token tau)
        tau_action_emb = self.time_mlp(tau_action)  # (B, dim)
        if self.arm == "b1":
            tau_intent_emb = self.time_mlp(tau_intent)  # (B, dim)

        # Build suffix
        if self.arm == "b0":
            # B0: action tokens only
            action_proj = self.action_in_proj(x_action)  # (B, H, dim)
            suffix = action_proj

            # Create per-token conditioning for suffix
            cond_suffix = tau_action_emb.unsqueeze(1).expand(-1, self.action_horizon, -1)  # (B, H, dim)
        else:
            # B1: intent + action tokens
            intent_proj = self.intent_in_proj(x_intent)  # (B, h, dim)
            action_proj = self.action_in_proj(x_action)  # (B, H, dim)
            suffix = torch.cat([intent_proj, action_proj], dim=1)  # (B, h+H, dim)

            # Per-token conditioning for suffix
            cond_intent = tau_intent_emb.unsqueeze(1).expand(-1, self.intent_horizon, -1)  # (B, h, dim)
            cond_action = tau_action_emb.unsqueeze(1).expand(-1, self.action_horizon, -1)  # (B, H, dim)
            cond_suffix = torch.cat([cond_intent, cond_action], dim=1)  # (B, h+H, dim)

        # Combine context + suffix
        x = torch.cat([context, suffix], dim=1)  # (B, num_context + suffix_len, dim)

        # Create full conditioning: context gets tau_action (or average), suffix gets per-token
        # For context tokens, use tau_action_emb broadcast over all context tokens
        cond_context = tau_action_emb.unsqueeze(1).expand(-1, num_context, -1)  # (B, num_context, dim)
        cond = torch.cat([cond_context, cond_suffix], dim=1)  # (B, num_context + suffix_len, dim)

        # Get attention mask for B1
        attn_mask = self._get_attention_mask_for_b1(B, num_context, device)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, cond=cond, attn_mask=attn_mask)

        # Extract suffix outputs
        suffix_out = x[:, num_context:, :]  # (B, suffix_len, dim)

        if self.arm == "b0":
            v_action = self.action_out_proj(suffix_out)  # (B, H, action_dim)
            v_intent = None
        else:
            # Split intent and action
            v_intent = self.intent_out_proj(suffix_out[:, :self.intent_horizon, :])  # (B, h, intent_dim)
            v_action = self.action_out_proj(suffix_out[:, self.intent_horizon:, :])  # (B, H, action_dim)

        return v_action, v_intent

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_dit(
    arm: Literal["b0", "b1"],
    depth: int = 6,
    dim: int = 512,
    num_heads: int = 8,
    vision_dim: int = 768,
    num_vision_tokens: int = 1,
    lang_dim: int = 768,
    **kwargs,
) -> DiT:
    """Factory function to create a DiT model."""
    return DiT(
        arm=arm,
        depth=depth,
        dim=dim,
        num_heads=num_heads,
        vision_dim=vision_dim,
        num_vision_tokens=num_vision_tokens,
        lang_dim=lang_dim,
        **kwargs,
    )
