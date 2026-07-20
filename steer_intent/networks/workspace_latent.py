"""Workspace encoder: per-step (pooled VLM + proprio) tokens -> causal w_t stream.

Implements the encoder half of the locked WSM design (2026-06-18 figures):
  * PatchPool  — a TRAINED attention pool ("trained mean-pool" in the figure): one learned query
    cross-attends the step's frozen-VLM patch tokens (192 = 3 views x 64) -> a single token.
  * WorkspaceEncoder — fuse(pooled_vlm, proprio) = w'_t (one token per timestep), then a CAUSAL
    AdaLN-Zero transformer over the T-step sequence (language as the AdaLN condition) -> w_t.
Each w_t is later decoded by the salient-patch decoder (keyframe_patch_head.py). Inputs come from
the frozen GR00T backbone via the feature cache (workspace_models/features/backbone_tap.py).
Governed by internal_planning_and_todos/04_wsm_roadmap.md + [[wsm-reference-impl-exists]].
"""
from __future__ import annotations

import torch
from torch import nn

from steer_intent.networks.adaln_zero import AdaLNZeroBlock


class PatchPool(nn.Module):
    """Trained attention pool: 1 learned query cross-attends the per-step patch tokens -> 1 token."""

    def __init__(self, backbone_dim: int, dim: int, n_heads: int) -> None:
        super().__init__()
        self.proj = nn.Linear(backbone_dim, dim)
        self.query = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches [B,T,P,backbone_dim] -> pooled [B,T,dim]
        b, t, p, _ = patches.shape
        x = self.proj(patches).reshape(b * t, p, -1)
        q = self.query.expand(b * t, -1, -1)
        pooled, _ = self.attn(q, x, x, need_weights=False)
        return pooled.reshape(b, t, -1)


class WorkspaceEncoder(nn.Module):
    """(pooled VLM, proprio, language) per step -> causal AdaLN-Zero stream -> w_t [B,T,dim]."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        # Optional input normalization (cfg.input_norm). RoboCasa GR00T patch tokens have RMS ~4.3 (vs pi
        # ~0.97) and are fed RAW into a bare Linear -> large activations -> forward overflow -> NaN late in
        # training. LayerNorm-ing the raw inputs tames magnitude at init (learnable affine can re-scale).
        self.input_norm = bool(getattr(cfg, "input_norm", False))
        if self.input_norm:
            self.patch_in_norm = nn.LayerNorm(cfg.backbone_dim)
            self.proprio_in_norm = nn.LayerNorm(cfg.proprio_dim)
            self.lang_in_norm = nn.LayerNorm(cfg.lang_dim)
        self.pool = PatchPool(cfg.backbone_dim, cfg.dim, cfg.n_heads)
        self.proprio_proj = nn.Linear(cfg.proprio_dim, cfg.dim)
        self.lang_proj = nn.Linear(cfg.lang_dim, cfg.dim)
        self.time_emb = nn.Parameter(torch.zeros(cfg.max_t, cfg.dim))
        nn.init.normal_(self.time_emb, std=0.02)
        self.blocks = nn.ModuleList(AdaLNZeroBlock(cfg.dim, cfg.n_heads, cfg.mlp_ratio)
                                    for _ in range(cfg.n_layers))
        self.out_norm = nn.LayerNorm(cfg.dim)

    def _causal_window_mask(self, t: int, device: torch.device) -> torch.Tensor:
        """Block-causal + C-horizon window over the T-step stream. True = MASKED (torch MHA)."""
        step = torch.arange(t, device=device)
        src, tgt = step.unsqueeze(0), step.unsqueeze(1)  # [tgt, src]
        return (src > tgt) | (src < tgt - (self.cfg.c_horizon - 1))

    def forward(self, patches: torch.Tensor, proprio: torch.Tensor, cond_lang: torch.Tensor) -> torch.Tensor:
        """patches [B,T,P,backbone_dim], proprio [B,T,proprio_dim], cond_lang [B,T,lang_dim]
        (subgoal emb at train w/ dropout -> global; global at inference). Returns w [B,T,dim]."""
        if self.input_norm:
            patches, proprio, cond_lang = (self.patch_in_norm(patches),
                                           self.proprio_in_norm(proprio), self.lang_in_norm(cond_lang))
        w_prime = self.pool(patches) + self.proprio_proj(proprio)         # fuse (figure's ⊕) -> w'_t
        b, t, _ = w_prime.shape
        x = w_prime + self.time_emb[:t].unsqueeze(0)
        cond = self.lang_proj(cond_lang)                                  # per-step AdaLN condition
        mask = self._causal_window_mask(t, x.device)
        for blk in self.blocks:
            x = blk(x, cond, mask)
        return self.out_norm(x)
