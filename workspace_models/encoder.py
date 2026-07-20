"""Attention pooler + workspace encoder (current-frame).

Pipeline (Fig 2, current-frame slice):
    DINOv3 patch tokens ──pool──> X ─┐
    proprio (optional) ──proj──────> ┼──[self-attn transformer]──> workspace token(s) w
    learned workspace slots z ───────┘

The paper's encoder is causal over time; for the current-frame setting the sequence is a
single frame, so we use plain (bidirectional) self-attention over [X ; proprio ; z] and read
out the outputs at the z positions as the workspace token(s).
"""

import torch
import torch.nn as nn

from .config import WorkspaceConfig


class AttentionPooler(nn.Module):
    """Pool a patch grid (B, N, feat_dim) -> (B, pool_tokens, hidden) via cross-attention."""

    def __init__(self, cfg: WorkspaceConfig):
        super().__init__()
        h = cfg.hidden_dim
        self.in_proj = nn.Linear(cfg.feat_dim, h)
        self.query = nn.Parameter(torch.randn(1, cfg.pool_tokens, h) * 0.02)
        self.attn = nn.MultiheadAttention(h, cfg.pool_heads, dropout=cfg.pool_dropout, batch_first=True)
        self.norm = nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(patch_tokens)                       # (B, N, h)
        q = self.query.expand(x.shape[0], -1, -1)            # (B, pool_tokens, h)
        pooled, _ = self.attn(q, x, x)                       # (B, pool_tokens, h)
        return self.norm(pooled + self.ff(pooled))


class WorkspaceEncoder(nn.Module):
    def __init__(self, cfg: WorkspaceConfig):
        super().__init__()
        h = cfg.hidden_dim
        self.cfg = cfg
        self.pooler = AttentionPooler(cfg)
        self.proprio_proj = nn.Linear(cfg.proprio_dim, h) if cfg.proprio_dim else None
        # learned workspace-slot inputs z ~ N(0, I) (their outputs become the workspace tokens)
        self.z = nn.Parameter(torch.randn(1, cfg.num_workspace_tokens, h))
        layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=cfg.enc_heads, dim_feedforward=int(h * cfg.enc_mlp_ratio),
            dropout=cfg.enc_dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.enc_layers)

    def forward(self, patch_tokens: torch.Tensor, proprio: torch.Tensor | None = None) -> torch.Tensor:
        B = patch_tokens.shape[0]
        toks = [self.pooler(patch_tokens)]                   # (B, pool_tokens, h)
        if self.proprio_proj is not None and proprio is not None:
            toks.append(self.proprio_proj(proprio).unsqueeze(1))  # (B, 1, h)
        toks.append(self.z.expand(B, -1, -1))                # (B, num_workspace_tokens, h)
        seq = torch.cat(toks, dim=1)
        out = self.encoder(seq)
        return out[:, -self.cfg.num_workspace_tokens:]       # workspace token(s): (B, W, h)
