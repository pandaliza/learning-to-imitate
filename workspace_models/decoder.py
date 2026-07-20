"""DETR-style set-reconstruction decoder (Workspace Models, Sec. 3.1).

m learnable query slots cross-attend to the workspace token(s) w, then a per-slot trunk MLP
produces (1) a patch-feature reconstruction p_hat_j and (2) an occupancy logit beta_j (whether
slot j is active). Occupancy accounts for the salient set having fewer than m elements.
"""

import torch
import torch.nn as nn

from .config import WorkspaceConfig


class WorkspaceDecoder(nn.Module):
    def __init__(self, cfg: WorkspaceConfig):
        super().__init__()
        h = cfg.hidden_dim
        self.queries = nn.Parameter(torch.randn(1, cfg.max_patches, h) * 0.02)  # m slots
        layer = nn.TransformerDecoderLayer(
            d_model=h, nhead=cfg.dec_heads, dim_feedforward=int(h * cfg.dec_mlp_ratio),
            dropout=cfg.dec_dropout, batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, cfg.dec_layers)
        # per-slot trunk MLP -> feature head (reconstruct DINO patch) + existence head (occupancy)
        trunk = []
        for _ in range(cfg.trunk_layers):
            trunk += [nn.Linear(h, cfg.trunk_hidden_dim), nn.GELU()]
            h = cfg.trunk_hidden_dim
        self.trunk = nn.Sequential(*trunk)
        self.feature_head = nn.Linear(cfg.trunk_hidden_dim, cfg.feat_dim)
        self.existence_head = nn.Linear(cfg.trunk_hidden_dim, 1)

    def forward(self, workspace: torch.Tensor):
        """workspace: (B, W, h) -> recon_feats (B, m, feat_dim), existence_logits (B, m)."""
        B = workspace.shape[0]
        q = self.queries.expand(B, -1, -1)                   # (B, m, h)
        slots = self.decoder(q, workspace)                   # cross-attend queries -> w
        z = self.trunk(slots)                                # (B, m, trunk_hidden)
        return self.feature_head(z), self.existence_head(z).squeeze(-1)
