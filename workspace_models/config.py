"""Workspace-model config (current-frame variant).

Defaults follow Workspace Models (CoRL 2026), Table 3. We implement the *current-frame*
setting only: no causal temporal encoder over o_{1:t}; the encoder sees a single frame's
DINOv3 patch tokens (+ optional proprio) plus the learned workspace-slot input. Everything
else -- the attention pooler, the DETR-style set-reconstruction decoder, Hungarian matching,
and the feature/existence losses -- matches the paper.
"""

from dataclasses import dataclass


@dataclass
class WorkspaceConfig:
    # --- DINOv3 backbone (frozen; produces patch tokens) — Table 3 "Model" ---
    feat_dim: int = 768            # DINOv3 patch dim (reconstruction target dim)
    num_patches: int = 196         # tokens per frame (196 CubeDrop / 49 others)
    proprio_dim: int | None = None # add a proprio token if set (paper feeds proprio); None => skip

    # --- Attention pooler (Fig 2 "Pooling Layer") — Table 3 "Pooler" ---
    pool_tokens: int = 1
    pool_heads: int = 4
    pool_dropout: float = 0.1

    # --- Workspace encoder — Table 3 "Workspace Encoder" ---
    hidden_dim: int = 768
    enc_layers: int = 2
    enc_heads: int = 4
    enc_mlp_ratio: float = 2.0
    enc_dropout: float = 0.1
    num_workspace_tokens: int = 1  # "Workspace tokens per step 1" (the bottleneck)

    # --- Workspace decoder / DETR set reconstruction — Table 3 "Workspace Decoder" ---
    max_patches: int = 8           # m = decoder query slots = max salient patches ("Slots 8")
    dec_layers: int = 2
    dec_heads: int = 4
    dec_mlp_ratio: float = 2.0
    dec_dropout: float = 0.1
    trunk_hidden_dim: int = 768
    trunk_layers: int = 2

    # --- Loss weights — Table 3 "Loss" (Eq. 3.2) ---
    feature_loss_weight: float = 1.0     # lambda_1 on L_feat (2.0 for DrawerRecall/BalanceBar)
    existence_loss_weight: float = 0.01  # lambda_2 on L_active

    # --- Hungarian matching costs — Table 3 "Loss" (Eq. under 3.1) ---
    match_feature_cost: float = 1.0
    match_existence_cost: float = 1.0    # lambda_0 in the matching cost
