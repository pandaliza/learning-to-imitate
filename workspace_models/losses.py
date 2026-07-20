"""Set-reconstruction losses under a fixed Hungarian matching (Eq. 3.1).

    L_feat   = (1/|S_t|) sum_j y(sigma(j)) || p_hat_j - p_{sigma(j)} ||^2     (matched slots)
    L_active = (1/m)     sum_j CE( beta_j ; y(sigma(j)) )                     (all slots)

Gradients flow through recon_feats / existence_logits (the matching indices are constants).
"""

import torch
import torch.nn.functional as F

from .config import WorkspaceConfig
from .matching import match


def set_losses(recon_feats: torch.Tensor, existence_logits: torch.Tensor,
               target_patches: torch.Tensor, target_mask: torch.Tensor, cfg: WorkspaceConfig):
    B, m, _ = recon_feats.shape
    device = recon_feats.device
    matches = match(recon_feats, existence_logits, target_patches, target_mask, cfg)

    feat_terms, active_targets = [], torch.zeros(B, m, device=device)
    for b, (slot_idx, patch_idx) in enumerate(matches):
        if slot_idx.numel() > 0:
            slot_idx, patch_idx = slot_idx.to(device), patch_idx.to(device)
            n_real = slot_idx.numel()
            se = (recon_feats[b, slot_idx] - target_patches[b, patch_idx]).pow(2).sum(-1)  # (n_real,)
            feat_terms.append(se.sum() / n_real)             # (1/|S_t|) sum ||.||^2
            active_targets[b, slot_idx] = 1.0                # matched slots -> occupied

    feat = torch.stack(feat_terms).mean() if feat_terms else recon_feats.sum() * 0.0
    existence = F.binary_cross_entropy_with_logits(existence_logits, active_targets)  # mean over B*m
    total = cfg.feature_loss_weight * feat + cfg.existence_loss_weight * existence
    return {"feat": feat, "existence": existence, "total": total}
