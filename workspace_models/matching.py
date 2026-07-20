"""Hungarian matching between decoder slots and the salient patch set (Sec. 3.1).

For each sample we solve an injective assignment sigma: [m] -> {0 (null), 1..|S_t|} that
minimizes the cumulative cost c(i, j) = y(i)||p_i - p_hat_j||^2 + lambda_0 * CE(beta_j; y(i)).
We do NOT backprop through the matching (footnote 2): costs are computed on detached tensors.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .config import WorkspaceConfig


@torch.no_grad()
def match(recon_feats: torch.Tensor, existence_logits: torch.Tensor,
          target_patches: torch.Tensor, target_mask: torch.Tensor, cfg: WorkspaceConfig):
    """Returns, per batch item, (slot_idx, patch_idx) LongTensors of matched active slots.

    recon_feats:   (B, m, D)     — slot patch reconstructions
    existence_logits: (B, m)     — slot occupancy logits
    target_patches: (B, P, D)    — padded salient patches
    target_mask:   (B, P) bool   — True where a target patch is real (not padding)
    """
    B, m, _ = recon_feats.shape
    out = []
    # CE(beta_j; y=1) = -log sigmoid(logit) = softplus(-logit); same across real patches (y(i)=1),
    # so it enters the cost as a per-slot column-constant that biases toward high-occupancy slots.
    exist_cost_active = F.softplus(-existence_logits)        # (B, m)
    for b in range(B):
        real = target_mask[b].nonzero(as_tuple=False).flatten()
        if real.numel() == 0:
            out.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
            continue
        tgt = target_patches[b, real]                        # (n_real, D)
        # feature cost ||p_i - p_hat_j||^2 -> (m, n_real)
        feat_cost = torch.cdist(recon_feats[b], tgt, p=2) ** 2
        cost = cfg.match_feature_cost * feat_cost + cfg.match_existence_cost * exist_cost_active[b, :, None]
        slot_idx, col_idx = linear_sum_assignment(cost.cpu().numpy())
        out.append((torch.as_tensor(slot_idx, dtype=torch.long),
                    real[torch.as_tensor(col_idx, dtype=torch.long)]))
    return out
