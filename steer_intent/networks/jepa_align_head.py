"""Model-free WSM aux head: JEPA-align the action-head PENULTIMATE features to the next workspace
latent `w_{t+1}`, + SIGReg isotropy on those features. The frozen encoder's `w` is a *training
target only* (precomputed) — never injected, never in the inference graph — so the eval-OOD failure
mode of the injection recipe is structurally impossible. See internal_planning_and_todos/12 +
[[wsm-jepa-penultimate-model-free-design]].

Pure-torch + testable: no gr00t/jax imports. The GR00T action-head wiring lives in
vla_training/train/train_base/_groot_wsm_jepa_common.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from steer_intent.networks.sigreg_loss import sigreg_epps_pulley


class JEPAPredictor(nn.Module):
    """penult (pooled) -> predicted next workspace latent. BYOL/JEPA-style predictor decouples
    "be predictive of w_{t+1}" from "be decodable to actions". `direct=True` = no predictor
    (penult IS the prediction; a bare Linear only if dims differ) — the ablation variant."""

    def __init__(self, in_dim: int, w_dim: int, hidden: int | None = None, direct: bool = False) -> None:
        super().__init__()
        if direct:
            self.net = nn.Identity() if in_dim == w_dim else nn.Linear(in_dim, w_dim)
        else:
            h = hidden or w_dim
            self.net = nn.Sequential(nn.Linear(in_dim, h), nn.GELU(), nn.Linear(h, w_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def wsm_jepa_sigreg_loss(
    penult_act: torch.Tensor,          # [B, H, Dp] per-action-token penultimate features (with grad)
    w_next: torch.Tensor,              # [B, Dw] precomputed next-grid workspace latent (constant target)
    predictor: JEPAPredictor,
    *,
    jepa_weight: float = 1.0,
    sigreg_weight: float = 0.05,
    global_step: int = 0,
    sigreg_per_token: bool = True,
) -> tuple[torch.Tensor, dict]:
    """L = jepa_weight·(1 - cos(predictor(mean_H penult), w_next)) + sigreg_weight·SIGReg(penult).

    JEPA target is a precomputed constant => natural stop-grad. SIGReg runs on the RAW (pre-norm)
    penult so isotropy is enforced, not divided out. Returns (total, metrics)."""
    pooled = penult_act.mean(dim=1)                         # [B, Dp] (keeps penult dtype)
    pp = next(predictor.parameters(), None)                 # predictor may be bf16 under the trainer
    pdtype = pp.dtype if pp is not None else pooled.dtype   # (direct=Identity has no params)
    pred = predictor(pooled.to(pdtype)).float()             # [B, Dw]
    tgt = w_next.float()
    cos = F.cosine_similarity(pred, tgt, dim=-1).mean()
    jepa = 1.0 - cos

    sig_in = penult_act.reshape(-1, penult_act.shape[-1]) if sigreg_per_token else pooled
    sig = sigreg_epps_pulley(sig_in.float(), global_step)

    total = jepa_weight * jepa + sigreg_weight * sig
    return total, {
        "jepa": float(jepa.detach()), "cos": float(cos.detach()),
        "sigreg": float(sig.detach()),
        "jepa_w": jepa_weight, "sigreg_w": sigreg_weight,
    }
