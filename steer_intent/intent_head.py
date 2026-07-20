"""Workspace-conditioned intent head (train-time only).

Taps the pi0.5 action expert's PENULTIMATE hidden (pre-``action_out_proj``) and runs a 3-layer MLP
that predicts INTENT (future EEF pose). Two auxiliary training signals shape the shared penultimate:

  * w_t  CFG-conditions MLP layer 1 (additive on the pre-activation), via the ported WSMCfgConditioner
    (zero-init proj + learned null + per-sample dropout) -> at init the condition is exactly 0, so the
    head is byte-identical to a no-w baseline and can only learn its way into using the signal.
  * w_{t+1} is a JEPA alignment TARGET for MLP layer 2 (cosine, stop-grad on the precomputed frozen w).
    w_t (input) and w_{t+1} (target) are DIFFERENT tensors, so the alignment has no copy shortcut.

w never enters the inference graph: at eval the action path is action_out_proj(penult) unchanged. The
losses only reshape the penultimate's weights at train time. `s`-guidance is intentionally not wired
(intent is a dead-end output) — see the discussion in the training script.

Reuses export modules verbatim: WSMCfgConditioner, JEPAPredictor, sigreg_epps_pulley.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from steer_intent.networks.jepa_align_head import JEPAPredictor
from steer_intent.networks.sigreg_loss import sigreg_epps_pulley
from steer_intent.networks.wsm_cfg_cond import WSMCfgConditioner


class WSMIntentHead(nn.Module):
    """penult [B,H,Dp] (+ w_t, w_{t+1}) -> intent_pred [B,intent_dim] + aux losses.

    pool='mean' pools the action-token axis (their JEPA default). Layer 1 is CFG-conditioned on w_t;
    layer 2's activation is JEPA-aligned to w_{t+1}; layer 3 regresses intent. jepa_weight/intent_weight/
    sigreg_weight scale the three terms (sigreg default 0 -> off; a frozen target can't collapse)."""

    def __init__(self, penult_dim: int, intent_dim: int, *, w_dim: int = 512, hidden: int = 512,
                 p_drop: float = 0.2, jepa_weight: float = 1.0, intent_weight: float = 1.0,
                 sigreg_weight: float = 0.0, pool: str = "mean") -> None:
        super().__init__()
        self.pool = pool
        self.jepa_weight, self.intent_weight, self.sigreg_weight = jepa_weight, intent_weight, sigreg_weight
        # w_t CFG conditioner: cond added to fc1's pre-activation. Single-slot use (w_next left None ->
        # its learned null); proj is zero-init so cond == 0 at init (baseline-identical).
        self.cond = WSMCfgConditioner(w_dim=w_dim, cond_dim=hidden, p_drop=p_drop)
        self.fc1 = nn.Linear(penult_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, intent_dim)
        # JEPA predictor: MLP layer-2 activation -> predicted w_{t+1} (cosine target).
        self.jepa_predictor = JEPAPredictor(in_dim=hidden, w_dim=w_dim)

    def forward(self, penult: torch.Tensor, w_t: torch.Tensor | None,
                w_next: torch.Tensor | None, intent_target: torch.Tensor | None = None,
                *, global_step: int = 0) -> tuple[torch.Tensor, torch.Tensor, dict]:
        pooled = penult.mean(dim=1) if self.pool == "mean" else penult.reshape(penult.shape[0], -1)
        cond = self.cond(w_t, None, training=self.training)      # [B, hidden], 0 at init
        h1 = F.gelu(self.fc1(pooled) + cond)
        h2 = F.gelu(self.fc2(h1))                                # JEPA alignment point (MLP layer 2)
        intent_pred = self.fc3(h2)

        losses: dict = {}
        total = intent_pred.new_zeros(())
        if intent_target is not None:
            li = F.mse_loss(intent_pred, intent_target)
            losses["intent"] = li.detach()
            total = total + self.intent_weight * li
        if w_next is not None:
            pred = self.jepa_predictor(h2.to(next(self.jepa_predictor.parameters()).dtype)).float()
            cos = F.cosine_similarity(pred, w_next.float(), dim=-1).mean()   # w_next precomputed => stop-grad
            lj = 1.0 - cos
            losses["jepa"] = lj.detach()
            losses["cos"] = cos.detach()
            total = total + self.jepa_weight * lj
        if self.sigreg_weight > 0:
            ls = sigreg_epps_pulley(h2.float(), global_step)
            losses["sigreg"] = ls.detach()
            total = total + self.sigreg_weight * ls
        return intent_pred, total, losses
