"""M9: the STRIPPED intent head — no workspace token, no CFG conditioning, no JEPA.

Companion to :mod:`steer_intent.intent_head` (M8, workspace-conditioned). Same tap (the pi0.5 action
expert's PENULTIMATE hidden, pre-``action_out_proj``), same target family (future EEF), same train-time-
only contract — but every trace of the workspace model is gone.

Why this exists
---------------
M8 wraps the intent loss in machinery: a stage-1 causal encoder trained on MolmoPoint saliency labels
produces ``w``, which CFG-conditions the head (``w_t``) and supplies a JEPA target (``w_{t+1}``). The
claim under test here is that NONE of that is load-bearing in a memory-free setting like LIBERO: the
only thing doing work is the auxiliary SUPERVISION — forcing the shared penultimate to encode where the
end-effector is going — and backprop does the steering without any external latent.

So M9 is M8 minus w. Note this is a REMOVAL, not a replacement: nothing fills w's seat. The intent loss
that remains was already there in M8. Any SR difference between the two arms is therefore attributable
to w and nothing else.

Practical consequence: no stage-1, no saliency labelling, no per-env w precompute. That is what lets the
arm move to robocasa / libero-long unchanged, which the M8 pipeline could not do without retraining an
encoder per environment.

Decoder choice
--------------
``decoder="flow"``  — flow matching over the intent, mirroring the action head's own objective. Models
the full conditional distribution, so a multimodal future (the arm could go to the drawer OR the bowl)
is represented as such instead of being averaged. Motivated once the target is the h-step TRAJECTORY.
``decoder="l1"``    — direct regression to the conditional median; snaps to a mode, robust to jerky
demo frames. Cheap and a good first probe.
``decoder="mse"``   — conditional mean. Matches M8's objective exactly; use it to isolate "did removing
w hurt?" from "did changing the decoder help?".

Nothing here enters the inference graph: at eval the action path is ``action_out_proj(penult)``,
byte-identical to the pi0.5 baseline. The head is dropped.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class IntentFlowHead(nn.Module):
    """penult [B, H, Dp] -> intent aux loss. No w, no CFG, no JEPA (M9).

    A 3-layer MLP, matching the M8 head's shape so the two arms differ only in the workspace signal.
    The penultimate is pooled over the action-token axis and projected to ``hidden``, then injected
    ADDITIVELY into layer 1 — the same seam where M8 adds its ``w_t`` condition.

    For ``decoder="flow"`` the MLP is the velocity field of a flow-matching model over the intent:
    layer 1 consumes the noised intent ``x_tau`` plus (penultimate cond + flow-time embedding), and
    layer 3 emits a velocity. For ``decoder="l1"|"mse"`` layer 1 consumes the pooled penultimate alone
    and layer 3 emits the intent directly.

    Args:
        penult_dim: action-expert width (``action_out_proj.in_features``).
        intent_dim: target width. h*eef_dim for the concat/trajectory target, eef_dim for the mean.
        hidden: MLP width.
        decoder: "flow" | "l1" | "mse".
        weight: scalar on the returned aux loss (the trainer adds it to the action BC loss).
        pool: "mean" pools the action-token axis (M8's default).
    """

    def __init__(self, penult_dim: int, intent_dim: int, *, hidden: int = 512,
                 decoder: str = "flow", weight: float = 1.0, pool: str = "mean") -> None:
        super().__init__()
        if decoder not in ("flow", "l1", "mse"):
            raise ValueError(f"decoder must be flow|l1|mse, got {decoder!r}")
        self.decoder, self.weight, self.pool = decoder, weight, pool
        self.intent_dim = intent_dim
        # Penultimate -> conditioning vector, added into layer 1 (M8's seam, minus w).
        self.cond_proj = nn.Linear(penult_dim, hidden)
        if decoder == "flow":
            self.fc1 = nn.Linear(intent_dim, hidden)   # consumes the NOISED intent
            self.tau_emb = nn.Linear(1, hidden)        # flow time -> additive embedding
        else:
            self.fc1 = nn.Linear(hidden, hidden)       # consumes the pooled/projected penultimate
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, intent_dim)

    def _velocity(self, x_tau: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """(noised intent, flow time, penult cond) -> velocity [B, intent_dim]."""
        h = F.gelu(self.fc1(x_tau) + cond + self.tau_emb(tau[:, None]))
        h = F.gelu(self.fc2(h))
        return self.fc3(h)

    def forward(self, penult: torch.Tensor, intent_target: torch.Tensor | None = None,
                *, global_step: int = 0) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Returns (pred_or_velocity, weighted_total_loss, metrics).

        With ``decoder="flow"`` the first element is the predicted VELOCITY, not the intent — the head
        never produces an intent at train time. Recovering one would mean integrating the ODE, which we
        only ever do as a diagnostic (and never at deploy, where the head is dropped entirely).
        """
        pooled = penult.mean(dim=1) if self.pool == "mean" else penult.reshape(penult.shape[0], -1)
        cond = self.cond_proj(pooled.to(self.cond_proj.weight.dtype)).float()

        losses: dict = {}
        if intent_target is None:                    # no target -> no aux signal (nothing to train on)
            return cond.new_zeros(cond.shape[0], self.intent_dim), cond.new_zeros(()), losses

        x1 = intent_target.float()                   # [B, intent_dim]
        if self.decoder == "flow":
            # Rectified-flow / conditional-FM: straight path from noise z to data x1.
            z = torch.randn_like(x1)
            tau = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
            x_tau = tau[:, None] * x1 + (1.0 - tau[:, None]) * z
            v_target = x1 - z                        # constant along the straight path
            out = self._velocity(x_tau, tau, cond)
            loss = F.mse_loss(out, v_target)
            losses["intent_fm"] = loss.detach()
        else:
            h = F.gelu(self.fc1(cond))
            h = F.gelu(self.fc2(h))
            out = self.fc3(h)
            loss = F.l1_loss(out, x1) if self.decoder == "l1" else F.mse_loss(out, x1)
            losses["intent"] = loss.detach()
        return out, self.weight * loss, losses
