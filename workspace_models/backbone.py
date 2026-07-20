"""Frozen DINOv3 backbone -> patch tokens (Table 3 "Model").

Kept separate from the model so the core (encoder/decoder/losses) stays backbone-agnostic and
testable on precomputed grids. The workspace model consumes patch tokens (B, N, feat_dim); use
this to produce them live at train/deploy, OR precompute grids offline (as in the DINOSAUR
pipeline) and skip the backbone entirely.
"""

import torch
import torch.nn as nn


class DinoV3Backbone(nn.Module):
    """Wraps a frozen DINOv3 ViT and returns patch tokens (B, N, feat_dim).

    Requires DINOv3 available via torch.hub (facebookresearch/dinov3). If unavailable, precompute
    patch grids offline instead and feed them straight into WorkspaceModel.
    """

    def __init__(self, model_name: str = "dinov3_vitb16", weights: str | None = None,
                 device: str = "cuda"):
        super().__init__()
        try:
            self.model = torch.hub.load("facebookresearch/dinov3", model_name, weights=weights)
        except Exception as e:  # noqa: BLE001 — surface a clear, actionable message
            raise RuntimeError(
                f"Could not load DINOv3 '{model_name}' via torch.hub ({e}). Install DINOv3 or "
                "precompute patch grids offline and pass them to WorkspaceModel directly."
            ) from e
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) normalized -> patch tokens (B, N, feat_dim)."""
        feats = self.model.forward_features(images)
        return feats["x_norm_patchtokens"] if isinstance(feats, dict) else feats
