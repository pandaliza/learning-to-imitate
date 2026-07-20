"""WorkspaceModel: current-frame workspace-token encoder + DETR set-reconstruction head.

Training:  w, losses = model(patch_tokens, target_patches, target_mask [, proprio])
Deploy:    w = model.encode(patch_tokens [, proprio])          # decoder is dropped

`w` (B, num_workspace_tokens, hidden_dim) is the lightweight intent/memory token: a
saliency-supervised summary distilled from VLM-curated salient DINOv3 patches. It plugs into
the Pi0.5 intent-conditioning path the same way the slot code z* does (project -> condition).
"""

import torch
import torch.nn as nn

from .config import WorkspaceConfig
from .encoder import WorkspaceEncoder
from .decoder import WorkspaceDecoder
from .losses import set_losses


class WorkspaceModel(nn.Module):
    def __init__(self, cfg: WorkspaceConfig | None = None):
        super().__init__()
        self.cfg = cfg or WorkspaceConfig()
        self.encoder = WorkspaceEncoder(self.cfg)
        self.decoder = WorkspaceDecoder(self.cfg)

    def encode(self, patch_tokens: torch.Tensor, proprio: torch.Tensor | None = None) -> torch.Tensor:
        """Deployable path: DINO patch tokens -> workspace token(s). No decoder / no VLM."""
        return self.encoder(patch_tokens, proprio)

    def forward(self, patch_tokens: torch.Tensor, target_patches: torch.Tensor,
                target_mask: torch.Tensor, proprio: torch.Tensor | None = None):
        w = self.encoder(patch_tokens, proprio)
        recon_feats, existence_logits = self.decoder(w)
        losses = set_losses(recon_feats, existence_logits, target_patches, target_mask, self.cfg)
        return w, losses


if __name__ == "__main__":  # smoke test: shapes + loss compute on random inputs
    torch.manual_seed(0)
    cfg = WorkspaceConfig(proprio_dim=8)
    model = WorkspaceModel(cfg)
    B, N, P = 4, cfg.num_patches, cfg.max_patches
    patches = torch.randn(B, N, cfg.feat_dim)
    proprio = torch.randn(B, cfg.proprio_dim)
    target = torch.randn(B, P, cfg.feat_dim)
    mask = torch.rand(B, P) > 0.4          # variable-size salient sets (some padding)
    w, losses = model(patches, target, mask, proprio)
    assert w.shape == (B, cfg.num_workspace_tokens, cfg.hidden_dim), w.shape
    losses["total"].backward()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ok] w={tuple(w.shape)}  losses=" +
          " ".join(f"{k}={float(v):.4f}" for k, v in losses.items()) +
          f"  params={n_params/1e6:.2f}M")
    print(f"[ok] deploy encode -> {tuple(model.encode(patches, proprio).shape)}")
