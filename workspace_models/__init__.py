"""Workspace Models (current-frame) — a saliency-supervised latent intent token.

Faithful re-implementation of the current-frame slice of Workspace Models (CoRL 2026):
DINOv3 patch tokens -> attention pooler -> workspace encoder (learned slot) -> workspace token,
distilled via a DETR-style set-reconstruction decoder (Hungarian matching + occupancy) against
a VLM-curated salient patch set. Intended as a drop-in *intent* representation for Pi0.5.
"""

from .config import WorkspaceConfig
from .model import WorkspaceModel
from .encoder import WorkspaceEncoder, AttentionPooler
from .decoder import WorkspaceDecoder
from .losses import set_losses
from .matching import match

__all__ = [
    "WorkspaceConfig", "WorkspaceModel", "WorkspaceEncoder", "AttentionPooler",
    "WorkspaceDecoder", "set_losses", "match",
]
