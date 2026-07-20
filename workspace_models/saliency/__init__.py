"""VLM saliency labeling (current-frame): point at task objects -> DINO patches -> salient set.

Produces the `(target_patches, target_mask)` supervision that WorkspaceModel distills. Pointing
backends are pluggable (`MockPointer` for tests, `MolmoPointer` for the real model).
"""

from .pointing import Pointer, MockPointer, MolmoPointer
from .patchify import point_to_patch_index, assemble_salient_set
from .label import LabelConfig, label_frame, label_dataset, objects_from_task_language

__all__ = [
    "Pointer", "MockPointer", "MolmoPointer",
    "point_to_patch_index", "assemble_salient_set",
    "LabelConfig", "label_frame", "label_dataset", "objects_from_task_language",
]
