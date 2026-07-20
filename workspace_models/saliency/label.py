"""Orchestrate saliency labeling over frames -> per-frame salient patch sets.

Current-frame path only: for each frame, point at the task's persistent objects (MolmoPoint),
map points -> DINO patches -> salient set. Output aligns 1:1 with the precomputed DINO grids that
feed WorkspaceModel training.

CLI (mock backend, no VLM weights needed) for a smoke run:
    python -m workspace_models.saliency.label --demo
"""

import re
from dataclasses import dataclass

import numpy as np

from .patchify import assemble_salient_set
from .pointing import Pointer, MockPointer


@dataclass
class LabelConfig:
    max_patches: int = 8                      # m; must match WorkspaceConfig.max_patches
    grid_hw: tuple[int, int] | None = None    # DINO grid (defaults to square sqrt(N))


def objects_from_task_language(lang: str) -> list[str]:
    """Heuristic: noun phrases after 'the'/'a' (e.g. 'put the bowl on the stove' -> [bowl, stove]).

    A convenience only — override with a curated per-task object list for best pointing."""
    return re.findall(r"\b(?:the|a|an)\s+([a-z][a-z ]*?)(?=\s+(?:on|in|to|into|onto|off|and|at)\b|$)",
                      lang.lower())


def label_frame(image: np.ndarray, patch_grid: np.ndarray, objects: list[str],
                pointer: Pointer, cfg: LabelConfig):
    """One frame -> (target_patches (m, D), target_mask (m,)). Points collected across all objects."""
    points: list[tuple[float, float]] = []
    for obj in objects:
        points.extend(pointer.point(image, obj))
    return assemble_salient_set(points, patch_grid, cfg.max_patches, cfg.grid_hw)


def label_dataset(frames, pointer: Pointer, cfg: LabelConfig):
    """frames: iterable of (image (H,W,3), patch_grid (N,D), objects list[str]).

    Returns stacked (target_patches (F, m, D), target_mask (F, m)) aligned with the input order."""
    tps, masks = [], []
    for image, patch_grid, objects in frames:
        tp, m = label_frame(image, patch_grid, objects, pointer, cfg)
        tps.append(tp)
        masks.append(m)
    return np.stack(tps), np.stack(masks)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run a mock smoke test (no VLM)")
    ap.add_argument("--num-patches", type=int, default=196)
    ap.add_argument("--feat-dim", type=int, default=768)
    args = ap.parse_args()

    if args.demo:
        rng = np.random.default_rng(0)
        cfg = LabelConfig(max_patches=8)
        frames = [(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                   rng.standard_normal((args.num_patches, args.feat_dim)).astype(np.float32),
                   objects_from_task_language("put the bowl on the stove"))
                  for _ in range(5)]
        tp, mask = label_dataset(frames, MockPointer(), cfg)
        print(f"[ok] objects={objects_from_task_language('put the bowl on the stove')}")
        print(f"[ok] target_patches={tp.shape}  target_mask={mask.shape}  "
              f"avg salient/frame={mask.sum(1).mean():.2f}")
