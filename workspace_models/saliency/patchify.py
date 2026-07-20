"""Convert VLM points -> DINO patch indices -> a padded salient patch set.

"...convert these points to DINO patches by simply choosing the unique patch in space which
overlaps with the point" (Sec. 3.2). The salient set is the DINO features at those patches,
deduplicated (unique patches) and capped at m = max_patches.
"""

import math

import numpy as np


def point_to_patch_index(x: float, y: float, grid_h: int, grid_w: int) -> int:
    """Normalized point (x, y) in [0, 1] -> flat patch index (row-major) on a grid_h x grid_w grid."""
    col = min(int(x * grid_w), grid_w - 1)
    row = min(int(y * grid_h), grid_h - 1)
    return row * grid_w + col


def points_to_indices(points: list[tuple[float, float]], num_patches: int, max_patches: int,
                      grid_hw: tuple[int, int] | None = None) -> np.ndarray:
    """points -> unique patch indices (encounter order, capped at m), -1-padded to max_patches.

    Storing indices (not features) keeps the DINO grid the single source of truth; the dataset
    gathers `grid[idx]` at load time. Returns int32 array of shape (max_patches,)."""
    gh, gw = grid_hw or (int(round(math.sqrt(num_patches))),) * 2
    assert gh * gw == num_patches, f"grid {gh}x{gw} != {num_patches}; pass grid_hw explicitly"
    seen, idxs = set(), []
    for x, y in points:
        idx = point_to_patch_index(x, y, gh, gw)
        if idx not in seen:
            seen.add(idx)
            idxs.append(idx)
        if len(idxs) >= max_patches:
            break
    out = np.full(max_patches, -1, dtype=np.int32)
    out[: len(idxs)] = idxs
    return out


def gather_salient(patch_grid: np.ndarray, salient_idx: np.ndarray):
    """salient_idx (m,) with -1 pad + grid (N, D) -> (target_patches (m, D), target_mask (m,))."""
    m, d = salient_idx.shape[0], patch_grid.shape[1]
    target = np.zeros((m, d), dtype=np.float32)
    mask = salient_idx >= 0
    if mask.any():
        target[mask] = patch_grid[salient_idx[mask]]
    return target, mask


def assemble_salient_set(points: list[tuple[float, float]], patch_grid: np.ndarray,
                         max_patches: int, grid_hw: tuple[int, int] | None = None):
    """points -> (target_patches (m, D) float32, target_mask (m,) bool).

    patch_grid: (N, D) DINO patch features for the frame. grid_hw defaults to a square sqrt(N).
    Unique patches, in encounter order, capped at max_patches; the rest is zero-padded (mask=0).
    """
    n, d = patch_grid.shape
    gh, gw = grid_hw or (int(round(math.sqrt(n))),) * 2
    assert gh * gw == n, f"grid {gh}x{gw} != {n} patches; pass grid_hw explicitly"

    seen, idxs = set(), []
    for x, y in points:
        idx = point_to_patch_index(x, y, gh, gw)
        if idx not in seen:
            seen.add(idx)
            idxs.append(idx)
        if len(idxs) >= max_patches:
            break

    target = np.zeros((max_patches, d), dtype=np.float32)
    mask = np.zeros((max_patches,), dtype=bool)
    if idxs:
        target[: len(idxs)] = patch_grid[idxs]
        mask[: len(idxs)] = True
    return target, mask
