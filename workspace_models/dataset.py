"""Stage-1 dataset: pair current-frame DINO grids with their VLM salient patch sets.

Loads the per-demo grid caches ({stem}__{demo}.npy) + salient-index caches ({stem}__{demo}.npz
from libero_adapter) and yields, per labeled frame:
    grid           (N, D)   — encoder input AND the pool the salient set is drawn from
    target_patches (m, D)   — gathered grid[salient_idx] (zero-padded)
    target_mask    (m,)     — True where a salient patch is real

No policy / no VLM in the loop (grids + salient indices are precomputed).
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .saliency.patchify import gather_salient


class WorkspaceGridDataset(Dataset):
    def __init__(self, vl_cache_dir: str, salient_dir: str, max_patches: int = 8):
        self.max_patches = max_patches
        # index every labeled frame as (grid_path, frame_idx, salient_idx_row)
        self.items: list[tuple[str, int, np.ndarray]] = []
        for npz_path in sorted(glob.glob(os.path.join(salient_dir, "*.npz"))):
            key = os.path.basename(npz_path)[:-4]                # "{stem}__{demo}"
            grid_path = os.path.join(vl_cache_dir, key + ".npy")
            if not os.path.exists(grid_path):
                continue
            z = np.load(npz_path)
            for fi, sidx in zip(z["frame_idx"], z["salient_idx"]):
                self.items.append((grid_path, int(fi), sidx.astype(np.int64)))
        assert self.items, f"no (grid, salient) pairs found in {salient_dir} / {vl_cache_dir}"
        self._cache: dict[str, np.ndarray] = {}

    def _grid(self, path: str) -> np.ndarray:
        if path not in self._cache:                              # mmap once per demo file
            self._cache[path] = np.load(path, mmap_mode="r")
        return self._cache[path]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        grid_path, fi, sidx = self.items[i]
        grid = np.asarray(self._grid(grid_path)[fi], dtype=np.float32)   # (N, D)
        target, mask = gather_salient(grid, sidx[: self.max_patches])
        return (torch.from_numpy(grid), torch.from_numpy(target), torch.from_numpy(mask))
