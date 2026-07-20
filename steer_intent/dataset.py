"""Stage-1 dataset for the CAUSAL workspace encoder.

Like workspace_models/dataset.py (single-frame) but yields a CAUSAL WINDOW of T consecutive grids
ending at each labeled frame, so the temporal encoder is supervised at the window's last step:

    patches (T, N, D)   — T past+current grids (left-padded by repeating the earliest)
    proprio (T, Dp)     — per-frame proprio (zeros unless a proprio source is wired; smoke default)
    lang    (T, Dl)     — per-frame language cond, broadcast from a per-task global (zeros default)
    target  (m, D)      — salient patches of the LAST (labeled) frame, zero-padded
    mask    (m,)        — True where a salient patch is real

Reuses the existing caches verbatim: vl grids {stem}__{demo}.npy (T_demo, N, D) and salient labels
{stem}__{demo}.npz (frame_idx [L], salient_idx [L, m]). proprio/lang are OPTIONAL: pass a lang_table
({stem: [Dl]} npz) to condition on real task language; otherwise they are zeros (plumbing smoke).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from workspace_models.saliency.patchify import gather_salient


class CausalWindowDataset(Dataset):
    def __init__(self, vl_cache_dir: str, salient_dir: str, max_patches: int, window: int,
                 proprio_dim: int = 0, lang_dim: int = 0, lang_table: str | None = None):
        self.max_patches, self.window = max_patches, window
        self.proprio_dim, self.lang_dim = proprio_dim, lang_dim
        # optional per-task language table {stem: [lang_dim]} (e.g. PaliGemma pool='lang' means)
        self.lang_table = dict(np.load(lang_table)) if lang_table else {}
        # index every labeled frame as (grid_path, task_stem, frame_idx, salient_idx_row)
        self.items: list[tuple[str, str, int, np.ndarray]] = []
        for npz_path in sorted(glob.glob(os.path.join(salient_dir, "*.npz"))):
            key = os.path.basename(npz_path)[:-4]                # "{stem}__{demo}"
            stem = key.split("__")[0]
            grid_path = os.path.join(vl_cache_dir, key + ".npy")
            if not os.path.exists(grid_path):
                continue
            z = np.load(npz_path)
            for fi, sidx in zip(z["frame_idx"], z["salient_idx"]):
                self.items.append((grid_path, stem, int(fi), sidx.astype(np.int64)))
        assert self.items, f"no (grid, salient) pairs found in {salient_dir} / {vl_cache_dir}"
        self._cache: dict[str, np.ndarray] = {}

    def _grid(self, path: str) -> np.ndarray:
        if path not in self._cache:                              # mmap once per demo file
            self._cache[path] = np.load(path, mmap_mode="r")
        return self._cache[path]

    def _causal_indices(self, fi: int) -> np.ndarray:
        """The T grid indices [fi-T+1 .. fi], clamped at 0 and left-padded by repeating frame 0.
        Never references a frame after fi (causal)."""
        idx = np.arange(fi - self.window + 1, fi + 1)
        return np.clip(idx, 0, None)                             # negatives -> 0 (repeat earliest)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        grid_path, stem, fi, sidx = self.items[i]
        g = self._grid(grid_path)
        win = self._causal_indices(fi)
        patches = np.asarray(g[win], dtype=np.float32)           # (T, N, D)
        target, mask = gather_salient(patches[-1], sidx[: self.max_patches])  # supervise last frame

        T = self.window
        proprio = np.zeros((T, self.proprio_dim), dtype=np.float32)
        lg = self.lang_table.get(stem)
        lang = (np.broadcast_to(lg, (T, self.lang_dim)).astype(np.float32) if lg is not None
                else np.zeros((T, self.lang_dim), dtype=np.float32))
        return (torch.from_numpy(patches), torch.from_numpy(proprio), torch.from_numpy(lang),
                torch.from_numpy(target), torch.from_numpy(mask))
