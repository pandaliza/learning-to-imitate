"""Feature cache reader for B0/B1 training on frozen VL features.

Provides simple interface to load pre-extracted vision-language features:
  - Reads npz files from cache directory
  - Validates manifest
  - Supports both pooled and grid16 variants
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FeatureCache:
    """Read-only interface to pre-extracted VL feature cache."""

    def __init__(self, cache_root: str | Path, variant: str = "pooled"):
        """Initialize feature cache reader.

        Args:
            cache_root: path to b0b1_features/<variant>/ directory
            variant: "pooled" or "grid16" (for validation)
        """
        self.cache_root = Path(cache_root)
        self.variant = variant

        # Load and validate manifest
        manifest_path = self.cache_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found at {manifest_path}")

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        if self.manifest["variant"] != variant:
            raise ValueError(
                f"Variant mismatch: manifest has '{self.manifest['variant']}', "
                f"expected '{variant}'"
            )

        self.vision_dim = self.manifest["vision_dim"]
        self.lang_dim = self.manifest["lang_dim"]
        self.num_vision_tokens = self.manifest["num_vision_tokens"]
        self.cameras = self.manifest["cameras"]

    def load_episode(self, task: str, episode_stem: str) -> dict[str, np.ndarray]:
        """Load features for one episode.

        Args:
            task: task name (e.g., "TurnOnElectricKettle")
            episode_stem: episode filename without extension (e.g., "episode_000000")

        Returns:
            dict with keys:
              - "base": (N_frames, D) or (N_frames, 16, D)
              - "wrist": (N_frames, D) or (N_frames, 16, D)
              - "lang": (D_l,)
              - "num_frames": scalar int
        """
        npz_path = self.cache_root / "features" / task / f"{episode_stem}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Feature cache not found: {npz_path}")

        data = np.load(npz_path)
        num_frames = data["base"].shape[0]

        return {
            "base": data["base"],  # (N, D) or (N, 16, D), fp16
            "wrist": data["wrist"],  # (N, D) or (N, 16, D), fp16
            "lang": data["lang"],  # (D_l,), fp16
            "num_frames": num_frames,
        }

    def get_manifest(self) -> dict:
        """Return manifest metadata."""
        return self.manifest.copy()
