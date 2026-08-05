"""Feature-based dataset adapter for B0/B1 training on cached VL features.

Reads precomputed features from /data/group_data/maxlab/common_datasets/pandaliza/b0b1_features/
instead of raw video frames. Interfaces with RobocasaCopredDataset's intent/action logic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from steer_intent.robocasa_copred_dataset import Normalizer, RobocasaCopredDataset


class FeatureBasedCopredDataset(Dataset):
    """Wraps RobocasaCopredDataset but loads cached VL features instead of images.

    The underlying RobocasaCopredDataset handles:
      - Episode scanning and frame indexing
      - Intent target extraction (8 waypoints, stride 2)
      - Action extraction and normalization
      - State normalization

    This adapter:
      - Replaces image loading with cached feature loading
      - Keeps all other batch structure identical
    """

    def __init__(
        self,
        root_dir: str | Path,
        feature_cache_dir: str | Path,
        task_names: list[str] | None = None,
        obs_steps: int = 2,
        action_horizon: int = 10,
        intent_horizon: int = 16,
        lookahead_stride: int = 2,
        normalize: bool = True,
        norm_stats_path: str | Path | None = None,
    ):
        """Initialize feature-based co-prediction dataset.

        Args:
            root_dir: RoboCasa /target directory
            feature_cache_dir: /data/group_data/maxlab/common_datasets/pandaliza/b0b1_features/<variant>
            task_names: list of task names (default: 9 train tasks)
            obs_steps: observation history length
            action_horizon: action chunk size
            intent_horizon: reach in env steps
            lookahead_stride: stride between waypoints
            normalize: whether to normalize state/action
            norm_stats_path: path to norm_stats.json
        """
        self.feature_cache_dir = Path(feature_cache_dir)
        self.load_images = False  # Don't load images from RobocasaCopredDataset

        # Initialize underlying dataset (without image loading)
        self.base_dataset = RobocasaCopredDataset(
            root_dir=root_dir,
            task_names=task_names,
            obs_steps=obs_steps,
            action_horizon=action_horizon,
            intent_horizon=intent_horizon,
            lookahead_stride=lookahead_stride,
            normalize=normalize,
            norm_stats_path=norm_stats_path,
            load_images=False,
        )

        # Load feature cache manifest
        manifest_path = self.feature_cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Feature manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.variant = self.manifest.get("variant", "pooled")
        self.vision_dim = self.manifest.get("vision_dim")
        self.num_vision_tokens = self.manifest.get("num_vision_tokens", 1)
        self.lang_dim = self.manifest.get("lang_dim")
        self.dtype = self.manifest.get("dtype", "float16")

        print(
            f"[FeatureBasedCopredDataset] variant={self.variant}, "
            f"vision_dim={self.vision_dim}, num_vision_tokens={self.num_vision_tokens}, "
            f"lang_dim={self.lang_dim}"
        )

        # Cache loaded features: (task_name, episode_number) -> npz_data
        self._feature_cache = {}

    def _load_features(self, task_name: str, episode_number: int) -> dict[str, np.ndarray]:
        """Load cached features for an episode.

        Args:
            task_name: task name (e.g., "TurnOnElectricKettle")
            episode_number: episode number

        Returns:
            dict with keys: "base" (N, D) or (N, 16, D), "wrist" (N, D) or (N, 16, D), "lang" (D_l,)
        """
        key = (task_name, episode_number)
        if key in self._feature_cache:
            return self._feature_cache[key]

        feature_path = (
            self.feature_cache_dir / "features" / task_name / f"episode_{episode_number:06d}.npz"
        )
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        data = np.load(feature_path, allow_pickle=True)
        features = {
            "base": data["base"],  # (N, D) or (N, 16, D)
            "wrist": data["wrist"],  # (N, D) or (N, 16, D)
            "lang": data["lang"],  # (D_l,)
        }
        self._feature_cache[key] = features
        return features

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        """Get training sample with cached features.

        Returns dict matching train_b0b1.py expectations:
          obs: {
            state: (obs_steps, state_dim=16)
            vision: (obs_steps * 2, num_vision_tokens, vision_dim)
          }
          action: (action_horizon, action_dim=12)
          wsm_intent_target: (num_waypoints, intent_dim=7)
          task_id: scalar int64
        """
        # Get base batch from underlying dataset (without images)
        batch = self.base_dataset[idx]

        # Extract episode info
        ep_idx, t = self.base_dataset._frame_index[idx]
        ep_info = self.base_dataset._episodes[ep_idx]
        task_name = ep_info["task_name"]
        episode_number = ep_info["episode_number"]

        # Load cached features
        features = self._load_features(task_name, episode_number)

        # Extract observation features [t - obs_steps + 1, t]
        obs_start = max(0, t - self.base_dataset.obs_steps + 1)
        obs_indices = list(range(obs_start, t + 1))

        # Pad if at episode start
        if len(obs_indices) < self.base_dataset.obs_steps:
            obs_indices = [obs_indices[0]] * (self.base_dataset.obs_steps - len(obs_indices)) + obs_indices

        # Extract vision features for both cameras
        base_features = features["base"][obs_indices]  # (obs_steps, D) or (obs_steps, 16, D)
        wrist_features = features["wrist"][obs_indices]  # (obs_steps, D) or (obs_steps, 16, D)
        lang_features = features["lang"]  # (D_l,)

        # Concatenate base and wrist: (obs_steps*2, D) or (obs_steps*2, 16, D)
        if base_features.ndim == 2:
            # Pooled variant: (obs_steps, D) + (obs_steps, D) -> (obs_steps*2, D)
            vision_features = np.concatenate([base_features, wrist_features], axis=0)
        else:
            # Grid16 variant: (obs_steps, 16, D) + (obs_steps, 16, D) -> (obs_steps*2, 16, D)
            vision_features = np.concatenate([base_features, wrist_features], axis=0)

        # Convert to float32
        vision_features = vision_features.astype(np.float32)
        lang_features = lang_features.astype(np.float32)

        # Return modified batch
        return {
            "obs": {
                "state": batch["obs"]["state"],  # (obs_steps, 16)
                "vision": vision_features,  # (obs_steps*2, D) or (obs_steps*2, 16, D)
            },
            "action": batch["action"],  # (action_horizon, 12)
            "wsm_intent_target": batch["wsm_intent_target"],  # (num_waypoints, 7)
            "lang": lang_features,  # (D_l,)
            "task_id": batch["task_id"],
        }


def make_feature_based_dataset(
    root_dir: str | Path,
    feature_cache_dir: str | Path,
    task_names: list[str] | None = None,
    obs_steps: int = 2,
    action_horizon: int = 10,
    intent_horizon: int = 16,
    lookahead_stride: int = 2,
    normalize: bool = True,
    norm_stats_path: str | Path | None = None,
) -> FeatureBasedCopredDataset:
    """Factory function."""
    return FeatureBasedCopredDataset(
        root_dir=root_dir,
        feature_cache_dir=feature_cache_dir,
        task_names=task_names,
        obs_steps=obs_steps,
        action_horizon=action_horizon,
        intent_horizon=intent_horizon,
        lookahead_stride=lookahead_stride,
        normalize=normalize,
        norm_stats_path=norm_stats_path,
    )
