"""RoboCasa M11 co-prediction dataset: LeRobot format with intent targets (h=8 waypoints, Δ=2).

Ported from LIBERO via IntentFlowDataset. Reads RoboCasa episodes in LeRobot v2.1 format
(parquet + mp4 videos) and emits the exact batch interface expected by train_pi05_m11.py.

Batch interface (matching D2 trainer report, section 2):
  obs: {
    state: (obs_steps=2, state_dim=16)
    agentview_rgb: (obs_steps=2, 3, 224, 224)
    eye_in_hand_rgb: (obs_steps=2, 3, 224, 224)
  }
  action: (action_horizon=10, action_dim=12)
  wsm_intent_target: (num_waypoints=8, intent_dim=7) — normalized eef_rel
  task_id: scalar int64

Normalizer interface (matching IntentFlowDataset):
  ds.normalizer["obs"]["state"] — Normalizer with .normalize()/.unnormalize()
  ds.normalizer["action"] — Normalizer with .normalize()/.unnormalize()

Key design decisions:
  - d_I = 7 (RoboCasa: pos3 + quat4) vs LIBERO's 6D
  - intent = state[7:14] (end_effector_position_relative + end_effector_rotation_relative)
  - Images resized to 224x224 (pi0.5 standard)
  - IntentFlowDataset uses state[7:14] for LIBERO (eef), here it's the natural 7D layout
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False


class Normalizer:
    """Normalizer with normalize/unnormalize interface matching openpi's Normalizer."""

    def __init__(self, mean: np.ndarray | list, std: np.ndarray | list):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize x: (x - mean) / std"""
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / (self.std + 1e-8)

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        """Unnormalize x: x * std + mean"""
        x = np.asarray(x, dtype=np.float32)
        return x * self.std + self.mean


class RobocasaCopredDataset(Dataset):
    """LeRobot RoboCasa episodes with intent co-prediction targets for M11 trainer.

    Attributes:
        obs_steps: observation history length (default 2)
        action_horizon: H, number of action steps predicted (default 10)
        intent_horizon: reach in env steps (h*Δ, default 16→h=8 with Δ=2)
        lookahead_stride: Δ, stride between waypoints (default 2)
    """

    def __init__(
        self,
        root_dir: str | Path,
        task_names: list[str] | None = None,
        obs_steps: int = 2,
        action_horizon: int = 10,
        intent_horizon: int = 16,
        lookahead_stride: int = 2,
        normalize: bool = True,
        norm_stats_path: str | Path | None = None,
        load_images: bool = True,
    ):
        """Initialize RoboCasa co-prediction dataset.

        Args:
            root_dir: path to /target (contains atomic/, composite/ subdirs)
            task_names: list of task names. If None, use default 9 train tasks.
            obs_steps: observation history length (default 2)
            action_horizon: action chunk size H (default 10)
            intent_horizon: reach in env steps (default 16→h=8 with Δ=2)
            lookahead_stride: stride Δ between waypoints (default 2)
            normalize: whether to normalize state/action
            norm_stats_path: path to norm_stats.json for loading statistics
            load_images: whether to load video images (False for fast norm stats computation)
        """
        self.root_dir = Path(root_dir)
        self.obs_steps = obs_steps
        self.action_horizon = action_horizon
        self.intent_horizon = intent_horizon
        self.lookahead_stride = lookahead_stride
        self.normalize = normalize
        self.load_images = load_images

        # Default to the 9 train tasks
        if task_names is None:
            task_names = [
                "TurnOnElectricKettle",
                "PickPlaceCounterToCabinet",
                "PickPlaceCounterToStove",
                "SlideDishwasherRack",
                "KettleBoiling",
                "LoadDishwasher",
                "PrepareCoffee",
                "PreSoakPan",
                "WashLettuce",
            ]
        self.task_names = task_names

        # Validate intent horizon divisibility
        assert (
            intent_horizon % lookahead_stride == 0
        ), f"intent_horizon ({intent_horizon}) must be divisible by lookahead_stride ({lookahead_stride})"
        self.num_intent_waypoints = intent_horizon // lookahead_stride

        # Initialize normalizer structure matching IntentFlowDataset
        self.normalizer = {
            "obs": {"state": None},
            "action": None,
        }

        # Scan episodes
        self._episodes = []
        self._frame_index = []
        self._task_id_map = {}
        self._scan_episodes()

        # Load norm stats if provided
        if self.normalize and norm_stats_path:
            self._load_norm_stats(norm_stats_path)

        print(
            f"[RobocasaCopredDataset] loaded {len(self._episodes)} episodes, "
            f"{len(self._frame_index)} frames, obs_steps={obs_steps}, "
            f"action_horizon={action_horizon}, intent_waypoints={self.num_intent_waypoints}"
        )

    def _scan_episodes(self):
        """Scan task directories and build frame index."""
        for task_name in self.task_names:
            found = False
            for category in ["atomic", "composite"]:
                task_dir = self.root_dir / category / task_name
                if not task_dir.exists():
                    continue
                found = True

                for date_dir in sorted(task_dir.iterdir()):
                    if not date_dir.is_dir():
                        continue

                    lerobot_dir = date_dir / "lerobot"
                    if not lerobot_dir.exists():
                        continue

                    meta_dir = lerobot_dir / "meta"
                    episodes_path = meta_dir / "episodes.jsonl"
                    data_dir = lerobot_dir / "data" / "chunk-000"

                    if not episodes_path.exists() or not data_dir.exists():
                        continue

                    # Register task_id
                    if task_name not in self._task_id_map:
                        self._task_id_map[task_name] = len(self._task_id_map)
                    task_id = self._task_id_map[task_name]

                    # Read episodes
                    with open(episodes_path) as f:
                        episodes_meta = [json.loads(line) for line in f]

                    for ep_meta in episodes_meta:
                        ep_idx = len(self._episodes)
                        ep_num = ep_meta["episode_index"]
                        ep_len = ep_meta["length"]

                        parquet_path = data_dir / f"episode_{ep_num:06d}.parquet"
                        if not parquet_path.exists():
                            continue

                        ep_info = {
                            "episode_index": ep_idx,
                            "task_name": task_name,
                            "task_id": task_id,
                            "episode_number": ep_num,
                            "length": ep_len,
                            "date_dir": date_dir,
                            "data_dir": data_dir,
                            "parquet_path": parquet_path,
                        }
                        self._episodes.append(ep_info)

                        # Add valid frames (need obs_steps:t+1 and future intent to t+intent_horizon)
                        max_t = max(0, ep_len - 1 - self.intent_horizon)
                        for t in range(max(self.obs_steps - 1, 0), max_t + 1):
                            self._frame_index.append((ep_idx, t))

                if found:
                    break

        if not self._frame_index:
            raise ValueError(f"No episodes found in {self.root_dir} for tasks {self.task_names}")

    def _load_norm_stats(self, norm_stats_path: str | Path):
        """Load normalization statistics from norm_stats.json file."""
        norm_stats_path = Path(norm_stats_path)
        if not norm_stats_path.exists():
            print(f"[WARNING] norm_stats.json not found at {norm_stats_path}")
            return

        with open(norm_stats_path) as f:
            data = json.load(f)

        stats = data.get("norm_stats", {})

        if "state" in stats:
            self.normalizer["obs"]["state"] = Normalizer(stats["state"]["mean"], stats["state"]["std"])

        if "actions" in stats:
            self.normalizer["action"] = Normalizer(stats["actions"]["mean"], stats["actions"]["std"])

    def _get_video_frame(self, video_path: Path, frame_idx: int) -> np.ndarray | None:
        """Decode single frame from mp4 using PyAV."""
        if not HAS_PYAV:
            return None

        container = None
        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]

            # Target the frame by pts: seek lands on the keyframe at/before the target,
            # then decode forward until the first frame at/after the target time.
            # (Counting decoded frames against an absolute index after a seek is wrong --
            # the count restarts at the keyframe, not at frame 0.)
            fps = float(stream.guessed_rate or 20.0)
            target_pts = int(round(frame_idx / fps / float(stream.time_base)))
            container.seek(target_pts, stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is not None and frame.pts >= target_pts:
                    return frame.to_rgb().to_ndarray()
            return None
        except Exception:
            return None
        finally:
            if container is not None:
                container.close()

    def __len__(self) -> int:
        return len(self._frame_index)

    def __getitem__(self, idx: int) -> dict:
        """Get training sample matching train_pi05_m11.py batch interface."""
        ep_idx, t = self._frame_index[idx]
        ep_info = self._episodes[ep_idx]

        # Load episode parquet
        df = pd.read_parquet(ep_info["parquet_path"])

        # Extract observation frames [t - obs_steps + 1, t]
        obs_start = max(0, t - self.obs_steps + 1)
        obs_indices = list(range(obs_start, t + 1))

        # Pad if at episode start
        if len(obs_indices) < self.obs_steps:
            obs_indices = [obs_indices[0]] * (self.obs_steps - len(obs_indices)) + obs_indices

        # Load state (16D)
        obs_state = np.stack([df.iloc[i]["observation.state"].astype(np.float32) for i in obs_indices])

        # Normalize state if available
        if self.normalizer["obs"]["state"] is not None:
            obs_state = self.normalizer["obs"]["state"].normalize(obs_state)

        # Load images (224x224)
        if self.load_images:
            video_dir = ep_info["date_dir"] / "lerobot" / "videos" / "chunk-000"
            obs_agentview = []
            obs_wrist = []

            for frame_i in obs_indices:
                agentview_path = (
                    video_dir / "observation.images.robot0_agentview_left"
                    / f"episode_{ep_info['episode_number']:06d}.mp4"
                )
                agentview_frame = self._get_video_frame(agentview_path, frame_i)
                if agentview_frame is None:
                    agentview_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    agentview_frame = cv2.resize(agentview_frame, (224, 224))
                obs_agentview.append(agentview_frame / 255.0)

                wrist_path = (
                    video_dir / "observation.images.robot0_eye_in_hand"
                    / f"episode_{ep_info['episode_number']:06d}.mp4"
                )
                wrist_frame = self._get_video_frame(wrist_path, frame_i)
                if wrist_frame is None:
                    wrist_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    wrist_frame = cv2.resize(wrist_frame, (224, 224))
                obs_wrist.append(wrist_frame / 255.0)

            obs_agentview = np.stack(obs_agentview).transpose(0, 3, 1, 2)
            obs_wrist = np.stack(obs_wrist).transpose(0, 3, 1, 2)
        else:
            obs_agentview = np.zeros((self.obs_steps, 3, 224, 224), dtype=np.float32)
            obs_wrist = np.zeros((self.obs_steps, 3, 224, 224), dtype=np.float32)

        # Extract action frames [t, t + action_horizon)
        act_indices = list(range(t, min(t + self.action_horizon, len(df))))
        act_data = np.stack([df.iloc[i]["action"].astype(np.float32) for i in act_indices])

        # Pad actions if near episode end
        if len(act_indices) < self.action_horizon:
            pad_len = self.action_horizon - len(act_indices)
            act_data = np.pad(act_data, ((0, pad_len), (0, 0)), mode="edge")

        # Normalize actions if available
        if self.normalizer["action"] is not None:
            act_data = self.normalizer["action"].normalize(act_data)

        # Extract intent targets: h waypoints at [t+Δ, t+2Δ, ..., t+hΔ]
        # Clamp at episode end
        intent_indices = []
        for k in range(1, self.num_intent_waypoints + 1):
            idx = min(t + k * self.lookahead_stride, len(df) - 1)
            intent_indices.append(idx)

        # Extract eef_rel (state[7:14]) from intent frames
        intent_eef = np.stack([df.iloc[i]["observation.state"][7:14].astype(np.float32) for i in intent_indices])

        # Return batch dict matching D2 trainer expectations
        batch = {
            "obs": {
                "state": obs_state,  # (obs_steps, 16)
                "agentview_rgb": obs_agentview,  # (obs_steps, 3, 224, 224)
                "eye_in_hand_rgb": obs_wrist,  # (obs_steps, 3, 224, 224)
            },
            "action": act_data,  # (action_horizon, 12)
            "task_id": np.array(ep_info["task_id"], dtype=np.int64),
            "wsm_intent_target": intent_eef,  # (num_waypoints, 7)
        }

        return batch


def make_robocasa_copred_dataset(
    root_dir: str | Path,
    task_names: list[str] | None = None,
    obs_steps: int = 2,
    action_horizon: int = 10,
    intent_horizon: int = 16,
    lookahead_stride: int = 2,
    normalize: bool = True,
    norm_stats_path: str | Path | None = None,
    load_images: bool = True,
) -> RobocasaCopredDataset:
    """Factory function to create a RoboCasa co-prediction dataset."""
    return RobocasaCopredDataset(
        root_dir=root_dir,
        task_names=task_names,
        obs_steps=obs_steps,
        action_horizon=action_horizon,
        intent_horizon=intent_horizon,
        lookahead_stride=lookahead_stride,
        normalize=normalize,
        norm_stats_path=norm_stats_path,
        load_images=load_images,
    )
