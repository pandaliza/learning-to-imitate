"""LIBERO dataset.

Loads LIBERO HDF5 demonstration files into the MIP training pipeline.
HDF5 layout mirrors robomimic: data/{demo_N}/actions, data/{demo_N}/obs/{key}.
"""

import bisect
import os

import h5py
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from mip.dataset_utils import (
    EmptyNormalizer,
    MinMaxNormalizer,
    ReplayBuffer,
    SequenceSampler,
    dict_apply,
)
from mip.datasets.base import BaseDataset


class _LazyVLGrids:
    """Disk-backed, episode-concatenated view over per-demo VL-grid .npy memmaps.

    Injected into the ReplayBuffer under the "vl_grid" key so the SequenceSampler windows it
    exactly like the pixel frames — but WITHOUT loading the full ~67GB cache into RAM. The
    sampler only ever slices a contiguous range lying within a single episode, so each slice
    maps to exactly one demo file (read from its memmap on demand)."""

    def __init__(self, specs):  # specs: list of (npy_path, episode_length) in episode order
        self._mm, self._starts, off = [], [], 0
        shape1 = dtype = None
        for path, n in specs:
            m = np.load(path, mmap_mode="r")
            assert m.shape[0] == n, f"VL grid {path}: len {m.shape[0]} != episode len {n}"
            self._mm.append(m)
            self._starts.append(off)
            off += n
            shape1, dtype = m.shape[1:], m.dtype
        self._starts.append(off)
        self.shape = (off,) + tuple(shape1)
        self.dtype = dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, sl):  # contiguous slice within ONE episode (sampler guarantee)
        ep = bisect.bisect_right(self._starts, sl.start) - 1
        base = self._starts[ep]
        return np.asarray(self._mm[ep][sl.start - base : sl.stop - base])


def make_dataset(task_config, mode="train"):
    """Create a LIBERO dataset from config."""
    if hasattr(task_config, "dataset_path") and task_config.dataset_path:
        dataset_paths = [os.path.expanduser(task_config.dataset_path)]
    elif hasattr(task_config, "dataset_paths") and task_config.dataset_paths:
        dataset_paths = [os.path.expanduser(p) for p in task_config.dataset_paths]
    else:
        raise ValueError("task_config must provide dataset_path or dataset_paths")

    intent_conditioning = getattr(task_config, "intent_conditioning", False)
    intent_horizon = getattr(task_config, "intent_horizon", task_config.act_steps)
    intent_type = getattr(task_config, "intent_type", "mean")
    intent_keys = getattr(task_config, "intent_keys", ["ee_states"])
    image_obs_keys = list(getattr(task_config, "image_obs_keys", None) or [])
    task_id_conditioning = getattr(task_config, "task_id_conditioning", False)
    num_tasks = len(dataset_paths) if task_id_conditioning else 1

    if intent_conditioning:
        pad_after = max(task_config.act_steps, intent_horizon) - 1
        dataset_horizon = max(
            task_config.horizon, task_config.obs_steps + intent_horizon
        )
    else:
        pad_after = task_config.act_steps - 1
        dataset_horizon = task_config.horizon

    slot_image_key = getattr(task_config, "slot_image_key", "agentview_rgb")
    slot_obj_state_key = getattr(task_config, "slot_obj_state_key", "ee_states")
    slot_obj_state_dim = getattr(task_config, "slot_obj_state_dim", -1)
    vl_cache_dir = getattr(task_config, "vl_cache_dir", None)
    wsm_w_cache_dir = getattr(task_config, "wsm_w_cache_dir", None)

    return LiberoDataset(
        dataset_paths=dataset_paths,
        obs_keys=task_config.obs_keys,
        image_obs_keys=image_obs_keys,
        horizon=dataset_horizon,
        pad_before=task_config.obs_steps - 1,
        pad_after=pad_after,
        obs_steps=task_config.obs_steps,
        intent_conditioning=intent_conditioning,
        intent_horizon=intent_horizon,
        intent_type=intent_type,
        intent_keys=intent_keys,
        task_id_conditioning=task_id_conditioning,
        num_tasks=num_tasks,
        slot_image_key=slot_image_key,
        slot_obj_state_key=slot_obj_state_key,
        slot_obj_state_dim=slot_obj_state_dim,
        vl_cache_dir=vl_cache_dir,
        wsm_w_cache_dir=wsm_w_cache_dir,
    )


class LiberoDataset(BaseDataset):
    """Dataset for LIBERO HDF5 demonstrations.

    Args:
        dataset_paths: List of .hdf5 file paths to load.
        obs_keys: List of state observation keys to concatenate.
        image_obs_keys: List of image observation keys (e.g. ["agentview_rgb"]).
            When non-empty, obs_type becomes "image" and images are loaded
            as (T, C, H, W) uint8 alongside the state vector.
        horizon: Sequence length drawn per sample.
        pad_before: Left padding (obs_steps - 1).
        pad_after: Right padding (act_steps - 1).
        obs_steps: Number of observation history steps.
        intent_conditioning: Whether to extract intent from future eef.
        intent_horizon: Number of future steps for intent computation.
        intent_type: "mean", "encoded_mean", or "slot".
        intent_keys: HDF5 obs keys used for intent (e.g. ["ee_states"]).
        slot_image_key: image key to use for slot attention frames (intent_type="slot").
        slot_obj_state_key: HDF5 obs key for slot aux supervision (intent_type="slot").
        slot_obj_state_dim: expected dim of slot_obj_state_key; -1 = auto-infer.
    """

    def __init__(
        self,
        dataset_paths: list[str],
        obs_keys: list[str],
        image_obs_keys: list[str] | None = None,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        obs_steps: int = 2,
        intent_conditioning: bool = False,
        intent_horizon: int = 8,
        intent_type: str = "mean",
        intent_keys: list[str] | None = None,
        task_id_conditioning: bool = False,
        num_tasks: int = 1,
        slot_image_key: str = "agentview_rgb",
        slot_obj_state_key: str = "ee_states",
        slot_obj_state_dim: int = -1,
        vl_cache_dir: str | None = None,
        wsm_w_cache_dir: str | None = None,
    ):
        super().__init__()
        self.obs_keys = obs_keys
        self.image_obs_keys = image_obs_keys or []
        self.obs_type = "image" if self.image_obs_keys else "state"
        self.obs_steps = obs_steps
        self.intent_conditioning = intent_conditioning
        self.intent_horizon = intent_horizon
        self.intent_type = intent_type
        self.intent_keys = intent_keys or ["ee_states"]
        self.task_id_conditioning = task_id_conditioning
        self.num_tasks = num_tasks
        self.slot_image_key = slot_image_key
        self.slot_obj_state_key = slot_obj_state_key
        self.slot_obj_state_dim = slot_obj_state_dim
        self.vl_cache_dir = vl_cache_dir
        self.wsm_w_cache_dir = wsm_w_cache_dir
        self._vl_specs = []  # (cache_path, episode_length) per demo, in load order
        self._w_specs = []   # (w_npy_path, episode_length) per demo, for the wsm arm
        self.lowdim_keys = ["state"]
        self._state_key_dims = {}

        self.replay_buffer = ReplayBuffer.create_empty_numpy()

        for task_id, path in enumerate(dataset_paths):
            logger.info(f"Loading LIBERO dataset: {path}")
            self._load_hdf5(path, task_id=task_id)

        logger.info(
            f"Loaded {self.replay_buffer.n_episodes} episodes, "
            f"{self.replay_buffer.n_steps} total steps"
        )

        # VL-grounded slot intent: inject the frozen-VL grid cache as a lazy, disk-backed
        # "vl_grid" key BEFORE the sampler is built, so it is windowed exactly like the
        # pixel frames without loading ~67GB into RAM (see _LazyVLGrids).
        if self.vl_cache_dir:
            lazy = _LazyVLGrids(self._vl_specs)
            assert lazy.shape[0] == self.replay_buffer.n_steps, (
                f"VL grid total {lazy.shape[0]} != buffer steps {self.replay_buffer.n_steps}")
            self.replay_buffer.data["vl_grid"] = lazy
            logger.info(f"[LiberoDataset] VL-grid cache: {len(self._vl_specs)} demos, grid {lazy.shape}")

        # wsm arm: inject the per-demo frozen workspace latents w [T, dim] as a lazy "w" key so the
        # sampler windows them with the frames (w_t = current frame, w_{t+1} = next frame in the window).
        if self.wsm_w_cache_dir:
            lazy_w = _LazyVLGrids(self._w_specs)
            assert lazy_w.shape[0] == self.replay_buffer.n_steps, (
                f"w cache total {lazy_w.shape[0]} != buffer steps {self.replay_buffer.n_steps}")
            self.replay_buffer.data["w"] = lazy_w
            logger.info(f"[LiberoDataset] w cache: {len(self._w_specs)} demos, w {lazy_w.shape}")

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )
        self.horizon = horizon

        # Track where the intent source lives inside the concatenated state vector.
        self.intent_start = None
        self.intent_end = None
        offset = 0
        for key in self.obs_keys:
            key_dim = self._state_key_dims.get(key)
            if key_dim is None:
                continue
            if key in self.intent_keys and self.intent_start is None:
                self.intent_start = offset
            if key in self.intent_keys:
                self.intent_end = offset + key_dim
            offset += key_dim

        self.normalizer = self._get_normalizer()

    def _load_hdf5(self, path: str, task_id: int = 0):
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys())
            for demo_key in tqdm(demos, desc=f"Loading {os.path.basename(path)}"):
                actions = f[f"data/{demo_key}/actions"][()].astype(np.float32)
                obs_parts = []
                for key in self.obs_keys:
                    arr = f[f"data/{demo_key}/obs/{key}"][()].astype(np.float32)
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    if key not in self._state_key_dims:
                        self._state_key_dims[key] = int(arr.shape[-1])
                    obs_parts.append(arr)
                state = np.concatenate(obs_parts, axis=-1)

                episode = {"state": state, "action": actions}
                if self.task_id_conditioning:
                    episode["task_id"] = np.full(len(actions), task_id, dtype=np.int64)

                # Image observations: load as uint8 CHW (convert from HWC).
                for img_key in self.image_obs_keys:
                    imgs = f[f"data/{demo_key}/obs/{img_key}"][()]  # (T, H, W, C)
                    imgs = np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))  # (T, C, H, W)
                    episode[img_key] = imgs  # uint8

                if self.intent_conditioning or self.wsm_w_cache_dir:  # wsm intent target also needs eef
                    eef_parts = []
                    for key in self.intent_keys:
                        arr = f[f"data/{demo_key}/obs/{key}"][()].astype(np.float32)
                        if arr.ndim == 1:
                            arr = arr[:, None]
                        eef_parts.append(arr)
                    episode["eef"] = np.concatenate(eef_parts, axis=-1)

                if self.intent_type == "slot":
                    slot_obj = f[f"data/{demo_key}/obs/{self.slot_obj_state_key}"][()].astype(np.float32)
                    if slot_obj.ndim == 1:
                        slot_obj = slot_obj[:, None]
                    if self.slot_obj_state_dim == -1:
                        self.slot_obj_state_dim = int(slot_obj.shape[-1])
                        logger.info(f"[LiberoDataset] Auto-inferred slot_obj_state_dim={self.slot_obj_state_dim} from '{self.slot_obj_state_key}'")
                    episode["slot_obj_state"] = slot_obj

                if self.vl_cache_dir:
                    stem = os.path.basename(path).replace("_demo.hdf5", "").replace(".hdf5", "")
                    cache_path = os.path.join(self.vl_cache_dir, f"{stem}__{demo_key}.npy")
                    self._vl_specs.append((cache_path, len(actions)))

                if self.wsm_w_cache_dir:
                    stem = os.path.basename(path).replace("_demo.hdf5", "").replace(".hdf5", "")
                    self._w_specs.append(
                        (os.path.join(self.wsm_w_cache_dir, f"{stem}__{demo_key}.npy"), len(actions)))

                self.replay_buffer.add_episode(episode)

    def _get_normalizer(self):
        state_normalizer = MinMaxNormalizer(self.replay_buffer["state"][:])
        action_normalizer = MinMaxNormalizer(self.replay_buffer["action"][:])
        norm = {"obs": {"state": state_normalizer}, "action": action_normalizer}
        for img_key in self.image_obs_keys:
            # LIBERO images are already represented in the same [0, 1] range at
            # train and eval time, so keep the image path identity-normalized.
            norm["obs"][img_key] = EmptyNormalizer()
        if self.intent_conditioning or self.wsm_w_cache_dir:
            norm["eef"] = MinMaxNormalizer(self.replay_buffer["eef"][:])
        if self.intent_type == "slot":
            norm["slot_obj_state"] = MinMaxNormalizer(self.replay_buffer["slot_obj_state"][:])
        return norm

    def sample_to_data(self, sample):
        state = sample["state"].astype(np.float32)
        state = self.normalizer["obs"]["state"].normalize(state)
        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        obs = {"state": state}
        # Images: (horizon, C, H, W) uint8 → float32 in [0, 1]
        for img_key in self.image_obs_keys:
            obs[img_key] = sample[img_key].astype(np.float32) / 255.0

        data = {"obs": obs, "action": action}

        if self.intent_conditioning and self.intent_type != "slot":
            eef = sample["eef"].astype(np.float32)
            eef_normed = self.normalizer["eef"].normalize(eef)
            future_eef = eef_normed[self.obs_steps : self.obs_steps + self.intent_horizon]
            if self.intent_type == "encoded_mean":
                data["intent"] = future_eef.astype(np.float32)  # (N, eef_dim)
            else:  # "mean"
                data["intent"] = future_eef.mean(axis=0).astype(np.float32)  # (eef_dim,)

        if self.intent_type == "slot":
            # intent_frames: future image frames (K, C, H, W) float32 in [0, 1]
            imgs = sample[self.slot_image_key]  # (horizon, C, H, W) uint8
            future_imgs = imgs[self.obs_steps : self.obs_steps + self.intent_horizon]
            data["intent_frames"] = future_imgs.astype(np.float32) / 255.0
            # object_states: future obj state normalized (K, obj_state_dim)
            slot_obj = sample["slot_obj_state"].astype(np.float32)
            slot_obj_normed = self.normalizer["slot_obj_state"].normalize(slot_obj)
            data["object_states"] = slot_obj_normed[self.obs_steps : self.obs_steps + self.intent_horizon]
            if self.vl_cache_dir:
                # VL-grounded slot input: frozen-VL agentview grids for the future window
                # (slot encoder z* input). intent_frames above stays as the pixel-recon target.
                grid = sample["vl_grid"]  # (horizon, N, vl_dim) fp16, windowed by the sampler
                data["intent_vl_grids"] = (
                    grid[self.obs_steps : self.obs_steps + self.intent_horizon].astype(np.float32))
                # current-obs VL mean (last obs frame) -> flow-map generator conditioning.
                data["vl_obs_mean"] = grid[self.obs_steps - 1].astype(np.float32).mean(axis=0)  # (vl_dim,)

        if self.wsm_w_cache_dir:
            # Frozen causal workspace latents, windowed by the sampler. w_t = current obs frame
            # (causal, past-only); w_{t+1} = next frame (the JEPA target). Different tensors.
            w = sample["w"].astype(np.float32)                          # (horizon, w_dim)
            cur = self.obs_steps - 1
            data["wsm_w_t"] = w[cur]                                    # (w_dim,)
            data["wsm_w_next"] = w[min(cur + 1, w.shape[0] - 1)]       # (w_dim,) clamp at episode end
            # intent target: mean future EEF pose over the intent horizon (normalized, same as intent path)
            eef_normed = self.normalizer["eef"].normalize(sample["eef"].astype(np.float32))
            future_eef = eef_normed[self.obs_steps : self.obs_steps + self.intent_horizon]
            data["wsm_intent_target"] = future_eef.mean(axis=0).astype(np.float32)  # (eef_dim,)

        if self.task_id_conditioning:
            data["task_id"] = np.array(int(sample["task_id"][0]), dtype=np.int64)

        return data

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict:
        sample = self.sampler.sample_sequence(idx)
        data = self.sample_to_data(sample)
        return dict_apply(data, torch.tensor)
