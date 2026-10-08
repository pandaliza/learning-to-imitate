"""OopsieVerse (ut-robin-lab/oopsieverse-demos): download, inspection, and state datasets.

Only the RoboCasa tasks (pick_egg, wipe_countertop) are trainable; the BEHAVIOR-1K
tasks store an unlabeled OmniGibson state vector. See docs/oopsieverse_dataset.md.
"""

import h5py
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from loguru import logger

from mip.dataset_utils import MinMaxNormalizer
from mip.datasets.base import BaseDataset
from mip.datasets.robomimic_dataset import RobomimicDataset

REPO_ID = "ut-robin-lab/oopsieverse-demos"

# task -> simulator backend / HDF5 schema
TASK_SCHEMA = {
    "add_firewood": "behavior1k",
    "shelve_item": "behavior1k",
    "pick_egg": "robocasa",
    "wipe_countertop": "robocasa",
}

VARIANTS = ("without_live_feedback", "with_live_feedback", "all_data")
CONDITIONS = ("without_live_feedback", "with_live_feedback")
STATE_TASKS = tuple(t for t, s in TASK_SCHEMA.items() if s == "robocasa")


def _remote_filename(task: str, variant: str) -> str:
    if task not in TASK_SCHEMA:
        raise ValueError(f"Unknown OopsieVerse task: {task}. Options: {list(TASK_SCHEMA)}")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Options: {VARIANTS}")
    return f"paper_experiments/teleop_data/{task}/{variant}.hdf5"


def download_oopsieverse(task: str, variant: str = "all_data") -> str:
    """Download one task/variant HDF5 into the HuggingFace cache and return its local path."""
    filename = _remote_filename(task, variant)
    logger.info(f"Downloading OopsieVerse dataset: {REPO_ID}/{filename}")
    path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    logger.info(f"Downloaded to: {path}")
    return path


def download_all_oopsieverse(
    tasks: list[str] | None = None, variants: list[str] | None = None
) -> dict[str, dict[str, str]]:
    """Download multiple task/variant combinations. Returns {task: {variant: local_path}}."""
    tasks = tasks or list(TASK_SCHEMA)
    variants = variants or list(VARIANTS)
    return {task: {v: download_oopsieverse(task, v) for v in variants} for task in tasks}


def inspect_oopsieverse(path: str, max_demos: int = 1) -> None:
    """Print a schema summary of a downloaded OopsieVerse HDF5 file."""
    with h5py.File(path, "r") as f:
        data = f["data"]
        demo_names = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
        logger.info(f"{path}")
        logger.info(f"  n_demos: {len(demo_names)}")
        for name in demo_names[:max_demos]:
            demo = data[name]
            logger.info(f"  {name} attrs: {list(demo.attrs.keys())}")
            if "action" in demo:
                logger.info(f"    schema=behavior1k action={demo['action'].shape} state={demo['state'].shape}")
            else:
                logger.info(
                    f"    schema=robocasa actions={demo['actions'].shape} "
                    f"states={demo['states'].shape} obs_keys={list(demo['obs'].keys())}"
                )


class OopsieverseStateDataset(BaseDataset):
    """Per-condition RobomimicDatasets concatenated under one shared normalizer."""

    def __init__(self, parts: dict[str, RobomimicDataset], normalizer: dict | None = None):
        super().__init__()
        if normalizer is None:
            normalizer = {
                "obs": {"state": MinMaxNormalizer(np.concatenate([p.replay_buffer["obs"][:] for p in parts.values()]))},
                "action": MinMaxNormalizer(np.concatenate([p.replay_buffer["action"][:] for p in parts.values()])),
            }
        for part in parts.values():
            part.normalizer = normalizer
        self.parts = parts
        self.normalizer = normalizer
        self._concat = torch.utils.data.ConcatDataset(list(parts.values()))

        first = next(iter(parts.values()))
        self.intent_start = first.intent_start
        self.intent_end = first.intent_end
        self.intent_slices = first.intent_slices

    def get_normalizer(self):
        return self.normalizer

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._concat[idx]


def _make_condition_dataset(task_config, condition: str, mode: str) -> RobomimicDataset:
    intent_horizon = getattr(task_config, "intent_horizon", task_config.act_steps)
    intent_type = getattr(task_config, "intent_type", "mean")
    # Same padding rule as the robomimic state path: future steps must cover intent extraction.
    if intent_type == "cv_proxy":
        pad_after = task_config.act_steps - 1
        horizon = task_config.horizon
    else:
        pad_after = max(task_config.act_steps, intent_horizon) - 1
        horizon = max(task_config.horizon, task_config.obs_steps + intent_horizon)
    return RobomimicDataset(
        download_oopsieverse(task_config.env_name, condition),
        horizon=horizon,
        obs_keys=task_config.obs_keys,
        pad_before=task_config.obs_steps - 1,
        pad_after=pad_after,
        abs_action=task_config.abs_action,
        mode=mode,
        val_dataset_percentage=task_config.val_dataset_percentage,
        intent_conditioning=getattr(task_config, "intent_conditioning", False),
        obs_steps=task_config.obs_steps,
        act_steps=task_config.act_steps,
        intent_horizon=intent_horizon,
        intent_type=intent_type,
        intent_keys=getattr(task_config, "intent_keys", ["robot0_eef_pos", "robot0_eef_quat"]),
    )


def make_dataset(task_config, mode: str = "train") -> OopsieverseStateDataset:
    """Training set over `task_config.conditions`, holding out the last val_dataset_percentage of each."""
    if task_config.env_name not in STATE_TASKS:
        raise ValueError(f"OopsieVerse task {task_config.env_name} has no state observations; use one of {STATE_TASKS}")
    if task_config.abs_action:
        raise ValueError("OopsieVerse actions are 12-dim deltas; set abs_action=false")
    parts = {c: _make_condition_dataset(task_config, c, mode) for c in task_config.conditions}
    return OopsieverseStateDataset(parts)


def make_val_datasets(task_config, normalizer: dict) -> dict[str, RobomimicDataset]:
    """Held-out demos per feedback condition, normalized with the training normalizer."""
    val = {}
    for condition in task_config.conditions:
        part = _make_condition_dataset(task_config, condition, "val")
        part.normalizer = normalizer
        val[condition] = part
    return val


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download OopsieVerse demos from HuggingFace")
    parser.add_argument("--task", type=str, default=None, choices=list(TASK_SCHEMA), help="Single task to download; omit to download all")
    parser.add_argument("--variant", type=str, default="all_data", choices=VARIANTS)
    parser.add_argument("--all-variants", action="store_true", help="Download all three variants instead of just --variant")
    parser.add_argument("--inspect", action="store_true", help="Print schema summary after downloading")
    args = parser.parse_args()

    tasks = [args.task] if args.task else list(TASK_SCHEMA)
    variants = list(VARIANTS) if args.all_variants else [args.variant]

    downloaded = download_all_oopsieverse(tasks=tasks, variants=variants)
    if args.inspect:
        for task_paths in downloaded.values():
            for p in task_paths.values():
                inspect_oopsieverse(p)
