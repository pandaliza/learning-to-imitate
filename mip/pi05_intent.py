"""Shared MIP flow-map intent generator for Pi0.5 slot-intent finetuning.

Loads a trained MIP `flow_intent` agent (mean or slot variant) and produces a
fixed-size intent vector from the *current* observation window via the intent
flow ODE -- i.e. the deployable p(z|s) generator, no future frames needed.

Used by:
  - examples/openpi/convert_libero_to_lerobot_intent.py  (offline, per demo frame)
  - the Pi0.5 eval sidecar                                (online, per env step)

See docs/pi05_slotintent_finetuning.md.
"""

import os

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from tensordict import TensorDict

from mip.datasets.libero_dataset import make_dataset
from mip.flow_intent_agent import FlowIntentAgent


class IntentGenerator:
    """Wraps a trained MIP flow-intent agent: obs window -> intent vector.

    Args:
        task_config_name: Hydra task config the checkpoint was trained with
            (e.g. "libero_goal_suite_image_flow_intent" for the 6D mean ckpt,
            "libero_goal_suite_image_slot_intent" for the 64D slot ckpt).
        ckpt_path: path to the MIP agent checkpoint (model_best.pt / model_latest.pt).
        config_dir: Hydra config dir (examples/configs).
        device: torch device.
        num_steps: intent ODE steps (-1 = config default).
    """

    def __init__(self, task_config_name, ckpt_path, config_dir, device="cuda", num_steps=-1):
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_dir)):
            cfg = compose(
                config_name="main",
                overrides=[f"task={task_config_name}", "network=mlp_flow_intent"],
            )
        cfg.optimization.device = device
        # Image mode: the encoder embedding dimension is the agent's obs_dim
        # (mirrors examples/train/train_libero.py main()).
        cfg.task.obs_dim = cfg.network.emb_dim
        self.cfg = cfg
        self.device = device
        self.num_steps = num_steps
        self.obs_steps = int(cfg.task.obs_steps)
        self.image_keys = list(cfg.task.image_obs_keys)
        self.intent_dim = int(cfg.task.intent_dim)

        # Rebuild the dataset normalizer (not stored in the checkpoint); same HDF5s
        # -> identical MinMax stats as training.
        self._normalizer = make_dataset(cfg.task).normalizer
        self.agent = FlowIntentAgent(cfg)
        self.agent.load(ckpt_path)

    @torch.no_grad()
    def intent_for_windows(self, state_windows: np.ndarray, image_windows: dict) -> np.ndarray:
        """Compute intent for a batch of obs windows.

        Args:
            state_windows: (B, To, state_dim) raw (un-normalized) state.
            image_windows: {img_key: (B, To, C, H, W) uint8} for each image key.
        Returns:
            (B, intent_dim) float32 intent vectors.
        """
        obs = {}
        st = self._normalizer["obs"]["state"].normalize(state_windows.astype(np.float32))
        obs["state"] = torch.as_tensor(st, device=self.device, dtype=torch.float32)
        for k in self.image_keys:
            im = image_windows[k].astype(np.float32) / 255.0
            obs[k] = torch.as_tensor(im, device=self.device, dtype=torch.float32)
        td = TensorDict(obs, batch_size=int(state_windows.shape[0]))
        intent = self.agent.sample_intent(td, use_ema=True, num_steps=self.num_steps)
        return intent.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def stack_windows(per_step: np.ndarray, obs_steps: int) -> np.ndarray:
        """Build (T, obs_steps, ...) sliding windows with left edge-padding.

        per_step: (T, ...) -> returns (T, obs_steps, ...) where window t is
        [t-obs_steps+1 .. t], clamped at 0 (repeat first frame).
        """
        T = per_step.shape[0]
        idx = np.clip(
            np.arange(T)[:, None] + np.arange(-obs_steps + 1, 1)[None, :], 0, T - 1
        )  # (T, obs_steps)
        return per_step[idx]
