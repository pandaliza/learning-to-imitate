"""Shared builders for the rollout audit (config, agents, dataset, envs).

Mirrors examples/train/train_robomimic.py main(): hydra compose of examples/configs,
obs_dim set from the env, dataset normalizers from make_dataset(config.task).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = "/scr/lizad/personal/learning-to-imitate"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

CKPT = {
    "action": f"{REPO}/checkpoints/square_mh_state_flow_mlp_512_h10_seed0_success62.pt",
    "intent": f"{REPO}/checkpoints/square_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success64.pt",
}
OVERRIDES = {
    "action": ["task=square_mh_state", "network=mlp"],
    "intent": ["task=square_mh_state_flow_intent", "network=mlp_flow_intent"],
}


def build_config(arm: str, device: str = "cuda", num_steps: int = 9, extra=()):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=f"{REPO}/examples/configs", version_base=None):
        cfg = compose(config_name="main", overrides=list(OVERRIDES[arm]) + [
            f"optimization.device={device}",
            f"optimization.num_steps={num_steps}",
            "task.save_video=false",
            "log.save_video=false",
            *extra,
        ])
    # train_robomimic main(): config.task.obs_dim = env obs dim (23 for square state);
    # Config A keeps base obs dim.
    cfg.task.obs_dim = 23
    return cfg


def load_agent(arm: str, cfg):
    from mip.agent import TrainingAgent
    from mip.flow_intent_agent import FlowIntentAgent
    if arm == "intent":
        agent = FlowIntentAgent(cfg)
    else:
        agent = TrainingAgent(cfg)
    agent.load(CKPT[arm], load_optimizer=False)  # strict load_state_dict inside
    agent.eval()
    return agent


def strict_load_report(arm: str, agent):
    """Re-verify strict loading explicitly (load_state_dict default strict=True)."""
    sd = torch.load(CKPT[arm], map_location="cpu", weights_only=False)
    pairs = (
        [("encoder", agent.encoder), ("encoder_ema", agent.encoder_ema),
         ("intent_flow_map", agent.intent_flow_map), ("intent_flow_map_ema", agent.intent_flow_map_ema),
         ("action_decoder", agent.action_decoder), ("action_decoder_ema", agent.action_decoder_ema)]
        if arm == "intent" else
        [("encoder", agent.encoder), ("encoder_ema", agent.encoder_ema),
         ("flow_map", agent.flow_map), ("flow_map_ema", agent.flow_map_ema)]
    )
    out = {}
    for k, m in pairs:
        r = m.load_state_dict(sd[k], strict=True)
        out[k] = {"missing": list(r.missing_keys), "unexpected": list(r.unexpected_keys)}
    ts = sd.get("training_state") or {}
    return out, ts


_DATASET_CACHE = {}


def get_dataset(cfg):
    from mip.datasets.robomimic_dataset import make_dataset
    key = (cfg.task.env_name, cfg.task.env_type, bool(getattr(cfg.task, "intent_conditioning", False)))
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = make_dataset(cfg.task)
    return _DATASET_CACHE[key]


def make_single_env(cfg, seed: int):
    """One MultiStepWrapper env whose first reset uses robosuite seed `seed` (idx=0)."""
    from mip.envs.robomimic.robomimic_env import make_env
    import io
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        env = make_env(cfg.task, 0, render=False, seed=seed)()
    finally:
        sys.stdout = old
    return env
