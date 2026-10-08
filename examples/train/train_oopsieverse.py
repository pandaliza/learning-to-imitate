"""Train flow policies on OopsieVerse RoboCasa demos with offline held-out evaluation.

There is no OopsieVerse simulator in this repo, so the env-rollout eval in
train_robomimic.py is replaced by action/intent error on held-out demos, scored
separately for each feedback condition.
"""

from collections import defaultdict
from functools import partial

import hydra
import loguru
import torch

from train_robomimic import train  # noqa: I001  (also sets the MuJoCo backend env vars)

from mip.agent import TrainingAgent
from mip.datasets.oopsieverse_dataset import make_dataset, make_val_datasets
from mip.intent_predictor import IntentPredictor
from mip.logger import Logger
from mip.torch_utils import limit_threads, set_seed

CONDITION_TAGS = {"without_live_feedback": "nofb", "with_live_feedback": "fb"}


def _cv_proxy(obs, intent_start, intent_end, intent_horizon):
    eef_now = obs[:, -1, intent_start:intent_end]
    eef_prev = obs[:, -2, intent_start:intent_end]
    return eef_now + (intent_horizon + 1) / 2.0 * (eef_now - eef_prev)


@torch.no_grad()
def offline_eval(config, envs, dataset, agent, logger, num_steps, intent_predictor=None, *, val_datasets):
    """Normalized MSE of the executed action chunk on held-out demos.

    Intent-conditioned policies are scored with three intent sources: the learned
    predictor, the constant-velocity proxy, and the ground-truth future (oracle upper bound).
    """
    device = config.optimization.device
    task = config.task
    start = task.obs_steps - 1
    end = start + task.act_steps
    action_norm = dataset.normalizer["action"]
    active = torch.as_tensor(action_norm.max > action_norm.min, device=device)

    metrics = {}
    totals = defaultdict(float)
    total_n = 0
    for condition, val_ds in val_datasets.items():
        sums = defaultdict(float)
        n = 0
        loader = torch.utils.data.DataLoader(val_ds, batch_size=1024, shuffle=False)
        for batch in loader:
            obs = batch["obs"]["state"][:, : task.obs_steps].to(device)
            act_gt = batch["action"][:, start:end].to(device)[..., active]
            bs = obs.shape[0]

            if task.get("intent_conditioning", False):
                intent_gt = batch["intent"].to(device)
                sources = {
                    "oracle": intent_gt,
                    "cv": _cv_proxy(obs, dataset.intent_start, dataset.intent_end, task.intent_horizon),
                }
                if intent_predictor is not None:
                    sources["learned"] = intent_predictor(obs)
                for name in ("cv", "learned"):
                    if name in sources:
                        sums[f"intent_mse_{name}"] += torch.mean((sources[name] - intent_gt) ** 2).item() * bs
            else:
                sources = {"none": None}

            for name, intent in sources.items():
                policy_obs = obs
                if intent is not None:
                    policy_obs = torch.cat([obs, intent.unsqueeze(1).expand(-1, task.obs_steps, -1)], dim=-1)
                act_0 = torch.randn((bs, task.horizon, task.act_dim), device=device)
                pred = agent.sample(act_0=act_0, obs=policy_obs, num_steps=num_steps, use_ema=True)
                err = (pred[:, start:end][..., active] - act_gt) ** 2
                sums[f"action_mse_{name}"] += err.mean().item() * bs
                sums[f"first_action_mse_{name}"] += err[:, 0].mean().item() * bs
            n += bs

        tag = CONDITION_TAGS[condition]
        for key, value in sums.items():
            metrics[f"val_{tag}_{key}_{num_steps}"] = value / n
            totals[key] += value
        total_n += n

    for key, value in totals.items():
        metrics[f"val_all_{key}_{num_steps}"] = value / total_n
    return metrics


@hydra.main(version_base=None, config_path="configs/", config_name="oopsieverse")
def main(config):
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    set_seed(config.optimization.seed)
    limit_threads(1)

    if config.network.get("arch_variant", "flow_action") != "flow_action":
        raise ValueError("train_oopsieverse supports arch_variant=flow_action only")
    if config.task.get("intent_type", "mean") not in ("mean", "final"):
        raise ValueError("train_oopsieverse supports intent_type mean or final only")

    logger = Logger(config)
    dataset = make_dataset(config.task)
    val_datasets = make_val_datasets(config.task, dataset.normalizer)
    loguru.logger.info(
        f"Train windows: {len(dataset)} | val windows: "
        + ", ".join(f"{c}={len(d)}" for c, d in val_datasets.items())
    )

    base_obs_dim = dataset[0]["obs"]["state"].shape[-1]
    config.task.obs_dim = base_obs_dim
    intent_predictor = None
    if config.task.get("intent_conditioning", False):
        config.task.obs_dim = base_obs_dim + config.task.intent_dim
        if config.task.get("intent_predictor", False):
            intent_predictor = IntentPredictor(
                obs_steps=config.task.obs_steps,
                base_obs_dim=base_obs_dim,
                intent_dim=config.task.intent_dim,
            ).to(config.optimization.device)
    loguru.logger.info(f"obs_dim={config.task.obs_dim} (base {base_obs_dim}), act_dim={config.task.act_dim}")

    agent = TrainingAgent(config)
    train(
        config,
        envs=None,
        dataset=dataset,
        agent=agent,
        logger=logger,
        intent_predictor=intent_predictor,
        eval_fn=partial(offline_eval, val_datasets=val_datasets),
    )
    logger.save_agent(agent=agent, identifier="final")


if __name__ == "__main__":
    main()
