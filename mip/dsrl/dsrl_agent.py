"""DSRL-SAC agent: learns initial noise x_0 in Gaussian space to steer a frozen flow policy.

Reference: "Steering Your Diffusion Policy with Latent Space Reinforcement Learning" (arXiv 2506.15799)

For flow matching: x_0 ~ N(0,1) is the ODE starting point. The actor outputs x_0 given obs,
replacing random noise with a learned noise that steers the flow toward high-reward actions.

Composition:  x_0 = actor(obs)
              a_exec = flow_policy.sample(act_0=x_0, obs=obs, sample_mode="stochastic")

The frozen flow policy is accessed externally (set via set_flow_agent).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mip.residual_sac.networks import TanhGaussianPolicy, QEnsemble


@dataclass
class DSRLConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    # Wide MLP recommended for high-dim noise space (Ta*act_dim can be 160+)
    hidden_dims: tuple[int, ...] = field(default_factory=lambda: (1024, 1024, 1024))
    discount: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.1
    target_entropy: float | None = None  # None = auto: -noise_dim * 0.5
    # Noise magnitude: x_0 ~ N(0,1) so values typically in [-3, 3].
    # TanhGaussian squashes to [-noise_mag, noise_mag].
    noise_mag: float = 2.0
    backup_entropy: bool = True
    max_grad_norm: float = 10.0
    weight_decay: float = 1e-4
    use_layer_norm: bool = False
    # Clamp Q targets to prevent bootstrap explosion.
    # Set to reward_scale * max_episode_reward / (1 - discount).
    # For reward_scale=0.1 and max_ep_reward=50: 0.1*50/0.01=500 (loose).
    # In practice set ~5-10x expected max return for the task.
    q_target_clip: float = float("inf")
    # Actor family:
    #   "tanh"       — TanhGaussianPolicy squashed to [-noise_mag, noise_mag] (legacy SAC).
    #   "mean_shift" — z = mu(o) + eps, eps ~ N(0,I); fixed unit variance, no tanh,
    #                  no learned alpha; regularized by 0.5*||mu||^2/d (KL to N(0,I)).
    #                  At init mu=0, so the steered distribution IS the BC prior.
    actor_type: str = "tanh"
    # Weight on the per-dimension KL-to-prior penalty (mean_shift actor only).
    kl_beta: float = 0.1
    # Warmup exploration distribution in train_dsrl.py:
    #   "uniform"  — U[-noise_mag, noise_mag] (legacy; mismatched with the flow prior).
    #   "gaussian" — N(0, I), matching what the frozen flow policy was trained on.
    warmup_dist: str = "uniform"
    device: str = "cuda"


def _safe_optimizer_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    parameters,
    max_grad_norm: float = 0.0,
) -> bool:
    """Transactionally apply one optimizer step.

    Skips the update entirely (no backward, no optimizer-state mutation) on a
    non-finite loss, and again after backward if any gradient is non-finite —
    a finite scalar loss can still produce non-finite gradients through
    unstable derivatives. Raises if parameters come out non-finite anyway,
    because silently continuing after optimizer-level corruption (e.g. via
    Adam's running moments) poisons Polyak-averaged targets forever.

    Returns True iff the step was applied.
    """
    params = [p for p in parameters]

    if not torch.isfinite(loss).all():
        optimizer.zero_grad(set_to_none=True)
        return False

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    for p in params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            optimizer.zero_grad(set_to_none=True)
            return False

    if max_grad_norm > 0:
        nn.utils.clip_grad_norm_(params, max_grad_norm)
    optimizer.step()

    for p in params:
        if not torch.isfinite(p).all():
            raise FloatingPointError(
                "Optimizer produced non-finite parameters despite finite loss and gradients."
            )
    return True


class MeanShiftPolicy(nn.Module):
    """Prior-regularized latent steering actor: z = mu(o) + eps, eps ~ N(0, I).

    Unit variance is fixed to the frozen flow policy's training prior N(0, I);
    the actor only learns a state-conditioned translation of that prior. The
    mean head is zero-initialized, so at init the steered distribution exactly
    equals the BC prior (first eval should reproduce base-policy SR).

    KL(N(mu, I) || N(0, I)) = 0.5 * ||mu||^2, an L2 penalty on the shift.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim_flat: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        use_layer_norm: bool = False,
    ):
        super().__init__()
        from mip.residual_sac.networks import _build_mlp

        self.act_dim_flat = act_dim_flat
        self.trunk = _build_mlp(obs_dim, hidden_dims[-1], hidden_dims[:-1], use_layer_norm=use_layer_norm)
        self.mean_head = nn.Linear(hidden_dims[-1], act_dim_flat)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

    def mu(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_head(self.trunk(obs))

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample with log-probability under N(mu, I).

        Returns:
            z:        (B, act_dim_flat)
            log_prob: (B,)
        """
        mu = self.mu(obs)
        eps = torch.randn_like(mu)
        z = mu + eps
        log_prob = (-0.5 * (eps.square() + _LOG_2PI)).sum(dim=-1)
        return z, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """The latent mode (z = mu). NOTE: for this actor family the policy's
        variance is part of the match to the BC prior — evaluation should
        normally use sample(), not the mode (see DSRLSACAgent.sample_noise)."""
        return self.mu(obs)


_LOG_2PI = float(np.log(2.0 * np.pi))


class DSRLSACAgent:
    """DSRL-SAC: SAC in the noise space of a frozen flow policy.

    Args:
        config:    DSRLConfig hyperparameters.
        obs_dim:   Flattened obs dimension (obs_steps * per_step_obs_dim).
        Ta:        Action horizon of the flow policy.
        act_dim:   Per-step action dimension.
    """

    def __init__(
        self,
        config: DSRLConfig,
        obs_dim: int,
        Ta: int,
        act_dim: int,
    ):
        self.config = config
        self.obs_dim = obs_dim
        self.Ta = Ta
        self.act_dim = act_dim
        self.noise_dim = Ta * act_dim
        device = config.device

        if config.actor_type == "mean_shift":
            self.actor = MeanShiftPolicy(
                obs_dim=obs_dim,
                act_dim_flat=self.noise_dim,
                hidden_dims=config.hidden_dims,
                use_layer_norm=config.use_layer_norm,
            ).to(device)
        elif config.actor_type == "tanh":
            self.actor = TanhGaussianPolicy(
                obs_dim=obs_dim,
                act_dim_flat=self.noise_dim,
                hidden_dims=config.hidden_dims,
                action_mag=config.noise_mag,
                use_layer_norm=config.use_layer_norm,
            ).to(device)
        else:
            raise ValueError(f"Unknown actor_type: {config.actor_type!r}")

        self.critic = QEnsemble(
            obs_dim=obs_dim,
            act_dim_flat=self.noise_dim,
            hidden_dims=config.hidden_dims,
            use_layer_norm=config.use_layer_norm,
        ).to(device)

        self.critic_target = deepcopy(self.critic).requires_grad_(False)

        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(config.init_alpha), dtype=torch.float32, device=device)
        )
        self.target_entropy = (
            config.target_entropy
            if config.target_entropy is not None
            else -self.noise_dim * 0.5
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr, weight_decay=config.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr, weight_decay=config.weight_decay)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

        self.flow_agent = None

    def set_flow_agent(self, flow_agent) -> None:
        self.flow_agent = flow_agent

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def sample_noise(
        self,
        obs_flat: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Sample initial noise x_0 given flattened obs.

        Returns:
            noise: (noise_dim,) numpy — x_0 for the flow ODE.
            log_prob: scalar.
        """
        device = self.config.device
        obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
        if deterministic and self.config.actor_type != "mean_shift":
            noise = self.actor.deterministic(obs_t)
            log_prob = 0.0
        else:
            # mean_shift ignores `deterministic`: its unit variance is part of
            # the match to the flow policy's N(0,I) prior — evaluating at the
            # mode (z=mu, which is z=0 at init) would evaluate a single latent,
            # not the steered distribution. Sampling keeps the invariant that
            # at init eval reproduces the base BC policy's success rate.
            noise, lp = self.actor.sample(obs_t)
            log_prob = lp.item()
        return noise.squeeze(0).cpu().numpy(), log_prob

    def update_critic(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, float], bool]:
        obs = batch["obs"]
        noise = batch["noise"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            next_noise, next_log_prob = self.actor.sample(next_obs)
            next_q = self.critic_target.min_q(next_obs, next_noise).squeeze(-1)
            if self.config.backup_entropy and self.config.actor_type == "tanh":
                next_q = next_q - self.alpha.detach() * next_log_prob
            target_q = reward + (1.0 - done) * self.config.discount * next_q
            if self.config.q_target_clip < float("inf"):
                target_q = target_q.clamp(-self.config.q_target_clip, self.config.q_target_clip)

        qs = self.critic(obs, noise)  # (num_qs, B, 1)
        critic_loss = sum(
            F.mse_loss(qs[i].squeeze(-1), target_q) for i in range(self.critic.num_qs)
        )

        info = {
            "critic_loss": critic_loss.item(),
            "q_mean": qs.mean().item(),
            "target_q_mean": target_q.mean().item(),
        }

        updated = _safe_optimizer_step(
            critic_loss, self.critic_optimizer, self.critic.parameters(), self.config.max_grad_norm
        )
        info["critic_updated"] = float(updated)
        return info, updated

    def update_actor(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        noise, log_prob = self.actor.sample(obs)
        q = self.critic.min_q(obs, noise).squeeze(-1)

        if self.config.actor_type == "mean_shift":
            # Prior-regularized steering: maximize Q subject to a per-dimension
            # KL(N(mu,I) || N(0,I)) = 0.5*||mu||^2 penalty. No learned alpha —
            # the KL already carries the relevant scale/entropy terms, and a
            # generic SAC target entropy would compete with matching the prior.
            mu = self.actor.mu(obs)
            kl_per_dim = 0.5 * mu.square().mean(dim=-1)
            actor_loss = (-q + self.config.kl_beta * kl_per_dim).mean()
            info = {
                "actor_loss": actor_loss.item(),
                "kl_prior_per_dim": kl_per_dim.mean().item(),
                "mu_l2_per_dim": mu.square().mean().item(),
            }
        else:
            actor_loss = (self.alpha.detach() * log_prob - q).mean()
            info = {
                "actor_loss": actor_loss.item(),
                "alpha": self.alpha.item(),
                "entropy": -log_prob.mean().item(),
            }

        updated = _safe_optimizer_step(
            actor_loss, self.actor_optimizer, self.actor.parameters(), self.config.max_grad_norm
        )
        info["actor_updated"] = float(updated)

        if self.config.actor_type == "tanh":
            # Temperature update only for the SAC-entropy actor; gated on the
            # actor step so alpha never adapts to an update that didn't happen.
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            info["alpha_loss"] = alpha_loss.item()
            if updated:
                _safe_optimizer_step(alpha_loss, self.alpha_optimizer, [self.log_alpha])

        # OOD diagnostics: how does the critic score the actor's latents vs the
        # true prior's? A growing gap while eval SR stagnates is the signature
        # of the actor exploiting critic error off-distribution.
        with torch.no_grad():
            z_prior = torch.randn_like(noise)
            q_prior = self.critic.min_q(obs, z_prior).squeeze(-1)
            info.update({
                "noise_norm": noise.norm(dim=-1).mean().item(),
                "noise_abs_max": noise.abs().amax(dim=-1).mean().item(),
                "q_actor": q.mean().item(),
                "q_prior": q_prior.mean().item(),
                "q_actor_prior_gap": (q - q_prior).mean().item(),
            })
        return info

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        tau = self.config.tau
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            if not torch.isfinite(p).all():
                raise FloatingPointError("Live critic is non-finite; refusing target update.")
            tp.data.lerp_(p.data, tau)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        info, critic_updated = self.update_critic(batch)
        info.update(self.update_actor(batch))
        # Polyak-average only from a critic that actually completed a finite
        # update this step; a skipped step must not leak into the target.
        if critic_updated:
            self.soft_update_targets()
        return info

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.data,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str, load_optimizers: bool = False) -> None:
        sd = torch.load(path, map_location=self.config.device, weights_only=False)
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.log_alpha.data.copy_(sd["log_alpha"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(sd["actor_optimizer"])
            self.critic_optimizer.load_state_dict(sd["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(sd["alpha_optimizer"])

    def train(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()
