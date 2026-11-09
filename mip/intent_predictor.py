"""Learned intent predictor for intent-conditioned flow policies.

Predicts the mean future end-effector pose (mean over next act_steps) from the
current normalized observation window.  Replaces the constant-velocity proxy at
inference time, eliminating the train/inference distribution mismatch.

Input  : (B, obs_steps * base_obs_dim)  — flattened, normalized obs (no intent)
Output : (B, intent_dim)                — predicted mean future eef (normalized)
Training: MSE against ground-truth intent from the dataset
"""

import torch
import torch.nn as nn


class IntentPredictor(nn.Module):
    """Two-layer MLP that predicts mean future eef from the current obs window."""

    def __init__(self, obs_steps: int, base_obs_dim: int, intent_dim: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = obs_steps * base_obs_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, intent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_steps, base_obs_dim) — normalized obs window (no intent appended)
        Returns:
            intent: (B, intent_dim)
        """
        B = obs.shape[0]
        return self.net(obs.reshape(B, -1))


class IntentFlowPredictor(nn.Module):
    """Rectified-flow intent generator (the I-PATH recipe, examples/oopsieverse_full/matrix/models.py).

    Training: z_t = (1 - t) z0 + t z1 with z0 ~ N(0, I), z1 = normalized GT intent, t ~ Beta(1.5, 1);
    loss = MSE(v(obs, z_t, t), z1 - z0). Training draws use a dedicated generator so they do not consume
    the global RNG streams. Inference (forward): Euler from N(0, I) in `nfe` steps, so the intent is a
    sample, not a mean. Unlike OopsieVerse's shared transformer, MIP's state MLP has no shared trunk, so
    this net sees the normalized obs window directly.
    """

    def __init__(self, obs_steps: int, base_obs_dim: int, intent_dim: int, hidden_dim: int = 512,
                 n_layers: int = 3, nfe: int = 16, t_emb_dim: int = 64, seed: int = 0):
        super().__init__()
        self.intent_dim = intent_dim
        self.nfe = nfe
        self.t_emb_dim = t_emb_dim
        self.seed = seed
        self._gen = None
        layers, d = [], obs_steps * base_obs_dim + intent_dim + t_emb_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
            d = hidden_dim
        layers.append(nn.Linear(d, intent_dim))
        self.net = nn.Sequential(*layers)

    def _t_emb(self, t):
        half = self.t_emb_dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half, device=t.device) / half)
        ang = 1000.0 * t[:, None] * freqs[None]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def velocity(self, obs_flat, z, t):
        return self.net(torch.cat([obs_flat, z, self._t_emb(t)], dim=-1))

    def _generator(self, device):
        if self._gen is None or self._gen.device != torch.device(device):
            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(self.seed + 7919)
        return self._gen

    def flow_loss(self, obs: torch.Tensor, intent_gt: torch.Tensor) -> torch.Tensor:
        B, dev = obs.shape[0], obs.device
        g = self._generator(dev)
        t = torch.rand(B, device=dev, generator=g) ** (1.0 / 1.5)  # Beta(1.5, 1) by inverse CDF
        z0 = torch.randn(intent_gt.shape, device=dev, generator=g)
        zt = (1 - t[:, None]) * z0 + t[:, None] * intent_gt
        return torch.mean((self.velocity(obs.reshape(B, -1), zt, t) - (intent_gt - z0)) ** 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample one intent per obs window: (B, obs_steps, base_obs_dim) -> (B, intent_dim)."""
        B = obs.shape[0]
        obs_flat = obs.reshape(B, -1)
        z = torch.randn(B, self.intent_dim, device=obs.device)
        dt = 1.0 / self.nfe
        for i in range(self.nfe):
            t = torch.full((B,), i * dt, device=obs.device)
            z = z + self.velocity(obs_flat, z, t) * dt
        return z
