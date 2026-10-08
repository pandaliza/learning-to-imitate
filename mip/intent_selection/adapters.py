"""Policy adapters for intent selection experiment.

Implements native policy sampling without any modifications, steering, or temperature
scaling. Supports both ACTION (flow-based action) and INTENT (flow-based intent + MLP decoder)
arms with reproducible per-candidate noise and deterministic sampling.

Validated against real rollouts by examples/intent_selection/audit_rollouts.py (A1 parity).

Fixes applied during the rollout audit (2026-10-05):
  * TrainingAgent / FlowIntentAgent are not nn.Modules: the old ``agent.to(device)`` and
    ``agent.parameters()`` calls raised AttributeError. Freezing now iterates the agent's
    modules; the device comes from ``config.optimization.device`` (load maps there).
  * ACTION sample mode is now an explicit argument (``action_sample_mode``). The checkpoint's
    native mode is "zero" (initial noise ignored, deterministic); "stochastic" is the labeled
    non-native Gaussian-initial-noise arm.
  * "Physical" chunks were only min-max unnormalized 10-D (pos, rot6d, gripper) actions. The
    environment consumes 7-D (pos, axis-angle, gripper) obtained via
    ``dataset.undo_transform_action`` (native eval, train_robomimic.py). The adapter now
    returns both: ``physical_action_chunk`` (10-D) and ``env_action_chunk`` (7-D).
  * Missing normalizer silently returned normalized actions as "physical"; now it raises.
  * Keyed noise is drawn from a CPU torch.Generator and moved to the device, so candidate
    noise is identical across CPU/GPU (CUDA generators produce a different stream).
  * Explicit noise injection (``initial_noise=``) and a vectorized batch path for audits.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from mip.agent import TrainingAgent
from mip.config import Config
from mip.flow_intent_agent import FlowIntentAgent
from mip.torch_utils import at_least_ndim


@dataclass
class Candidate:
    """Sampled action candidate with full provenance and reproducibility info."""

    # Sampling inputs and noise
    initial_noise: torch.Tensor  # ACTION: (Ta, act_dim); INTENT: (1, intent_dim)
    intent_decoded: Optional[torch.Tensor] = None  # (intent_dim,) normalized-obs space (INTENT arm)
    decoder_noise_seed: Optional[int] = None  # MLP decoder is deterministic -> unused

    # Model outputs
    normalized_action_full: torch.Tensor = None  # (horizon, act_dim) normalized model output
    normalized_action_chunk: torch.Tensor = None  # (act_steps, act_dim) normalized exec slice
    physical_action_chunk: torch.Tensor = None  # (act_steps, 10) unnormalized pos/rot6d/grip
    env_action_chunk: Optional[np.ndarray] = None  # (act_steps, 7) pos/axis-angle/grip, sent to env

    # Reproducibility
    noise_seed: int = None
    sampling_latency_ms: float = None

    # Hashes for validation
    initial_noise_hash: str = None
    intent_hash: str = None
    action_hash: str = None


def _stable_hash_tensor(t: torch.Tensor) -> str:
    """Compute stable blake2b hash of a tensor."""
    data = t.detach().cpu().numpy().tobytes()
    h = hashlib.blake2b(data, digest_size=16)
    return h.hexdigest()


def _stable_hash_float(val: float) -> str:
    data = np.float32(val).tobytes()
    h = hashlib.blake2b(data, digest_size=16)
    return h.hexdigest()


def keyed_normal(shape, seed: int, device) -> torch.Tensor:
    """Standard normal noise from a CPU generator keyed by `seed` (device independent)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device)


class PolicyAdapter:
    """Loads a pretrained policy and provides native sampling interface."""

    def __init__(
        self,
        checkpoint_path: str,
        config: Config,
        arm_name: str = "action",
        device: Optional[str] = None,
        use_ema: bool = True,
        action_normalizer=None,
        undo_transform_action: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        action_sample_mode: Optional[str] = None,
        num_steps: int = 9,
    ):
        """Initialize adapter for either ACTION or INTENT arm.

        Args:
            checkpoint_path: Path to checkpoint
            config: Full configuration object (same one used to build the agent in training)
            arm_name: "action" or "intent"
            device: must equal config.optimization.device (agent weights are loaded there)
            use_ema: Whether to use EMA weights (native eval uses EMA)
            action_normalizer: dataset.normalizer["action"] (MinMaxNormalizer); required
            undo_transform_action: dataset.undo_transform_action (rot6d -> axis-angle)
            action_sample_mode: ACTION arm only: "zero" (native) or "stochastic" (Gaussian)
            num_steps: Euler steps (native eval NFE = 9)
        """
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.arm_name = arm_name
        self.device = device or config.optimization.device
        if str(self.device) != str(config.optimization.device):
            raise ValueError(
                f"device {self.device} != config.optimization.device {config.optimization.device}"
            )
        self.use_ema = use_ema
        if action_normalizer is None:
            raise ValueError("action_normalizer (dataset.normalizer['action']) is required")
        self.action_normalizer = action_normalizer
        self.undo_transform_action = undo_transform_action
        self.num_steps = int(num_steps)
        self.action_sample_mode = action_sample_mode or config.optimization.sample_mode
        if self.action_sample_mode not in ("zero", "stochastic"):
            raise ValueError(f"unsupported action_sample_mode {self.action_sample_mode}")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        is_flow_intent = "intent_flow_map" in ckpt
        del ckpt
        if (arm_name == "intent") != is_flow_intent:
            raise ValueError(f"arm_name={arm_name} does not match checkpoint type")

        if is_flow_intent:
            self.agent = FlowIntentAgent(config)
            self.is_intent_arm = True
        else:
            self.agent = TrainingAgent(config)
            self.is_intent_arm = False

        # strict load_state_dict inside both agents' load()
        self.agent.load(checkpoint_path, load_optimizer=False)
        self.agent.eval()

        if self.is_intent_arm:
            self.encoder = self.agent.encoder_ema if use_ema else self.agent.encoder
            self.intent_flow_map = (
                self.agent.intent_flow_map_ema if use_ema else self.agent.intent_flow_map
            )
            self.action_decoder = (
                self.agent.action_decoder_ema if use_ema else self.agent.action_decoder
            )
            self._modules = [self.encoder, self.intent_flow_map, self.action_decoder]
        else:
            self.encoder = self.agent.encoder_ema if use_ema else self.agent.encoder
            self.flow_map = self.agent.flow_map_ema if use_ema else self.agent.flow_map
            self._modules = [self.encoder, self.flow_map]
        for m in self._modules:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------ noise
    def noise_shape(self):
        if self.is_intent_arm:
            return (1, self.config.task.intent_dim)
        return (self.config.task.horizon, self.config.task.act_dim)

    def make_noise(self, noise_seed: int) -> torch.Tensor:
        return keyed_normal(self.noise_shape(), noise_seed, self.device)

    # --------------------------------------------------------------- sampling
    def sample_candidate(
        self,
        obs_history: torch.Tensor,
        noise_seed: Optional[int] = None,
        decoder_seed: Optional[int] = None,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> Candidate:
        """Sample one candidate.

        Args:
            obs_history: (obs_steps, obs_dim) normalized observation history
            noise_seed: key for the initial noise (ignored if initial_noise is given)
            decoder_seed: unused (MLP decoder is deterministic); recorded only
            initial_noise: explicit initial noise of shape noise_shape()
        """
        if initial_noise is None:
            if noise_seed is None:
                raise ValueError("need noise_seed or initial_noise")
            initial_noise = self.make_noise(noise_seed)
        cands = self._sample(obs_history.unsqueeze(0), initial_noise.unsqueeze(0).to(self.device))
        c = cands[0]
        c.noise_seed = noise_seed
        c.decoder_noise_seed = decoder_seed
        return c

    def sample_batch(
        self,
        obs_history: torch.Tensor,
        noise_seeds: list[int],
        decoder_seeds: Optional[list[int]] = None,
        vectorized: bool = False,
    ) -> list[Candidate]:
        """Sample candidates at one state with per-candidate keyed noise.

        vectorized=False: one forward pass per candidate.
        vectorized=True : single batched pass (obs repeated); must match the loop.
        """
        if not vectorized:
            return [
                self.sample_candidate(
                    obs_history, s, decoder_seeds[i] if decoder_seeds else None
                )
                for i, s in enumerate(noise_seeds)
            ]
        noise = torch.stack([self.make_noise(s) for s in noise_seeds], dim=0)
        obs = obs_history.unsqueeze(0).expand(len(noise_seeds), *obs_history.shape).contiguous()
        cands = self._sample(obs, noise)
        for i, c in enumerate(cands):
            c.noise_seed = noise_seeds[i]
            c.decoder_noise_seed = decoder_seeds[i] if decoder_seeds else None
        return cands

    def decode_given_intent(self, obs_history: torch.Tensor, intents: torch.Tensor) -> torch.Tensor:
        """INTENT arm: decoder output (B, horizon, act_dim) for explicit intents (B, intent_dim)."""
        assert self.is_intent_arm
        B = intents.shape[0]
        obs = obs_history.unsqueeze(0).expand(B, *obs_history.shape).contiguous().to(self.device)
        with torch.no_grad():
            obs_emb = self.encoder(obs, None)
            return self.action_decoder(obs_emb, intents.to(self.device))

    def _sample(self, obs_in: torch.Tensor, noise: torch.Tensor) -> list[Candidate]:
        t0 = time.perf_counter()
        obs_in = obs_in.to(self.device)
        B = obs_in.shape[0]
        intent_vec = None
        with torch.no_grad():
            obs_emb = self.encoder(obs_in, None)  # (B, obs_steps, emb_dim)
            t_schedule = np.linspace(0, 1, self.num_steps + 1)
            if self.is_intent_arm:
                x = noise  # (B, 1, intent_dim): intent flow always starts from Gaussian noise
                fm = self.intent_flow_map
            else:
                x = noise if self.action_sample_mode == "stochastic" else torch.zeros_like(noise)
                fm = self.flow_map
            for i in range(self.num_steps):
                s = torch.full((B,), t_schedule[i], device=self.device)
                t = torch.full((B,), t_schedule[i + 1], device=self.device)
                b_s = fm.get_velocity(s, x, obs_emb)
                x = x + b_s * (at_least_ndim(t, x.dim()) - at_least_ndim(s, x.dim()))
            if self.is_intent_arm:
                intent_vec = x.squeeze(1)  # (B, intent_dim)
                action_full = self.action_decoder(obs_emb, intent_vec)  # (B, horizon, act_dim)
            else:
                action_full = x
        latency = (time.perf_counter() - t0) * 1000.0 / B

        s0 = self.config.task.obs_steps - 1
        a_steps = self.config.task.act_steps
        full_np = action_full.detach().cpu().numpy()
        phys_full = self.action_normalizer.unnormalize(full_np)  # native: unnormalize, then slice
        phys = phys_full[:, s0 : s0 + a_steps]
        env_act = self.undo_transform_action(phys) if self.undo_transform_action else None

        out = []
        for b in range(B):
            norm_chunk = action_full[b, s0 : s0 + a_steps].clone()
            phys_t = torch.from_numpy(phys[b].copy())
            iv = intent_vec[b].clone() if intent_vec is not None else None
            out.append(
                Candidate(
                    initial_noise=noise[b].clone(),
                    intent_decoded=iv,
                    normalized_action_full=action_full[b].clone(),
                    normalized_action_chunk=norm_chunk,
                    physical_action_chunk=phys_t,
                    env_action_chunk=None if env_act is None else env_act[b],
                    sampling_latency_ms=latency,
                    initial_noise_hash=_stable_hash_tensor(noise[b]),
                    intent_hash=_stable_hash_tensor(iv) if iv is not None else None,
                    action_hash=_stable_hash_tensor(phys_t),
                )
            )
        return out

    def _denormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        """Min-max unnormalize (normalized [-1,1] -> 10-D pos/rot6d/gripper)."""
        action_np = action_norm.detach().cpu().numpy()
        action_denorm = self.action_normalizer.unnormalize(action_np)
        return torch.from_numpy(action_denorm).to(action_norm.device).to(action_norm.dtype)

    def parameters(self):
        """Return inference-module parameters (for freezing check)."""
        return [p for m in self._modules for p in m.parameters()]


class DeterministicNoiseSampler:
    """Keyed, stateless noise sampler using blake2b + torch.Generator."""

    def __init__(self, base_seed: int = 0):
        self.base_seed = base_seed

    def get_noise_seed(
        self,
        experiment_id: str,
        arm: str,
        checkpoint_pair_id: str,
        state_id: int,
        candidate_id: int,
    ) -> int:
        key_str = f"{experiment_id}:{arm}:{checkpoint_pair_id}:{state_id}:{candidate_id}"
        h = hashlib.blake2b(key_str.encode(), digest_size=8)
        return int.from_bytes(h.digest(), byteorder="big") % (2**31 - 1)


__all__ = [
    "Candidate",
    "PolicyAdapter",
    "DeterministicNoiseSampler",
    "keyed_normal",
    "_stable_hash_tensor",
]
