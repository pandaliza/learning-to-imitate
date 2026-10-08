"""Intent-conditioned and auxiliary-prediction variants of the official CFMPolicy (a739f99), as subclasses.

Backbone (unchanged, shared): SegmentationEncoder + StateEncoder -> 7 obs tokens, the SAME 4-layer post-LN
transformer (d=512, 8 heads), final LayerNorm, action_proj / timestep_proj / action_head, sinusoidal positions.

Intent flow (all intent arms + AUX-PATH-G; one declared recipe):
  - The future-motion target is a sequence of n_tok tokens of width tok_dim (1x7 I-MEAN, 4x9 I-PATH, 4x10 I-PATH-G,
    1x9 canonical mean; AUX uses 4x10). Normalised to [-1, 1] on mixed-train.
  - Rectified-flow on intent tokens: z_t = (1-t) z0 + t z1, z0 ~ N(0, I), t ~ Beta(1.5, 1) (same as the action
    flow), velocity MSE. Network = the SAME transformer over [obs(7), noisy intent tokens(n_tok)]; noisy intent
    token = intent_in_proj(z_t) + timestep_proj(emb(t)); obs tokens cannot attend to intent tokens (as obs cannot
    attend to actions natively). Output via intent_head (Linear). Inference: Euler, INTENT_NFE = 16 steps from N(0,I).
Action flow conditioning (I-MEAN, I-PATH, I-PATH-G, canonical mean; ONE mechanism for all):
  - Extra tokens: each clean intent token -> intent_cond_proj (Linear-SiLU-Linear, same form as action_proj) is
    inserted between the obs tokens and the noisy action tokens: [obs(7), intent(n_tok), action(8)].
    Mask: obs tokens attend only to obs; intent tokens attend to obs + intent; action tokens attend to all.
  - Training: teacher forcing. The action flow is conditioned on the ground-truth (normalised, detached) future
    intent; the intent flow is trained on the same target. Loss = L_action + 1.0 * L_intent, one AdamW, one
    gradient clip (native settings). No predicted-intent mixing (the established recipe's default,
    decoder_uses_sampled_intent=False, mip/flow_intent_agent.py).
  - Inference: intent ~ intent flow (16 Euler steps from N(0,I)), then action ~ action flow (16 Euler steps from
    N(0,I)) conditioned on the sampled intent.
AUX-PATH-G: identical intent flow on the I-PATH-G target (co-prediction, shared encoder + transformer), but the
action pass is EXACTLY the native one ([obs(7), action(8)], native positions/mask). No conditioning.

RNG discipline (data-order pairing with A-MIX): the action flow consumes the global RNGs exactly as the native
CFMPolicy.compute_loss (CPU Beta for t, CUDA randn for x0); every intent-flow draw (t, z0) uses a dedicated CUDA
torch.Generator, so the CPU global stream, which also seeds the DataLoader shuffles, is identical to A-MIX.
"""

import os
import sys

import torch
import torch.nn as nn

OOPSIE = "/edrive1/lizad/oopsieverse_full_study/src/oopsieverse"
if OOPSIE not in sys.path:
    sys.path.insert(0, OOPSIE)

from learning.models import CFMPolicy  # noqa: E402
from learning.models.cfm_policy import PositionalEncoding, timestep_embedding  # noqa: E402
from learning.utils import denormalize_action  # noqa: E402

INTENT_NFE = 16
INTENT_LOSS_WEIGHT = 1.0
STATE_DIM = 23


def _mlp(i, d):
    return nn.Sequential(nn.Linear(i, d), nn.SiLU(), nn.Linear(d, d))


class IntentCFMPolicy(CFMPolicy):
    def __init__(self, config, n_tok, tok_dim, mode, intent_seed=0):
        assert mode in ("condition", "aux")
        self._intent_spec = (int(n_tok), int(tok_dim), mode, int(intent_seed))
        super().__init__(config)
        self.intent_gen = None
        self.intent_seed = int(intent_seed)
        self.reset_component_log()

    # ── build ──
    def _build_model(self):
        super()._build_model()  # identical module creation order / RNG draws as the native policy
        n_tok, tok_dim, mode, seed = self._intent_spec
        cfg = self.config
        self.n_tok, self.tok_dim, self.mode = n_tok, tok_dim, mode
        # extra modules: initialised under a forked CPU RNG (seeded from the run seed) so the global CPU stream,
        # which also drives the DataLoader shuffle and the action-flow time draws, stays identical to A-MIX.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1_000_003 + seed)
            self.intent_in_proj = _mlp(tok_dim, cfg.feature_dim)
            self.intent_head = nn.Linear(cfg.feature_dim, tok_dim)
            if mode == "condition":
                self.intent_cond_proj = _mlp(tok_dim, cfg.feature_dim)
        # longer sinusoidal table (identical first rows); buffer only, no parameters
        self.pos_encoding = PositionalEncoding(cfg.feature_dim, self.num_obs_tokens + n_tok + cfg.action_chunk_size)

    def _gen(self, device):
        if self.intent_gen is None or self.intent_gen.device != torch.device(device):
            self.intent_gen = torch.Generator(device=device)
            self.intent_gen.manual_seed(self.intent_seed + 7919)
        return self.intent_gen

    # ── pieces ──
    def split_state(self, state):
        if state.shape[-1] == STATE_DIM:
            return state, None
        it = state[:, STATE_DIM:].reshape(state.shape[0], self.n_tok, self.tok_dim)
        return state[:, :STATE_DIM], it

    def obs_tokens(self, seg_images, state):
        return torch.cat([self.encode_segmentation(seg_images), self.encode_state(state)], dim=1)

    def _transformer(self, sequence, mask):
        B = sequence.shape[0]
        sequence = sequence + self.pos_encoding(sequence.shape[1], B, sequence.device)
        for block in self.transformer_blocks:
            sequence = block(sequence, attn_mask=mask)["output"]
        return self.final_norm(sequence)

    def _mask(self, n_obs, n_mid, n_tail, device):
        L = n_obs + n_mid + n_tail
        m = torch.zeros(L, L, device=device)
        m[:n_obs, n_obs:] = -torch.inf
        if n_tail:
            m[n_obs:n_obs + n_mid, n_obs + n_mid:] = -torch.inf
        return m

    def intent_velocity(self, obs_tok, z_t, t):
        tok = self.intent_in_proj(z_t) + self.encode_timestep(t)
        seq = torch.cat([obs_tok, tok], dim=1)
        out = self._transformer(seq, self._mask(obs_tok.shape[1], self.n_tok, 0, seq.device))
        return self.intent_head(out[:, -self.n_tok:])

    def action_velocity(self, obs_tok, noisy_action, t, intent=None):
        a_tok = self.encode_action(noisy_action)
        a_tok = a_tok + self.encode_timestep(t)
        if self.mode == "condition":
            assert intent is not None
            c_tok = self.intent_cond_proj(intent)
            seq = torch.cat([obs_tok, c_tok, a_tok], dim=1)
            mask = self._mask(obs_tok.shape[1], self.n_tok, a_tok.shape[1], seq.device)
        else:  # native sequence and mask
            seq = torch.cat([obs_tok, a_tok], dim=1)
            mask = self.construct_mask(seq.device)
        out = self._transformer(seq, mask)
        return self.action_head(out[:, -self.config.action_chunk_size:])

    # ── training ──
    def reset_component_log(self):
        self._comp = {"action_loss": 0.0, "intent_loss": 0.0, "n": 0}

    def pop_component_means(self):
        n = max(self._comp["n"], 1)
        out = {"train/action_loss": self._comp["action_loss"] / n, "train/intent_loss": self._comp["intent_loss"] / n}
        self.reset_component_log()
        return out

    def flow_losses(self, seg_images, state, action, t=None, x0=None, ti=None, z0=None, intent_for_action=None):
        """Returns (L_action, L_intent). Draw order for t/x0 matches native compute_loss."""
        state, intent_gt = self.split_state(state)
        assert intent_gt is not None, "intent arms need the target appended to proprio"
        B, device = action.shape[0], action.device
        if t is None:
            t = self._sample_timestep(B, device)          # CPU global RNG, as native
        if x0 is None:
            x0 = self.sample_x0(action.shape, device)     # CUDA global RNG, as native
        g = self._gen(device)
        if ti is None:
            ti = torch.rand(B, device=device, generator=g) ** (1.0 / 1.5)   # Beta(1.5, 1) by inverse CDF
        if z0 is None:
            z0 = torch.randn(intent_gt.shape, device=device, generator=g)
        obs_tok = self.obs_tokens(seg_images, state)
        # action flow (teacher-forced on the GT intent unless intent_for_action is given)
        te = t[:, None, None]
        xt = (1 - te) * x0 + te * action
        cond = intent_gt.detach() if intent_for_action is None else intent_for_action
        v = self.action_velocity(obs_tok, xt, t, cond if self.mode == "condition" else None)
        l_act = torch.mean((v - (action - x0)) ** 2)
        # intent flow
        tie = ti[:, None, None]
        zt = (1 - tie) * z0 + tie * intent_gt
        vi = self.intent_velocity(obs_tok, zt, ti)
        l_int = torch.mean((vi - (intent_gt - z0)) ** 2)
        return l_act, l_int

    def compute_loss(self, seg_images, state, action):
        l_act, l_int = self.flow_losses(seg_images, state, action)
        self._comp["action_loss"] += float(l_act.detach())
        self._comp["intent_loss"] += float(l_int.detach())
        self._comp["n"] += 1
        return l_act + INTENT_LOSS_WEIGHT * l_int

    # ── inference ──
    @torch.no_grad()
    def sample_intent(self, obs_tok, intent_noise=None, nfe=INTENT_NFE):
        B = obs_tok.shape[0]
        z = intent_noise if intent_noise is not None else torch.randn(B, self.n_tok, self.tok_dim, device=obs_tok.device)
        z = z.detach().clone()
        dt = 1.0 / nfe
        for k in range(nfe):
            t = torch.full((B,), k * dt, device=obs_tok.device)
            z = z + self.intent_velocity(obs_tok, z, t) * dt
        return z

    @torch.no_grad()
    def generate(self, seg_images, state, init_noise=None, intent_noise=None, intent=None, denorm=True):
        """Returns (action [B,8,A] (denormalised if denorm), intent [B,n_tok,tok_dim] normalised or None)."""
        state, _gt = self.split_state(state)
        B, device = state.shape[0], state.device
        obs_tok = self.obs_tokens(seg_images, state)
        if self.mode == "condition":
            if intent is None:
                intent = self.sample_intent(obs_tok, intent_noise)
        x = init_noise.detach().clone() if init_noise is not None else self.sample_x0((B, *self.action_shape), device)
        dt = 1.0 / self.config.num_inference_steps
        for k in range(self.config.num_inference_steps):
            t = torch.full((B,), k * dt, device=device)
            x = x + self.action_velocity(obs_tok, x, t, intent) * dt
        if denorm and self.config.action_min is not None:
            x = denormalize_action(x, self.action_min, self.action_max)
        return x, intent

    @torch.no_grad()
    def generate_action(self, seg_images, state, n_actions=1, init_noise=None):
        """Official API (used by the official evaluate()): the appended GT intent, if present, is IGNORED;
        the intent is sampled from the intent flow."""
        assert n_actions == 1
        return self.generate(seg_images, state, init_noise=init_noise)[0]

    def get_parameter_count(self):
        d = super().get_parameter_count()
        c = d["components"]
        c["action_proj+timestep_proj"] = sum(p.numel() for m in (self.action_proj, self.timestep_proj) for p in m.parameters())
        c["intent_in_proj"] = sum(p.numel() for p in self.intent_in_proj.parameters())
        c["intent_head"] = sum(p.numel() for p in self.intent_head.parameters())
        c["intent_cond_proj"] = sum(p.numel() for p in self.intent_cond_proj.parameters()) if self.mode == "condition" else 0
        c["final_norm"] = sum(p.numel() for p in self.final_norm.parameters())
        return d
