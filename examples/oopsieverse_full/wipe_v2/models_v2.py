"""wipe_countertop v2 screening arms (reports/wipe_v2/wipe_v2_spec.md) as subclasses of the OFFICIAL CFMPolicy
(a739f99). Official sources and the matrix files are imported, never edited.

Shared, unchanged: the SegmentationEncoder's embedding + conv trunk (128^2 seg -> 8x8x256 map, shared across views),
StateEncoder, the 4-layer post-LN transformer (d=512, 8 heads, dropout 0.2), final LayerNorm, action_proj /
timestep_proj / action_head, sinusoidal sequence positions, the obs/action attention mask, Beta(1.5,1) flow times,
N(0,I) action noise, Euler-16 inference, compute_loss / generate_action. Only the READOUT of the 8x8x256 map changes:

  W0  official: AvgPool2d(4,4) -> 2x2x256 -> Linear(1024,d)-SiLU-Linear(d,d). 1 token per view/frame; 7 obs tokens.
  W1  spatial softmax: Conv1x1(256->32) -> softmax over the 64 positions per keypoint channel (temperature 1, fixed)
      -> expected (x, y) on a [-1,1] grid (x = column, y = row) -> 32 keypoints x 2 = 64-D (interleaved x1,y1,x2,..)
      -> Linear(64,d)-SiLU-Linear(d,d) (same form as the official projection). 1 token per view/frame; 7 obs tokens.
  W2  8x8 tokens: each view/frame map -> 64 tokens = Linear(256,d)(feature at (r,c)) + row_emb[r] + col_emb[c]
      (learned factorised 2-D positional embedding, N(0, 0.02^2) init, shared by views/frames). Obs tokens =
      3 views x 2 frames x 64 + 1 proprio = 385, ordered view-major, then frame, then row-major position (the
      official view/frame order). num_obs_tokens / max_seq_len / sinusoidal table are resized; construct_mask,
      forward, compute_loss and generate_action are the official methods unchanged (obs tokens never attend to
      action tokens; action tokens attend to everything).
  W3  dirt-targeted intent on the W0 readout: matrix IntentCFMPolicy (examples/oopsieverse_full/matrix/models.py)
      in "condition" mode with n_tok = 1, tok_dim = 5 -- the SAME intent machinery as I-PATH: intent rectified flow
      through the same transformer over [obs(7), noisy intent(1)], conditioning token intent_cond_proj(intent)
      inserted as [obs(7), intent(1), action(8)], teacher forcing (GT intent for the action flow), loss
      L_action + 1.0 L_intent, inference intent ~ 16 Euler steps from N(0,I) then action ~ 16 Euler steps.
      Target definition (exact, with edge cases): dirt_targets.py header. In short, for sample t: S = markers wiped
      by the chunk's 8 actions (playback info[t..t+7], i.e. on arrival at states t+1..t+8); if S is empty, the
      markers of the first later wipe event; token = [R_t^T (centroid_world(S) - x_t) (3), |S|/50, has_target],
      MinMax-normalised per dim to [-1,1] on the mixed train split; no future wipe at all -> [0,0,0, 0, 0] raw.

RNG discipline: every new module is initialised under a forked CPU RNG seeded from the run seed, after the official
modules were created in their native order, so the global CPU stream (DataLoader shuffle, action-flow t draws) is
the same as W0's at step 0. The official modules that W1/W2 no longer use (avg_pool, projection) are dropped after
creation, so all remaining shared modules (conv trunk, state encoder, transformer, heads) start from W0's init.

Loader for the evaluator: load_policy(run_dir, device, ckpt="step_30000.pth") -> policy (eval mode), keyed on
meta.json["arm"] in {"w0","w1","w2","w3"}. W0/W1/W2 expose the official generate_action(seg, state, init_noise);
W3 additionally exposes generate(seg, state, init_noise=x0, intent_noise=z) -> (actions, intent) (matrix API; the
matrix evaluator's per-chunk draw order is x0 [1,8,12] then z [1,1,5] from the per-episode generator).
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(os.path.dirname(HERE), "matrix")
OOPSIE = "/edrive1/lizad/oopsieverse_full_study/src/oopsieverse"
for p in (OOPSIE, MATRIX):
    if p not in sys.path:
        sys.path.insert(0, p)

from learning.models import CFMPolicy, PolicyConfig  # noqa: E402
from learning.models.cfm_policy import PositionalEncoding  # noqa: E402
from models import IntentCFMPolicy  # noqa: E402  (matrix)

N_KEYPOINTS = 32
GRID = 8          # 128 / 16
FEAT_C = 256
W3_SPEC = (1, 5)  # n_tok, tok_dim
ARMS_V2 = ("w0", "w1", "w2", "w3")


def _proj(i, d):
    return nn.Sequential(nn.Linear(i, d), nn.SiLU(), nn.Linear(d, d))


def _trunk(enc, x):
    """The official SegmentationEncoder.forward up to (and including) conv_layers: [N,H,W] ids -> [N,256,8,8]."""
    x = torch.clamp(x.long(), min=0, max=enc.num_classes)
    x = enc.embedding(x).permute(0, 3, 1, 2).contiguous()
    return enc.conv_layers(x)


class SpatialSoftmaxSegEncoder(nn.Module):
    """W1 readout on the official trunk (embedding + conv_layers are the official modules, reused)."""

    def __init__(self, official_enc, feature_dim, seed):
        super().__init__()
        self.config = official_enc.config
        self.num_classes = official_enc.num_classes
        self.embedding = official_enc.embedding
        self.conv_layers = official_enc.conv_layers
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2_000_003 + seed)
            self.keypoint_conv = nn.Conv2d(FEAT_C, N_KEYPOINTS, kernel_size=1)
            self.projection = _proj(2 * N_KEYPOINTS, feature_dim)
        lin = torch.linspace(-1.0, 1.0, GRID)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")  # gy: row coordinate, gx: column coordinate
        self.register_buffer("pos_x", gx.reshape(1, 1, -1), persistent=False)
        self.register_buffer("pos_y", gy.reshape(1, 1, -1), persistent=False)

    def keypoints(self, x):
        f = self.keypoint_conv(_trunk(self, x))              # [N, K, 8, 8]
        N, K, H, W = f.shape
        assert H == GRID and W == GRID, (H, W)
        a = F.softmax(f.reshape(N, K, H * W), dim=-1)       # temperature 1
        ex = (a * self.pos_x).sum(-1)
        ey = (a * self.pos_y).sum(-1)
        return torch.stack([ex, ey], dim=-1).reshape(N, 2 * K)  # [N, 64]

    def forward(self, x):
        return self.projection(self.keypoints(x))           # [N, d]


class TokenSegEncoder(nn.Module):
    """W2 readout: 64 tokens per image."""

    def __init__(self, official_enc, feature_dim, seed):
        super().__init__()
        self.config = official_enc.config
        self.num_classes = official_enc.num_classes
        self.embedding = official_enc.embedding
        self.conv_layers = official_enc.conv_layers
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3_000_017 + seed)
            self.token_proj = nn.Linear(FEAT_C, feature_dim)
            self.row_emb = nn.Parameter(torch.randn(GRID, feature_dim) * 0.02)
            self.col_emb = nn.Parameter(torch.randn(GRID, feature_dim) * 0.02)

    def forward(self, x):
        f = _trunk(self, x)                                   # [N, 256, 8, 8]
        N, C, H, W = f.shape
        assert H == GRID and W == GRID, (H, W)
        tok = self.token_proj(f.flatten(2).transpose(1, 2))  # [N, 64, d], row-major (r, c)
        pos = (self.row_emb[:, None, :] + self.col_emb[None, :, :]).reshape(H * W, -1)
        return tok + pos[None]


class SpatialSoftmaxCFMPolicy(CFMPolicy):
    """W1."""

    def __init__(self, config, seed=0):
        self._v2_seed = int(seed)
        super().__init__(config)

    def _build_model(self):
        super()._build_model()  # official modules, native order/draws
        self.seg_encoder = SpatialSoftmaxSegEncoder(self.seg_encoder, self.config.feature_dim, self._v2_seed)


class TokenCFMPolicy(CFMPolicy):
    """W2."""

    def __init__(self, config, seed=0):
        self._v2_seed = int(seed)
        super().__init__(config)

    def _build_model(self):
        super()._build_model()
        cfg = self.config
        self.seg_encoder = TokenSegEncoder(self.seg_encoder, cfg.feature_dim, self._v2_seed)
        self.tokens_per_image = GRID * GRID
        self.num_obs_tokens = cfg.num_seg_views * cfg.frame_stack * self.tokens_per_image + 1
        self.max_seq_len = self.num_obs_tokens + cfg.action_chunk_size
        self.pos_encoding = PositionalEncoding(cfg.feature_dim, self.max_seq_len)  # buffer only (no RNG)

    def encode_segmentation(self, seg_images):
        tokens = []
        for _view, seg in seg_images.items():
            B, T, H, W = seg.shape
            f = self.seg_encoder(seg.reshape(B * T, H, W))     # [B*T, 64, d]
            tokens.append(f.reshape(B, T * self.tokens_per_image, -1))
        return torch.cat(tokens, dim=1)                         # [B, V*T*64, d]


class DirtIntentCFMPolicy(IntentCFMPolicy):
    """W3: matrix intent machinery (condition mode), dirt target 1 x 5."""

    def __init__(self, config, seed=0):
        super().__init__(config, W3_SPEC[0], W3_SPEC[1], "condition", intent_seed=seed)


def build_policy(arm, pcfg, seed):
    if arm == "w0":
        return CFMPolicy(pcfg)
    if arm == "w1":
        return SpatialSoftmaxCFMPolicy(pcfg, seed=seed)
    if arm == "w2":
        return TokenCFMPolicy(pcfg, seed=seed)
    if arm == "w3":
        return DirtIntentCFMPolicy(pcfg, seed=seed)
    raise ValueError(arm)


def policy_config_from_meta(meta):
    pc = dict(meta["policy_config"])
    pc["action_min"] = np.asarray(pc["action_min"], np.float32)
    pc["action_max"] = np.asarray(pc["action_max"], np.float32)
    return PolicyConfig(**pc)


def load_policy(run_dir, device, ckpt="step_30000.pth"):
    """-> policy in eval mode on `device`. Keyed on meta.json["arm"]. Attributes added for the evaluator:
    v2_arm, intent_nfe (0 for w0-w2), intent_shape (None or (n_tok, tok_dim)), ckpt_path, meta."""
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    arm = meta["arm"]
    assert arm in ARMS_V2, arm
    pol = build_policy(arm, policy_config_from_meta(meta), int(meta["seed"]))
    path = ckpt if os.path.isabs(ckpt) else os.path.join(run_dir, "ckpt", ckpt)
    sd = torch.load(path, map_location="cpu")["policy_state_dict"]
    pol.load_state_dict(sd)
    pol.to(device).eval()
    pol.v2_arm, pol.meta, pol.ckpt_path = arm, meta, path
    pol.intent_nfe = 16 if arm == "w3" else 0
    pol.intent_shape = W3_SPEC if arm == "w3" else None
    return pol
