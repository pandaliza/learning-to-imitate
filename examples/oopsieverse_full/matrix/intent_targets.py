"""Future-motion (intent) targets for the policy matrix (spec section 8). numpy only.

Index convention (official playback HDF5, roster/alignment_audit.md):
  proprio[t]  = obs at state t (length T+1), proprio[t, 14:17] = robot0_eef_pos (world),
                proprio[t, 17:21] = robot0_eef_quat (xyzw, world).
  action[t]   = command executed FROM state t, leading to state t+1 (length T).
  Native sample t (t = 1 .. T-H, PlaybackDataset): input obs[t] (+ seg obs[t-1]), target action[t : t+H].
  The pose reached after executing the first k actions of the target chunk is proprio[t+k], k = 1..H.
  The gripper command at chunk step k is action[t+k-1, 6] (the k-th command of the chunk).
  (In teleop indexing this is obs[t+k] and actions[t+k]; playback action[t] == teleop actions[t+1].)

H = native predicted action horizon = action_chunk = 8 control steps (learning/configs/*.yaml).

Targets (one row per sample t; each target is a sequence of `n_tok` tokens of width `tok_dim`):
  imean   : legacy mean future EEF pose (mip/datasets/robomimic_dataset.py:364-396, intent_type="mean",
            intent_keys = [robot0_eef_pos, robot0_eef_quat], intent_horizon = 8):
            mean_{k=1..8} of the per-frame MinMax-normalised [eef_pos, eef_quat] (world frame, xyzw quaternion
            averaged component-wise, no hemisphere alignment, no renormalisation). 1 token x 7.
  ipath   : ordered future EEF poses at k in PATH_KS = (2, 4, 6, 8) (~H/4, H/2, 3H/4, H), each relative to the
            CURRENT EEF frame: p = R_t^T (x_{t+k} - x_t), R = R_t^T R_{t+k} -> 6D (first two columns of R).
            4 tokens x 9.
  ipathg  : ipath + gripper command action[t+k-1, 6] per timestamp. 4 tokens x 10.
  cmean   : canonical mean-pose control: mean over k = 1..8 of the SAME current-EEF-frame [p, 6D] encoding
            as ipath (arithmetic mean of the 9-D vectors). 1 token x 9.
  auxpathg: the ipathg target (co-prediction head, no conditioning).
Normalisation (fitted on the mixed TRAIN split only, shared by every arm that uses the target):
  imean: legacy MinMaxNormalizer of the per-frame [eef_pos, eef_quat] (range over train frames t = 1..T),
         mapped to [-1, 1]; since the map is affine per dim, normalise(mean) == mean(normalise) exactly as legacy.
  others: per-(token, dim) MinMax over all train target vectors to [-1, 1] (range 0 -> 1, as MinMaxNormalizer).
"""

import numpy as np

H = 8
PATH_KS = (2, 4, 6, 8)
EEF_POS = slice(14, 17)
EEF_QUAT = slice(17, 21)
GRIP_DIM = 6

TARGET_SPECS = {
    # name: (n_tok, tok_dim)
    "imean": (1, 7),
    "ipath": (4, 9),
    "ipathg": (4, 10),
    "cmean": (1, 9),
}


# ───────────────────────────── rotation utilities ─────────────────────────────

def quat_xyzw_to_mat(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def mat_to_quat_xyzw(R):
    """Shepperd's method; returns w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4))
    for i, m in enumerate(flat):
        tr = np.trace(m)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        q = np.array([x, y, z, w])
        out[i] = q if w >= 0 else -q
    return out.reshape(R.shape[:-2] + (4,))


def mat_to_6d(R):
    """Continuous 6D rotation representation (Zhou et al. 2019): first two COLUMNS of R, [c0; c1]."""
    R = np.asarray(R)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def sixd_to_mat(d6):
    """Gram-Schmidt inverse of mat_to_6d; valid for any non-degenerate 6-vector."""
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2p / np.linalg.norm(a2p, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def relative_pose(pos_cur, quat_cur, pos_fut, quat_fut):
    """Pose of the future EEF expressed in the current EEF frame. Returns (p_rel[...,3], R_rel[...,3,3])."""
    Rc = quat_xyzw_to_mat(quat_cur)
    Rf = quat_xyzw_to_mat(quat_fut)
    RcT = np.swapaxes(Rc, -1, -2)
    p = np.einsum("...ij,...j->...i", RcT, np.asarray(pos_fut, np.float64) - np.asarray(pos_cur, np.float64))
    return p, RcT @ Rf


def compose_pose(pos_cur, quat_cur, p_rel, R_rel):
    """Inverse of relative_pose: world pose of the future EEF. Returns (pos, R)."""
    Rc = quat_xyzw_to_mat(quat_cur)
    return np.asarray(pos_cur, np.float64) + np.einsum("...ij,...j->...i", Rc, p_rel), Rc @ R_rel


# ───────────────────────────── extraction ─────────────────────────────

def sample_indices(T, frame_stack=2, chunk=H):
    """Native PlaybackDataset sample range for a demo with T actions: t = frame_stack-1 .. T-chunk."""
    return np.arange(frame_stack - 1, T - chunk + 1)


def raw_targets(proprio, action, ts=None):
    """All raw (un-normalised) targets for one demo. proprio [T+1, 23], action [T, A].
    Returns {name: float64 [n_samples, n_tok, tok_dim]} for samples ts (default: native range)."""
    proprio = np.asarray(proprio, np.float64)
    action = np.asarray(action, np.float64)
    T = action.shape[0]
    assert proprio.shape[0] == T + 1, (proprio.shape, action.shape)
    if ts is None:
        ts = sample_indices(T)
    ts = np.asarray(ts)
    assert ts.min() >= 0 and ts.max() + H <= T, "future window must lie inside the demo (no padding)"
    pos, quat = proprio[:, EEF_POS], proprio[:, EEF_QUAT]
    ks_all = np.arange(1, H + 1)
    fut = ts[:, None] + ks_all[None, :]                       # [N, 8] state indices t+1..t+8
    # legacy mean (raw world-frame pos + xyzw quat, component-wise mean)
    imean_raw = proprio[fut][..., 14:21].mean(axis=1)          # [N, 7]
    # current-frame relative poses for all k = 1..8
    p_rel, R_rel = relative_pose(pos[ts][:, None], quat[ts][:, None], pos[fut], quat[fut])  # [N,8,3], [N,8,3,3]
    rel9 = np.concatenate([p_rel, mat_to_6d(R_rel)], axis=-1)  # [N, 8, 9]
    kidx = np.array(PATH_KS) - 1
    ipath = rel9[:, kidx]                                      # [N, 4, 9]
    grip = action[ts[:, None] + kidx[None, :], GRIP_DIM]       # [N, 4] = action[t+k-1, 6]
    ipathg = np.concatenate([ipath, grip[..., None]], axis=-1)  # [N, 4, 10]
    cmean = rel9.mean(axis=1, keepdims=True)                   # [N, 1, 9]
    return {"imean": imean_raw[:, None, :], "ipath": ipath, "ipathg": ipathg, "cmean": cmean}


class MinMax:
    """Same semantics as mip.dataset_utils.MinMaxNormalizer: [-1, 1], zero range -> 1."""

    def __init__(self, vmin, vmax):
        self.min = np.asarray(vmin, np.float64)
        self.max = np.asarray(vmax, np.float64)
        rng = self.max - self.min
        rng[rng == 0] = 1
        self.range = rng

    @classmethod
    def fit(cls, X):
        X = np.asarray(X, np.float64).reshape(-1, np.shape(X)[-1])
        return cls(X.min(0), X.max(0))

    def normalize(self, x):
        return ((np.asarray(x, np.float64) - self.min) / self.range) * 2 - 1

    def unnormalize(self, x):
        return (np.asarray(x, np.float64) + 1) / 2 * self.range + self.min

    def to_dict(self):
        return {"min": self.min.tolist(), "max": self.max.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["min"], d["max"])


def fit_normalizer(name, demos):
    """demos: list of (proprio [T+1,23], action [T,A]) of the mixed TRAIN split."""
    if name == "imean":
        # legacy: per-frame normaliser over the observation frames used in training (t = 1..T; the stale
        # playback obs[0] is excluded, roster/alignment_audit.md)
        frames = np.concatenate([np.asarray(p, np.float64)[1:, 14:21] for p, _ in demos])
        return MinMax.fit(frames)
    tgts = np.concatenate([raw_targets(p, a)[name] for p, a in demos])  # [N, n_tok, d]
    return MinMax(tgts.min(0), tgts.max(0))  # per (token, dim): timestamps have different scales
