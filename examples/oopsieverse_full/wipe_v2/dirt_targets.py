"""Dirt-targeted intent target for W3/W4 (reports/wipe_v2/wipe_v2_spec.md). numpy only.

Inputs per demo
  proprio [T+1, 23]  playback obs/proprio; proprio[t,14:17] = EEF world position, proprio[t,17:21] = EEF world quat
                     (xyzw). Verified == the sim EEF site position after each step to <= 2e-7 m (extract_markers.py).
  markers npz        from extract_markers.py (sim state for the demo's ep_meta + model_file + states, deterministic):
                     marker_pos_world [50,3], wiped_step [50] (playback info index i at which marker m was wiped,
                     -1 = never). The per-step counts are asserted equal to the playback part's info/newly_wiped.
Index convention (matrix intent_targets.py / alignment_audit.md): native sample t (t = 1 .. T-8) has input obs[t]
and target action[t : t+8]; playback info[i] is the result of action[i] executed from state i (arrival at state
i+1). So "markers wiped in steps t+1..t+8" = markers whose wiped_step i lies in [t, t+7], i.e. wiped by the 8
actions of the sample's own target chunk (chunk step k = i - t + 1).

Target (one token, 5 dims) for sample t:
  S_t = { m : t <= wiped_step[m] <= t+7 }                                   (in-window markers, may span events)
  if S_t is empty: i* = min{ wiped_step[m] : wiped_step[m] >= t+8 }, S_t = { m : wiped_step[m] == i* }
                                                                            (the first future wipe EVENT only)
  if still empty (no marker is wiped after t: all wiped earlier, or the demo never wipes again):
      raw token = [0, 0, 0, 0, 0] (has_target = 0).
  else: c = mean of marker_pos_world[S_t] (world, 3-D; all markers share z = counter top + 0.46)
      p = R_t^T (c - x_t)   (current EEF frame; x_t, R_t from proprio[t])
      raw token = [p_x, p_y, p_z, |S_t| / 50, 1]
  Window markers already wiped before t never re-enter (wiped_step is unique per marker).
Normalisation: per-dim MinMax over all mixed-TRAIN samples to [-1, 1] (matrix intent_targets.MinMax: zero range ->
  1). The same fitted normaliser is used for dev. Shape [N, 1, 5] like the matrix targets.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "matrix"))
import intent_targets as IT  # noqa: E402

MARKERS = "/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/markers/{cond}__{demo}.npz"
NUM_MARKERS = 50
NAME = "dirt"
SPEC = (1, 5)
COND = {"fb": "with", "nofb": "without", "with": "with", "without": "without"}


def load_markers(member):
    """member: 'wipe_countertop/fb/demo_10' or 'with/demo_10'."""
    parts = member.split("/")
    cond, demo = COND[parts[-2]], parts[-1]
    z = np.load(MARKERS.format(cond=cond, demo=demo))
    assert bool(z["counts_equal"]), f"{member}: sim wipe counts != playback info/newly_wiped"
    return z["marker_pos_world"].astype(np.float64), z["wiped_step"].astype(np.int64)


def target_sets(wiped_step, ts, H=IT.H):
    """-> list of index arrays S_t, and a 'source' code per t: 1 = in-window, 2 = first future event, 0 = none."""
    sets, src = [], []
    ws = np.asarray(wiped_step)
    for t in ts:
        S = np.nonzero((ws >= t) & (ws <= t + H - 1))[0]
        code = 1
        if len(S) == 0:
            fut = ws[ws >= t + H]
            if len(fut):
                S = np.nonzero(ws == fut.min())[0]
                code = 2
            else:
                code = 0
        sets.append(S)
        src.append(code)
    return sets, np.asarray(src)


def raw_dirt_targets(proprio, action, marker_pos, wiped_step, ts=None, return_src=False):
    proprio = np.asarray(proprio, np.float64)
    T = np.asarray(action).shape[0]
    assert proprio.shape[0] == T + 1
    assert len(wiped_step) == NUM_MARKERS and wiped_step.max() < T
    if ts is None:
        ts = IT.sample_indices(T)
    sets, src = target_sets(wiped_step, ts)
    out = np.zeros((len(ts), 1, 5))
    for j, (t, S) in enumerate(zip(ts, sets)):
        if len(S) == 0:
            continue
        c = marker_pos[S].mean(0)
        R = IT.quat_xyzw_to_mat(proprio[t, 17:21])
        out[j, 0, :3] = R.T @ (c - proprio[t, 14:17])
        out[j, 0, 3] = len(S) / NUM_MARKERS
        out[j, 0, 4] = 1.0
    return (out, src) if return_src else out


def fit_normalizer(raws):
    X = np.concatenate(raws)
    return IT.MinMax(X.min(0), X.max(0))
