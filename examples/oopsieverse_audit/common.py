"""Shared loaders and stored-field label derivations for the OopsieVerse Stage-0 audit.

Everything here reads stored HDF5 fields only; no simulator is used.
"""

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

SNAPSHOT = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--ut-robin-lab--oopsieverse-demos/snapshots/"
        "c0ec29395fdcf898de2aed8dc0b3e786dc2a37d3/paper_experiments/teleop_data"
    )
)
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "oopsieverse_audit"
FIG = OUT / "figures"

ROBOCASA_TASKS = {"pick_egg": "egg", "wipe_countertop": "sponge"}
B1K_TASKS = ("add_firewood", "shelve_item")
SPLITS = ("without_live_feedback", "with_live_feedback")
SPLIT_TAG = {"without_live_feedback": "nofb", "with_live_feedback": "fb"}

# Repo intent definition (examples/configs/task/*_state_intent.yaml + oopsieverse_base.yaml)
OBS_KEYS = ["object-state", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
INTENT_KEYS = ["robot0_eef_pos", "robot0_eef_quat"]
OBS_STEPS = 2
INTENT_H = 8
VAL_PCT = 0.1

# Gripper width (qpos[0]-qpos[1]) at or above this counts as "open" (fully open ~0.078-0.080 m).
OPEN_WIDTH = 0.075


def h5path(task, split):
    return SNAPSHOT / task / f"{split}.hdf5"


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def demo_names(f):
    return sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))


def _f(x):
    return float(x.decode() if isinstance(x, bytes) else x)


def load_robocasa_episode(task, split, name, f=None):
    """Load one RoboCasa episode and derive stored-field labels."""
    own = f is None
    if own:
        f = h5py.File(h5path(task, split), "r")
    d = f["data"][name]
    obj = ROBOCASA_TASKS[task]
    ep_meta = json.loads(d.attrs["ep_meta"])
    names = [str(n) for n in d.attrs["health_list_link_names"]]
    obs = {k: d["obs"][k][:] for k in d["obs"].keys()}
    health = obs["health"].astype(np.float64)
    T = len(d["actions"])
    dmg_rows, obj_force, ent_force = [], [], []
    for s in d["info/damage_info"][:]:
        j = json.loads(s)
        row = {}
        for a in j:
            for b in j[a]:
                row[f"{a}@{b}"] = j[a][b]["mechanical"]
        dmg_rows.append(row)
    ent_keys = names
    dmg = np.array([[r.get(k, {}).get("damage", 0.0) for k in ent_keys] for r in dmg_rows])
    force = np.array([[r.get(k, {}).get("filtered_raw_sim_forces", 0.0) for k in ent_keys] for r in dmg_rows])
    extra_ents = sorted({k for r in dmg_rows for k in r} - set(ent_keys))

    ep = dict(
        task=task,
        split=split,
        name=name,
        episode_id=f"{task}/{SPLIT_TAG[split]}/{name}",
        idx=int(name.split("_")[1]),
        T=T,
        actions=d["actions"][:],
        dones=d["dones"][:],
        rewards=d["rewards"][:],
        states=d["states"][:],
        obs=obs,
        health=health,
        health_names=names,
        dmg=dmg,
        force=force,
        extra_damage_entities=extra_ents,
        ep_meta=ep_meta,
        model_hash=hashlib.sha256(d.attrs["model_file"].encode()).hexdigest()[:16],
        obs_keys=sorted(obs.keys()),
        info_keys=sorted(d["info"].keys()),
        obj=obj,
    )
    if task == "wipe_countertop":
        ep["proportion_wiped"] = np.array([_f(x) for x in d["info/proportion_wiped"][:]])
        ep["total_wiped"] = np.array([_f(x) for x in d["info/total_wiped"][:]])
        ep["total_markers"] = np.array([_f(x) for x in d["info/total_markers"][:]])
        ep["dirt"] = np.array(ep_meta["dirt_marker_positions"], dtype=np.float64)
    if own:
        f.close()
    derive_labels(ep)
    derive_geometry(ep)
    return ep


def derive_labels(ep):
    """Stored-field outcome labels. See report section 'Derivations'."""
    h = ep["health"]
    init = h[0]
    below = h < init[None, :] - 1e-6
    any_t = below.any(1)
    ep["any_damage"] = bool(any_t.any())
    ep["first_damage_step"] = int(np.argmax(any_t)) if ep["any_damage"] else None
    ep["damaged_entities"] = [ep["health_names"][k] for k in range(h.shape[1]) if below[:, k].any()]
    ep["initial_health"] = init
    ep["final_health"] = h[-1]
    ep["min_health"] = h.min(0)
    ep["max_health_loss"] = float((init - h.min(0)).max())
    ep["health_monotone"] = bool(np.all(np.diff(h, axis=0) <= 1e-6))
    # consistency: health[t] == health[t-1] - damage_info[t].damage (per entity, clipped at 0)
    pred = np.maximum(h[:-1] - ep["dmg"][1:], 0.0)
    ep["health_dmg_max_abs_err"] = float(np.abs(pred - h[1:]).max())
    dmg_t = (ep["dmg"] > 0).any(1)
    ep["first_damage_step_info"] = int(np.argmax(dmg_t)) if dmg_t.any() else None

    if ep["task"] == "wipe_countertop":
        pw = ep["proportion_wiped"]
        ep["final_proportion_wiped"] = float(pw[-1])
        ep["task_success"] = bool(ep["total_wiped"][-1] >= ep["total_markers"][-1])
        ep["success_basis"] = "stored info/total_wiped[-1] == info/total_markers[-1] (proportion_wiped=1.0)"
        ep["first_full_wipe_step"] = int(np.argmax(pw >= 1.0)) if (pw >= 1.0).any() else None
    else:
        z = ep["obs"][f"{ep['obj']}_pos"][:, 2]
        ep["proxy_lift_m"] = float(z[-1] - z[0])
        ep["proxy_max_lift_m"] = float(z.max() - z[0])
        ep["task_success"] = None  # not stored
        ep["success_basis"] = "not stored (rewards all 0, dones all False, no success key)"

    ts = ep["task_success"]
    if ep["any_damage"]:
        ep["safe_success"] = False
    else:
        ep["safe_success"] = ts  # None if success unknown

    def cls(success):
        if success is None:
            return "unknown"
        if not success:
            return "failed"
        return "successful_damaging" if ep["any_damage"] else "safe_success"

    ep["outcome_class"] = cls(ts)
    # conditional on the lift proxy (pick_egg only; clearly labelled as a proxy)
    if ep["task"] == "pick_egg":
        ep["proxy_task_success"] = ep["proxy_lift_m"] >= 0.05
        ep["outcome_class_proxy"] = cls(ep["proxy_task_success"])
    else:
        ep["proxy_task_success"] = ts
        ep["outcome_class_proxy"] = ep["outcome_class"]


def derive_geometry(ep):
    """Outcome-free initial geometry and per-step stage variables."""
    o = ep["obs"]
    base = np.array(ep["ep_meta"]["init_robot_base_pos"][:2])
    ep["base_xy"] = base
    ep["base_yaw"] = float(ep["ep_meta"]["init_robot_base_ori"][2])
    ep["obj0_world"] = o[f"{ep['obj']}_pos"][0].astype(np.float64)
    ep["obj0_rel_base"] = ep["obj0_world"][:2] - base
    ep["eef0_world"] = o["robot0_eef_pos"][0].astype(np.float64)
    ep["eef0_rel_base"] = o["robot0_base_to_eef_pos"][0].astype(np.float64)
    if "dirt" in ep:
        ep["dirt_centroid"] = ep["dirt"].mean(0)
        c = ep["dirt"] - ep["dirt"].mean(0)
        u, s, vt = np.linalg.svd(c, full_matrices=False)
        ang = np.arctan2(vt[0, 1], vt[0, 0]) % np.pi
        ep["dirt_angle"] = float(ang)
        ep["dirt_extent"] = float(s[0] / np.sqrt(len(c)))
    eef = o["robot0_eef_pos"].astype(np.float64)
    ep["eef"] = eef
    ep["quat"] = o["robot0_eef_quat"].astype(np.float64)
    g = o["robot0_gripper_qpos"].astype(np.float64)
    ep["grip_width"] = g[:, 0] - g[:, 1]
    ep["grip_cmd"] = ep["actions"][:, 6]
    ep["speed"] = np.r_[0.0, np.linalg.norm(np.diff(eef, axis=0), axis=1)]
    ep["eef_obj_dist"] = np.linalg.norm(o[f"{ep['obj']}_to_robot0_eef_pos"].astype(np.float64), axis=1)
    ep["obj_lift"] = o[f"{ep['obj']}_pos"][:, 2].astype(np.float64) - ep["obj0_world"][2]
    ep["open"] = ep["grip_width"] >= OPEN_WIDTH
    # Stage: approach = before the first time the gripper leaves the open state.
    # Grippers start partly closed (~0.054 m) and are opened by the operator first, so the onset is the
    # first closing AFTER the first fully-open step (frozen_rules ERRATA v1.1).
    first_open = int(np.argmax(ep["open"])) if ep["open"].any() else 0
    closed_after = np.where(~ep["open"][first_open:])[0]
    ep["first_open"] = first_open
    ep["grasp_onset"] = first_open + int(closed_after[0]) if len(closed_after) else ep["T"]
    k = ep["obj"]
    ep["obj_force"] = ep["force"][:, ep["health_names"].index(f"{k}@{k}_main")]
    finger_idx = [i for i, n in enumerate(ep["health_names"]) if "finger" in n or "gripper" in n or "hand" in n or "eef" in n]
    ep["finger_force_max"] = ep["force"][:, finger_idx].max(1)


def load_task(task):
    eps = []
    for split in SPLITS:
        with h5py.File(h5path(task, split), "r") as f:
            for n in demo_names(f):
                eps.append(load_robocasa_episode(task, split, n, f))
    return eps


def repo_intent_normalizer(eps):
    """Min/max over train portion (first 90% of each condition), as OopsieverseStateDataset does."""
    rows = []
    for split in SPLITS:
        sub = [e for e in eps if e["split"] == split]
        n_val = int(len(sub) * VAL_PCT)
        for e in sorted(sub, key=lambda e: e["idx"])[: len(sub) - n_val]:
            rows.append(np.concatenate([e["obs"][k] for k in OBS_KEYS], -1))
    X = np.concatenate(rows).astype(np.float32)
    mn, mx = X.min(0), X.max(0)
    rng = mx - mn
    rng[rng == 0] = 1
    # intent slice in concatenated obs: eef_pos (3) + eef_quat (4)
    dims = [eps[0]["obs"][k].shape[-1] for k in OBS_KEYS]
    off = np.cumsum([0] + dims)
    s, e = off[OBS_KEYS.index("robot0_eef_pos")], off[OBS_KEYS.index("robot0_eef_quat") + 1]
    return dict(min=mn, range=rng, slice=(int(s), int(e)), obs_dim=int(off[-1]))


def repo_obs(ep):
    return np.concatenate([ep["obs"][k] for k in OBS_KEYS], -1).astype(np.float32)


def normalize(x, norm):
    return ((x - norm["min"]) / norm["range"]) * 2 - 1


def repo_intent_at(ep, t, norm):
    """Repo 'mean' intent for decision step t (obs history t-1,t; future t+1..t+H), normalized space.

    Mirrors RobomimicDataset.sample_to_data: mean(state_norm[obs_steps:obs_steps+H, intent_slice]).
    Uses only windows fully inside the episode (no end padding).
    """
    s, e = norm["slice"]
    X = normalize(repo_obs(ep)[t + 1 : t + 1 + INTENT_H], norm)[:, s:e]
    return X.mean(0)


def valid_t(ep, t):
    return OBS_STEPS - 1 <= t and t + INTENT_H < ep["T"]


def fmt(x):
    if x is None:
        return "unknown"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    return x
