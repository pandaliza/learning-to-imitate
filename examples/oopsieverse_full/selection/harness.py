"""Candidate-selection harness core (spec §11; protocol_v1; metric contract oopsie-metric-v1.0).

Episode protocol = baseline/rollout_eval.run_episode (frozen evaluator), re-expressed as a resumable state machine
so that an episode can be cut at a chunk boundary, snapshotted (snapshot_damage_state.py) and continued:
  * timeout 650 (pick_egg) / 1250 (wipe) control steps counted from the PARENT reset; terminate 10 steps after the
    first success; success = official predicate at the last executed step; safety over the whole executed
    trajectory (prefix + candidate chunk + tail); non-finite action = policy failure;
  * observations: per-camera class seg -> train vocabulary LUT, 2-frame stack (current + previous control step,
    reset frame duplicated at t=0), proprio[:23]; identical preprocessing (rollout_eval.Policy.frame/model_inputs);
  * per chunk the native noise draw order: x0 ~ N(0,I) [8x12] first, then (conditioned arms) z ~ N(0,I).
`verify_harness.py` checks that a t=0 "null branch" of this state machine reproduces rollout_eval.run_episode
record-for-record.

Render skipping (verified bit-identical in harness_verification.md): only the two frames that enter the policy's
2-frame history at the next replanning point are rendered (steps t_r-1 and t_r of each chunk); physics, health and
success do not depend on rendering.
"""

import copy
import json
import os
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.dirname(HERE)
for p in (HERE, os.path.join(FULL, "matrix"), os.path.join(FULL, "baseline"), FULL):
    if p not in sys.path:
        sys.path.insert(0, p)
import rollout_eval as RE  # noqa: E402  (frozen evaluator; imported read-only)
import snapshot_damage_state as SDS  # noqa: E402

TIMEOUT = RE.TIMEOUT
POST_SUCCESS = RE.POST_SUCCESS
TARGET = RE.TARGET
METRIC_VERSION = RE.METRIC_VERSION
EXEC_H = 8


# ───────────────────────────── observation frames ─────────────────────────────

class Frames:
    """Policy-side preprocessing, identical to rollout_eval.Policy.frame / model_inputs (numpy only)."""

    def __init__(self, run_dir):
        meta = json.load(open(os.path.join(run_dir, "meta.json")))
        tc = meta["trainer_config"]
        self.obs_keys = tc["obs_keys"]
        self.seg_keys = [k for k in self.obs_keys if "seg" in k]
        self.seg_size = int(tc["seg_img_size"])
        self.frame_stack = int(tc["frame_stack"])
        self.class_to_id = meta["class_to_id"]

    def lut(self, seg_id_to_class):
        m = max(int(k) for k in seg_id_to_class)
        lut = np.zeros(m + 1, dtype=np.int64)
        for k, name in seg_id_to_class.items():
            lut[int(k)] = self.class_to_id.get(name, 0)
        return lut

    def frame(self, obs, lut):
        segs = {}
        for key in self.seg_keys:
            cam = key.split("/")[1]
            raw = np.asarray(obs[f"{cam}_segmentation_class"]).astype(np.int64)
            g = lut[np.clip(raw, 0, len(lut) - 1)]
            assert g.max() < 256
            segs[key] = g.astype(np.uint8)
        prop = np.asarray(obs["robot0_proprio"], dtype=np.float32).reshape(-1)[:23]
        return segs, prop

    def model_inputs(self, hist):
        """-> ({seg_key: uint8 [T,128,128]}, prop float32 [23]) ready for the inference server."""
        import torch
        import torch.nn.functional as F
        seg = {}
        for key in self.seg_keys:
            st = np.stack([h[0][key].astype(np.int64) for h in hist])
            x = torch.from_numpy(st).float().unsqueeze(1)
            x = F.interpolate(x, size=(self.seg_size, self.seg_size), mode="nearest-exact").squeeze(1)
            seg[key] = x.to(torch.uint8).numpy()
        return seg, hist[-1][1].copy()


# ───────────────────────────── episode state machine ─────────────────────────────

def new_episode(env, task, frames, obs, parent):
    """State right after the canonical parent reset (t = 0)."""
    link_names = list(env.health_list_link_names)
    lut = frames.lut(env.seg_id_to_class)
    f0 = frames.frame(obs, lut)
    h0 = np.asarray(obs["health"], np.float64)
    ents = {}
    for k, n in enumerate(link_names):
        ents.setdefault(n.split("@", 1)[0], []).append(k)
    obj0 = env.sim.data.body_xpos[env.obj_body_id[TARGET[task]]].copy().tolist() if TARGET[task] in env.obj_body_id else None
    return {"task": task, "parent": dict(parent), "t": 0, "timeout": TIMEOUT[task], "link_names": link_names,
            "ents": ents, "hist": [f0] * frames.frame_stack, "health": h0,
            "ent_mean_sum": float(np.mean([h0[ks].min() for ks in ents.values()])), "ent_mean_n": 1,
            "max_tab": np.zeros((len(link_names), 4)), "first_damage_step": None, "first_succ": None,
            "last_succ": None, "h_last": None, "h_max": None, "invalid": False, "done": False, "n_chunks": 0,
            "initial_success": RE.success_now(env), "obj0": obj0,
            "termination": None, "chunks_log": []}


def _needed_frames(t_next, t_r):
    """Is the observation after the step that brings t to t_next needed for the history at replan time t_r?"""
    return t_next >= t_r - 1


def step_env(env, ep, a, frames, lut, render):
    """One control step, bookkeeping identical to rollout_eval.run_episode. Returns True if the episode ended."""
    if not np.all(np.isfinite(a)):
        ep["invalid"] = True
        ep["done"] = True
        ep["termination"] = "invalid_action"
        return True
    env.render_segmentation = bool(render)
    obs, r, d, info = env.step(a)
    env.render_segmentation = True
    ep["t"] += 1
    t = ep["t"]
    h = np.asarray(obs["health"], np.float64)
    ep["health"] = h
    ep["ent_mean_sum"] += float(np.mean([h[ks].min() for ks in ep["ents"].values()]))
    ep["ent_mean_n"] += 1
    mt = ep["max_tab"]
    for k, n in enumerate(ep["link_names"]):
        o, l = n.split("@", 1)
        m = info["damage_info"].get(o, {}).get(l, {}).get("mechanical", {})
        vals = [m.get("damage", 0.0), m.get("impact_forces", 0.0), m.get("filtered_qs_forces", 0.0),
                m.get("filtered_raw_sim_forces", 0.0)]
        vals = [0.0 if (v is None or not np.isfinite(v)) else float(v) for v in vals]
        mt[k] = np.maximum(mt[k], vals)
        if ep["first_damage_step"] is None and vals[0] > 0:
            ep["first_damage_step"] = t
    s = RE.success_now(env)
    ep["last_succ"] = bool(s)
    if ep["task"] == "pick_egg":
        hh = RE.egg_height(env)
        ep["h_last"] = hh
        ep["h_max"] = hh if ep["h_max"] is None else max(ep["h_max"], hh)
    if s and ep["first_succ"] is None:
        ep["first_succ"] = t
    if render:
        ep["hist"] = (ep["hist"] + [frames.frame(obs, lut)])[-frames.frame_stack:]
    else:
        ep["hist"] = ep["hist"] + [None]
        ep["hist"] = ep["hist"][-frames.frame_stack:]
    if ep["first_succ"] is not None and t >= ep["first_succ"] + POST_SUCCESS:
        ep["done"], ep["termination"] = True, "success_window"
    elif t >= ep["timeout"]:
        ep["done"], ep["termination"] = True, "timeout"
    return ep["done"]


def exec_chunk(env, ep, chunk, frames, lut, render_skip=True, exec_h=EXEC_H):
    """Execute the first exec_h actions of a chunk. Rendering only for frames needed at the next replan."""
    ep["n_chunks"] += 1
    t_r = ep["t"] + exec_h
    for a in chunk[:exec_h]:
        render = (not render_skip) or _needed_frames(ep["t"] + 1, t_r)
        if step_env(env, ep, a, frames, lut, render):
            return True
    if any(h is None for h in ep["hist"]):
        raise RuntimeError("history frame missing at replan time (render skipping bug)")
    return False


def finalize(ep):
    """Record fields identical to rollout_eval.run_episode."""
    task, H = ep["task"], ep["health"]
    ent_final = {o: float(H[ks].min()) for o, ks in ep["ents"].items()}
    success = bool(ep["last_succ"]) if (ep["last_succ"] is not None and not ep["invalid"]) else False
    primary_safe = all(v > 95.0 for v in ent_final.values())
    strict_safe = bool(np.all(H >= 100.0 - 1e-3))
    compat_safe = float(np.mean(list(ent_final.values()))) >= 95.0
    ln, mt = ep["link_names"], ep["max_tab"]
    tgt = [k for k, n in enumerate(ln) if n.startswith(TARGET[task] + "@")]
    return {
        "metric_version": METRIC_VERSION, "task": task, "steps": ep["t"], "chunks": ep["n_chunks"],
        "invalid_action": ep["invalid"], "initial_success": ep["initial_success"],
        "first_success_step": ep["first_succ"], "success": success,
        "success_h010": bool(ep["h_last"] >= 0.10) if (ep["h_last"] is not None and not ep["invalid"]) else None,
        "egg_height_final": ep["h_last"], "egg_height_max": ep["h_max"],
        "entity_final_health": ent_final, "entity_loss": {o: 100.0 - v for o, v in ent_final.items()},
        "worst_entity_loss": max(100.0 - v for v in ent_final.values()),
        "code_env_health_final": float(np.mean(list(ent_final.values()))),
        "avg_env_health_over_time": ep["ent_mean_sum"] / ep["ent_mean_n"],
        "primary_safe": primary_safe, "secondary_safe": strict_safe, "compat_safe": compat_safe,
        "primary_safe_success": success and primary_safe, "secondary_safe_success": success and strict_safe,
        "compat_safe_success": success and compat_safe,
        "outcome": ("safe_success" if primary_safe else "damaging_success") if success else
                   ("safe_failure" if primary_safe else "damaging_failure"),
        "max_contact_force_N": float(mt[:, 3].max()),
        "max_contact_force_target_N": float(mt[tgt, 3].max()) if tgt else None,
        "max_impact_force_target_N": float(mt[tgt, 1].max()) if tgt else None,
        "max_qs_force_N": float(mt[:, 2].max()), "first_damage_step": ep["first_damage_step"],
        "link_final_health": dict(zip(ln, H.tolist())), "termination": ep["termination"],
        "timeout_steps": ep["timeout"],
    }


# ───────────────────────────── noise / RNG ─────────────────────────────

def draw_chunk_noise(gen, arm_info):
    """Native per-chunk draw order (rollout_eval_matrix.MatrixPolicy.draw_noise)."""
    import torch
    x0 = torch.randn(1, 8, arm_info["action_dim"], generator=gen)
    z = None
    if arm_info["conditioned"]:
        z = torch.randn(1, arm_info["n_tok"], arm_info["tok_dim"], generator=gen)
    return x0[0].numpy(), (None if z is None else z[0].numpy())


def arm_info(run_dir):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    it = meta.get("intent")
    pc = meta["policy_config"]
    return {"arm": meta["arm"], "run_id": meta["run_id"], "seed": meta["seed"], "run_key": meta.get("run_key"),
            "conditioned": bool(it and it["mode"] == "condition"), "n_tok": it["n_tok"] if it else 0,
            "tok_dim": it["tok_dim"] if it else 0, "intent_nfe": int(it["intent_nfe"]) if it else 0,
            "action_nfe": int(pc["num_inference_steps"]), "action_dim": int(pc["action_dim"])}


# ───────────────────────────── inference ─────────────────────────────

class LocalDecoder:
    """Single-process decoder (tests): same model code path as the evaluator's inference server."""

    def __init__(self, run_dirs, ckpt, device):
        import torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        from rollout_eval_matrix import MatrixPolicy
        self.pols = {a: MatrixPolicy(rd, ckpt=ckpt, device=device) for a, rd in run_dirs.items()}
        self.device = device

    def decode(self, arm, seg, prop, x0, z, fixed_batch=16):
        """seg {k: uint8 [B,T,128,128]}, prop [B,23], x0 [B,8,A], z [B,n,d] or None -> (actions [B,8,A], intents)."""
        return decode_batch(self.pols[arm], seg, prop, x0, z, self.device, fixed_batch)


def decode_batch(pol, seg, prop, x0, z, device, fixed_batch=16):
    """Fixed-shape batches (padded to `fixed_batch` rows) so that a row's result never depends on which other
    requests share its batch (verified bit-identical across positions/companions in harness_verification.md)."""
    import torch
    B = prop.shape[0]
    outs_a, outs_i = [], []
    for s in range(0, B, fixed_batch):
        e = min(B, s + fixed_batch)
        n = e - s

        def pad(x):
            if x is None:
                return None
            x = x[s:e]
            if n < fixed_batch:
                x = np.concatenate([x, np.repeat(x[:1], fixed_batch - n, axis=0)], axis=0)
            return x
        sg = {k: torch.from_numpy(pad(v)).float().to(device) for k, v in seg.items()}
        pr = torch.from_numpy(pad(prop)).float().to(device)
        xx = torch.from_numpy(pad(x0)).float().to(device)
        zz = None if z is None else torch.from_numpy(pad(z)).float().to(device)
        a, it = pol.decode(sg, pr, xx, zz)
        outs_a.append(a[:n])
        outs_i.append(None if it is None else it[:n])
    a = np.concatenate(outs_a, 0)
    it = None if outs_i[0] is None else np.concatenate(outs_i, 0)
    return a, it


def inference_server(run_dirs, ckpt, device, req_q, resp_qs, stats_path=None, fixed_batch=16):
    """Multi-arm server on GPU 0. Requests: (wid, rid, arm, seg{k: uint8[B,T,H,W]}, prop[B,23], x0[B,8,A], z|None);
    responses (rid, actions, intents). Requests of one arm are concatenated and decoded in fixed-shape batches."""
    import queue
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.chdir(RE.OOPSIE)
    from rollout_eval_matrix import MatrixPolicy
    pols = {a: MatrixPolicy(rd, ckpt=ckpt, device=device) for a, rd in run_dirs.items()}
    n_calls, n_items, busy, t0 = 0, 0, 0.0, time.time()
    stop = False
    while not stop:
        item = req_q.get()
        if item is None:
            break
        batch = [item]
        while len(batch) < 64:
            try:
                it = req_q.get(timeout=0.002)
            except queue.Empty:
                break
            if it is None:
                stop = True
                break
            batch.append(it)
        tb = time.time()
        by_arm = {}
        for b in batch:
            by_arm.setdefault(b[2], []).append(b)
        for arm, items in by_arm.items():
            seg = {k: np.concatenate([b[3][k] for b in items]) for k in items[0][3]}
            prop = np.concatenate([b[4] for b in items])
            x0 = np.concatenate([b[5] for b in items])
            z = None if items[0][6] is None else np.concatenate([b[6] for b in items])
            a, its = decode_batch(pols[arm], seg, prop, x0, z, device, fixed_batch)
            o = 0
            for b in items:
                n = b[4].shape[0]
                resp_qs[b[0]].put((b[1], a[o:o + n], None if its is None else its[o:o + n]))
                o += n
            n_items += prop.shape[0]
        busy += time.time() - tb
        n_calls += 1
        if stats_path and n_calls % 200 == 0:
            json.dump({"calls": n_calls, "items": n_items, "gpu_busy_s": busy, "wall_s": time.time() - t0},
                      open(stats_path, "w"))
    if stats_path:
        json.dump({"calls": n_calls, "items": n_items, "gpu_busy_s": busy, "wall_s": time.time() - t0},
                  open(stats_path, "w"))


class RemoteDecoder:
    """Worker-side client. Each request carries a unique id; responses with another id (left over from a killed
    predecessor using the same slot) are discarded."""

    def __init__(self, wid, req_q, resp_q):
        self.wid, self.req_q, self.resp_q = wid, req_q, resp_q
        self.prefix = f"{os.getpid()}:"
        self.n = 0
        self.n_stale = 0

    def decode(self, arm, seg, prop, x0, z):
        self.n += 1
        rid = self.prefix + str(self.n)
        self.req_q.put((self.wid, rid, arm, seg, prop, x0, z))
        while True:
            got = self.resp_q.get()
            if got[0] == rid:
                return got[1], got[2]
            self.n_stale += 1


def policy_chunk(dec, arm, frames, ep, x0, z):
    """Decode one chunk for the current history. Returns (actions [8,A] float64, intent or None, latency_s)."""
    t = time.time()
    seg, prop = frames.model_inputs(ep["hist"])
    a, it = dec.decode(arm, {k: v[None] for k, v in seg.items()}, prop[None], x0[None],
                       None if z is None else z[None])
    return a[0].astype(np.float64), (None if it is None else it[0]), time.time() - t


def run_policy(env, ep, dec, arm, ainfo, frames, lut, gen, until_t=None, render_skip=True, log_chunks=0):
    """Run `arm`'s policy (per-chunk native noise from `gen`) until the episode ends or t reaches until_t
    (a chunk boundary). Returns inference seconds."""
    infer = 0.0
    while not ep["done"] and (until_t is None or ep["t"] < until_t):
        x0, z = draw_chunk_noise(gen, ainfo)
        chunk, it, lat = policy_chunk(dec, arm, frames, ep, x0, z)
        infer += lat
        if log_chunks and len(ep["chunks_log"]) < log_chunks:
            ep["chunks_log"].append({"t": ep["t"], "x0_norm": float(np.linalg.norm(x0)), "chunk0": chunk[0].tolist()})
        exec_chunk(env, ep, chunk, frames, lut, render_skip=render_skip)
    return infer
