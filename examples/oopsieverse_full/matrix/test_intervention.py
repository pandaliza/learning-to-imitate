"""Verification 2: intent interventions on short-trained intent-arm models (dev split windows).

For each run (checkpoint) and N dev windows, with the ACTION noise held fixed:
  1. Diversity: K intent samples from K distinct N(0,I) draws -> all distinct (pairwise L2 > 0); the same draw twice
     -> identical intent (determinism of the Euler sampler given noise).
  2. Resampling: decoded-action spread across the K sampled intents (fixed action noise), compared with the spread
     across K action noises at a fixed intent.
  3. Clamping (directional): clamp the intent to GT, then to GT with the future EEF positions displaced by
     d (3 cm at k=H, scaled k/H along the path; +-x, +-y, +-z in WORLD frame, converted to each target's frame).
     The decoded action change is mapped to a predicted world EEF displacement through a least-squares map
     W (sum of chunk arm-delta commands -> realised 8-step EEF world displacement) fitted on TRAIN windows.
     Report cos(predicted displacement change, d) and gain |change| / |d|.
  4. Gripper clamp (I-PATH-G, AUX): all gripper tokens set to the train max vs train min; decoded gripper command
     change (chunk mean, raw units).
  AUX-PATH-G must show zero action change under any intent clamp (no conditioning path).
"""

import contextlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
OOPSIE = "/edrive1/lizad/oopsieverse_full_study/src/oopsieverse"
sys.path.insert(0, OOPSIE)
os.chdir(OOPSIE)

import intent_targets as IT  # noqa: E402
from data import MatrixPlaybackDataset  # noqa: E402
from models import IntentCFMPolicy  # noqa: E402
from learning.models import PolicyConfig  # noqa: E402

DEV = os.environ.get("INTERVENTION_DEVICE", "cuda")
REPO = "/scr/lizad/personal/learning-to-imitate"
OUT = os.path.join(REPO, "outputs/oopsieverse_full_study/matrix/verification")


def load(run_dir, ckpt):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    pc = dict(meta["policy_config"])
    pc["action_min"] = np.asarray(pc["action_min"], np.float32)
    pc["action_max"] = np.asarray(pc["action_max"], np.float32)
    it = meta["intent"]
    m = IntentCFMPolicy(PolicyConfig(**pc), it["n_tok"], it["tok_dim"], it["mode"], intent_seed=meta["seed"])
    m.load_state_dict(torch.load(os.path.join(run_dir, "ckpt", ckpt), map_location="cpu")["policy_state_dict"])
    return m.to(DEV).eval(), meta


_WMAP = {}


def world_map(train_hdf5):
    import h5py
    if train_hdf5 in _WMAP:
        return _WMAP[train_hdf5]
    X, Y = [], []
    f = h5py.File(train_hdf5, "r")
    for d in f["data"]:
        p, a = f["data"][d]["obs/proprio"][:].astype(np.float64), f["data"][d]["action"][:].astype(np.float64)
        ts = IT.sample_indices(len(a))
        X.append(np.stack([a[t:t + 8, :3].sum(0) for t in ts]))
        Y.append(p[ts + 8, 14:17] - p[ts, 14:17])
    X, Y = np.concatenate(X), np.concatenate(Y)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    r2 = 1 - ((Y - X @ W) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    _WMAP[train_hdf5] = (W, float(r2))
    return _WMAP[train_hdf5]


def displaced(name, raw, prop, d_world):
    """raw intent [N, n_tok, dim] (un-normalised); d_world [3] displacement at k=H. Returns displaced raw intent."""
    out = raw.copy()
    if name == "imean":
        # world-frame mean of future positions: mean_k (k/H) d = d * (H+1)/(2H)
        out[:, 0, :3] += d_world * (IT.H + 1) / (2 * IT.H)
        return out
    Rc = IT.quat_xyzw_to_mat(prop[:, 17:21])
    d_eef = np.einsum("nji,j->ni", Rc, d_world)  # R^T d
    if name == "cmean":
        out[:, 0, :3] += d_eef * (IT.H + 1) / (2 * IT.H)
    else:
        for j, k in enumerate(IT.PATH_KS):
            out[:, j, :3] += d_eef * k / IT.H
    return out


def run(run_dir, ckpt, n=64, K=8, seed=0):
    model, meta = load(run_dir, ckpt)
    name = meta["intent"]["name"]
    norm = IT.MinMax.from_dict(meta["intent"]["normaliser"])
    kw = dict(frame_stack=2, action_chunk_size=8, seg_img_size=(128, 128), normalize_action=True,
              objects_of_interest=meta["trainer_config"]["objects_of_interest"], obs_keys=meta["trainer_config"]["obs_keys"])
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        dev = MatrixPlaybackDataset(data_path=meta["dev_hdf5"], intent_name=name, **kw)
    W, r2 = world_map(meta["train_hdf5"])
    dev.action_min = np.asarray(meta["action_min"], np.float32)
    dev.action_max = np.asarray(meta["action_max"], np.float32)
    dev.setup_intent(norm=norm)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dev), size=n, replace=False)
    batch = [dev[i] for i in idx]
    seg = {k: torch.stack([b["obs"]["extero"][k] for b in batch]).to(DEV) for k in batch[0]["obs"]["extero"]}
    state = torch.stack([b["obs"]["proprio"] for b in batch]).to(DEV)
    s23, it_gt = model.split_state(state)
    prop = s23.cpu().numpy().astype(np.float64)
    g = torch.Generator(device=DEV).manual_seed(seed)
    x0 = torch.randn(n, 8, model.config.action_dim, device=DEV, generator=g)
    res = {"run": run_dir, "ckpt": ckpt, "arm": meta["arm"], "intent": name, "mode": model.mode, "n_windows": n,
           "world_map_r2": r2}
    with torch.no_grad():
        obs_tok = model.obs_tokens(seg, s23)
        zs = [torch.randn(it_gt.shape, device=DEV, generator=g) for _ in range(K)]
        ints = [model.sample_intent(obs_tok, z) for z in zs]
        again = model.sample_intent(obs_tok, zs[0])
        pd = [float((ints[i] - ints[j]).flatten(1).norm(dim=1).min()) for i in range(K) for j in range(i + 1, K)]
        pd_mean = [float((ints[i] - ints[j]).flatten(1).norm(dim=1).mean()) for i in range(K) for j in range(i + 1, K)]
        res["diversity"] = {"K": K, "min_pairwise_l2_over_windows": min(pd), "mean_pairwise_l2": float(np.mean(pd_mean)),
                            "all_distinct": min(pd) > 0, "same_noise_identical": bool(torch.equal(again, ints[0])),
                            "sample_vs_gt_mse": float(torch.stack([(i - it_gt).pow(2).mean() for i in ints]).mean()),
                            "gt_intent_std_across_windows": float(it_gt.std(0).mean())}
        cond = model.mode == "condition"
        acts_int = [model.generate(seg, s23, init_noise=x0, intent=i if cond else None)[0] for i in ints]
        xs = [torch.randn(x0.shape, device=DEV, generator=g) for _ in range(K)]
        acts_noise = [model.generate(seg, s23, init_noise=x, intent=ints[0] if cond else None)[0] for x in xs]

        def spread(A):
            return float(np.mean([(A[i] - A[j])[..., :7].flatten(1).norm(dim=1).mean().item() for i in range(K) for j in range(i + 1, K)]))
        res["resampling"] = {"action_spread_across_intent_samples_fixed_action_noise": spread(acts_int),
                             "action_spread_across_action_noise_fixed_intent": spread(acts_noise)}
        # directional clamp
        raw_gt = norm.unnormalize(it_gt.cpu().numpy().astype(np.float64))
        a_gt = model.generate(seg, s23, init_noise=x0, intent=it_gt if cond else None)[0].cpu().numpy().astype(np.float64)
        dirs = {}
        src = name if name != "ipathg" else "ipath"
        for lab, v in {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1]}.items():
            d = 0.03 * np.asarray(v, np.float64)
            raw_d = displaced(src, raw_gt, prop, d)
            it_d = torch.from_numpy(norm.normalize(raw_d).astype(np.float32)).to(DEV)
            a_d = model.generate(seg, s23, init_noise=x0, intent=it_d if cond else None)[0].cpu().numpy().astype(np.float64)
            dy = (a_d[:, :, :3].sum(1) - a_gt[:, :, :3].sum(1)) @ W  # predicted world displacement change
            cos = (dy @ d) / (np.linalg.norm(dy, axis=1) * np.linalg.norm(d) + 1e-12)
            dirs[lab] = {"mean_cos": float(cos.mean()), "frac_cos_pos": float((cos > 0).mean()),
                         "gain": float(np.linalg.norm(dy, axis=1).mean() / np.linalg.norm(d)),
                         "max_abs_action_change": float(np.abs(a_d - a_gt).max())}
        res["clamp_direction"] = dirs
        res["clamp_direction_mean_cos"] = float(np.mean([v["mean_cos"] for v in dirs.values()]))
        if name == "ipathg":
            hi, lo = raw_gt.copy(), raw_gt.copy()
            hi[:, :, 9] = norm.max[:, 9]
            lo[:, :, 9] = norm.min[:, 9]
            a_hi = model.generate(seg, s23, init_noise=x0, intent=torch.from_numpy(norm.normalize(hi).astype(np.float32)).to(DEV) if cond else None)[0]
            a_lo = model.generate(seg, s23, init_noise=x0, intent=torch.from_numpy(norm.normalize(lo).astype(np.float32)).to(DEV) if cond else None)[0]
            dg = (a_hi[:, :, 6] - a_lo[:, :, 6]).mean(1).cpu().numpy()
            res["gripper_clamp"] = {"train_grip_min": norm.min[:, 9].tolist(), "train_grip_max": norm.max[:, 9].tolist(),
                                    "mean_decoded_grip_change_hi_minus_lo": float(dg.mean()), "frac_positive": float((dg > 0).mean())}
    return res


if __name__ == "__main__":
    ckpt = sys.argv[1]
    runs = sys.argv[2:]
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    out = [run(r, ckpt) for r in runs]
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, os.environ.get("INTERVENTION_OUT", "intervention.json")), "w"), indent=1)
    print(json.dumps(out, indent=1))
