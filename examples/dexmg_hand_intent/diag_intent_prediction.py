"""Diagnostics 1-2 for the rendered-hand intent arm (CPU-only, no simulator).

1. Predictability: the policy's intent predictor (MLP on current 38-D proprio, taken from a BC-RNN
   checkpoint) vs the sidecar target `hand_intent` (128-D, standardized), per future offset k, against
     - constant-mean predictor (outputs the training mean = 0 in standardized units)
     - copy-current: the *current* rendered latents repeated for every k (needs the renders; the policy
       never sees these, so this only measures how much the hands change over k steps)
     - ridge regression from the same 38-D proprio (fit on the AE-train demos)
   Scored on the AE's validation demos. NOTE: the policy (and its predictor) trained on ALL demos, so
   for the predictor these demos are not held out; ridge is fit without them.
2. Decodes: for a few validation frames around grasp onsets, actual future renders at t+k vs the AE
   decode of the true future latent vs the AE decode of the predicted latent, per hand.

Usage: python diag_intent_prediction.py --sidecar S.hdf5 --renders R.hdf5 --ae_dir AE --ckpt C1.pth [C2 ...]
                                        --out_dir OUT
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from train_hand_ae import ConvAE  # noqa: E402

PFX = "policy.nets.encoder."


def load_predictor(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]["nets"]
    w = [(sd[f"{PFX}predictor.{i}.weight"].numpy(), sd[f"{PFX}predictor.{i}.bias"].numpy()) for i in (0, 2, 4)]
    lo, scale = sd[f"{PFX}lo"].numpy(), sd[f"{PFX}scale"].numpy()

    def f(x):
        h = 2 * (x - lo) / scale - 1
        for j, (W, b) in enumerate(w):
            h = h @ W.T + b
            if j < 2:
                h = h / (1 + np.exp(-h))  # SiLU
        return h

    return f


def mse_by_offset(pred, tgt, K):
    d = ((pred - tgt) ** 2).reshape(len(tgt), K, -1)
    return d.mean(axis=(0, 2))


def r2_by_offset(pred, tgt, K):
    p, t = pred.reshape(len(tgt), K, -1), tgt.reshape(len(tgt), K, -1)
    return 1 - ((p - t) ** 2).sum(axis=(0, 2)) / ((t - t.mean(0)) ** 2).sum(axis=(0, 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sidecar", required=True)
    p.add_argument("--renders", required=True)
    p.add_argument("--ae_dir", required=True)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_decode", type=int, default=4, help="grasp-onset frames per hand in the decode panel")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    ae, val_demos = {}, None
    for side in ("left", "right"):
        ck = torch.load(os.path.join(args.ae_dir, f"ae_{side}", "best.pt"), map_location="cpu", weights_only=False)
        m = ConvAE(**{k: v for k, v in ck["cfg"].items()})
        m.load_state_dict(ck["state_dict"])
        ae[side] = m.eval()
        vd = np.array(ck["val_demos"])
        assert val_demos is None or np.array_equal(val_demos, vd), "AEs used different val splits"
        val_demos = vd

    with h5py.File(args.sidecar, "r") as f:
        a = f["data"].attrs
        lowdim_keys = [k.decode() if isinstance(k, bytes) else str(k) for k in a["lowdim_keys"]]
        mean, std = np.asarray(a["hand_intent_mean"]), np.asarray(a["hand_intent_std"])
        layout = json.loads(a["hand_intent_layout"])
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        X, Y, D, T = [], [], [], []
        for i, d in enumerate(demos):
            g = f["data"][d]
            X.append(np.concatenate([g["obs"][k][()] for k in lowdim_keys], -1).astype(np.float32))
            Y.append(g["hand_intent"][()])
            D.append(np.full(len(Y[-1]), i))
            T.append(np.arange(len(Y[-1])))
    X, Y, D, T = map(np.concatenate, (X, Y, D, T))
    K, L = len(layout["waypoints"]), layout["latent_per_hand"]
    va = np.isin(D, val_demos)
    tr = ~va
    print(f"{len(demos)} demos, {len(Y)} frames; eval on {va.sum()} frames of {len(val_demos)} AE-val demos; K={K}")

    # current latents (copy-current baseline), standardized with each offset's stats
    z_now = np.concatenate([np.load(os.path.join(args.ae_dir, f"ae_{s}", "latents.npy")) for s in ("left", "right")], -1)
    assert len(z_now) == len(Y)
    copy = (np.tile(z_now, (1, K)) - mean) / std

    # ridge from proprio, fit on AE-train demos
    mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = np.c_[(X - mx) / sx, np.ones(len(X))]
    W = np.linalg.solve(Xs[tr].T @ Xs[tr] + 1e-2 * np.eye(Xs.shape[1]), Xs[tr].T @ Y[tr])
    ridge = Xs @ W

    rows = {"constant_mean": np.zeros_like(Y[va]), "copy_current_latent": copy[va], "ridge_proprio": ridge[va]}
    preds = {}
    for c in args.ckpt:
        name = "/".join(c.split("/")[-5:-4] + [os.path.basename(c).split("_Two")[0].replace(".pth", "")])
        preds[name] = load_predictor(c)(X).astype(np.float32)
        rows[f"policy_predictor:{name}"] = preds[name][va]

    res = {}
    print(f"\n{'predictor':58s} {'MSE':>6s} {'R2':>6s} | MSE at k=" + " ".join(f"{k:>5d}" for k in layout["waypoints"]))
    for n, pr in rows.items():
        m = float(((pr - Y[va]) ** 2).mean())
        r2 = float(1 - ((pr - Y[va]) ** 2).sum() / ((Y[va] - Y[va].mean(0)) ** 2).sum())
        mk, rk = mse_by_offset(pr, Y[va], K), r2_by_offset(pr, Y[va], K)
        res[n] = dict(mse=m, r2=r2, mse_by_k=mk.tolist(), r2_by_k=rk.tolist())
        print(f"{n:58s} {m:6.3f} {r2:6.3f} | " + "         ".join(f"{x:.3f}" for x in mk)[:0] +
              " ".join(f"{x:5.3f}" for x in mk))
    res["_meta"] = dict(eval_demos=val_demos.tolist(), n_eval_frames=int(va.sum()),
                        note="policy predictor trained on all demos (incl. these); ridge fit on the rest")
    json.dump(res, open(os.path.join(args.out_dir, "intent_prediction_metrics.json"), "w"), indent=1)

    # ---- decode panel (first checkpoint): frames at grasp onsets in val demos
    name0 = list(preds)[0]
    pred0 = preds[name0]
    with h5py.File(args.renders, "r") as r:
        q = {s: r[f"qpos_{s}"][()] for s in ("left", "right")}
        imgs = {s: r[s] for s in ("left", "right")}
        up, tiles_rows, labels = 2, [], []
        for hi, side in enumerate(("left", "right")):
            closure = q[side].sum(1)
            vel = np.zeros_like(closure)
            vel[1:] = closure[1:] - closure[:-1]
            vel[np.r_[True, D[1:] != D[:-1]]] = 0
            cand = np.where(va & (T >= 4))[0]
            # strongest closing moments, one per demo, spread across demos
            order = cand[np.argsort(-vel[cand])]
            picked, seen = [], set()
            for i in order:
                if D[i] not in seen:
                    picked.append(i - 4)  # start 4 steps before the fastest closure
                    seen.add(D[i])
                if len(picked) == args.n_decode:
                    break
            for i in picked:
                row_real, row_true, row_pred = [], [], []
                for j, k in enumerate(layout["waypoints"]):
                    t_idx = i + k if (i + k < len(D) and D[i + k] == D[i]) else np.where(D == D[i])[0][-1]
                    row_real.append(imgs[side][t_idx] / 255.0)
                    sl = slice(j * 2 * L + hi * L, j * 2 * L + (hi + 1) * L)
                    for z, row in ((Y[i, sl] * std[sl] + mean[sl], row_true), (pred0[i, sl] * std[sl] + mean[sl], row_pred)):
                        row.append(ae[side].decode(torch.from_numpy(z[None]).float())[0, 0].numpy())
                tiles_rows.append(np.concatenate([np.concatenate(rw, 1) for rw in (row_real, row_true, row_pred)], 0))
                labels.append(f"{side} demo {D[i]} t={T[i]}")
    S = tiles_rows[0].shape[0] // 3
    gap = 6
    H = len(tiles_rows) * (3 * S + gap)
    canvas = np.ones((H, tiles_rows[0].shape[1] + 0)) * 0.25
    for n, tr_ in enumerate(tiles_rows):
        canvas[n * (3 * S + gap): n * (3 * S + gap) + 3 * S] = tr_
    im = Image.fromarray((np.kron(canvas, np.ones((up, up))) * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
    pad = 230
    out = Image.new("RGB", (im.width + pad, im.height + 24), (255, 255, 255))
    out.paste(im, (pad, 24))
    dr = ImageDraw.Draw(out)
    for j, k in enumerate(layout["waypoints"]):
        dr.text((pad + (j * S + S // 2 - 10) * up, 6), f"t+{k}", fill=(0, 0, 0))
    for n, lab in enumerate(labels):
        y0 = 24 + n * (3 * S + gap) * up
        dr.text((4, y0 + 4), lab, fill=(0, 0, 0))
        for rr, rl in enumerate(("actual render", "decode(true z)", "decode(predicted z)")):
            dr.text((4, y0 + (rr * S + S // 2) * up), rl, fill=(80, 80, 80))
    out.save(os.path.join(args.out_dir, "decode_panel.png"))
    print(f"\nwrote {args.out_dir}/intent_prediction_metrics.json and decode_panel.png (predictor: {name0})")


if __name__ == "__main__":
    main()
