"""Stage 3: one small deterministic convolutional autoencoder per hand on the hand-only renders.

- input: hand_renders_{size}.hdf5 from render_hands.py (uint8 grayscale, demo_index per frame)
- split by demonstration (val_frac of demos held out for AE validation; all demos are policy-training demos)
- pixel MSE, Adam, checkpoint selected on validation reconstruction MSE
- reports: AE vs train-mean-image baseline (full-frame and foreground-only MSE), per-latent std, PCA of the
  latent, linear predictability of the latent from the 12 joints and from their first principal component
  (does the latent carry more than open/close?), reconstruction panel on held-out demos
- writes: ae_{side}/best.pt, metrics.json, recon_panel.png, latents.npy (encoding of every frame, best ckpt)

Usage: python train_hand_ae.py --renders R.hdf5 --out_dir OUT --side right|left [--latent 16 --epochs 30]
"""

import argparse
import hashlib
import json
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

FG_THRESH = 8 / 255.0  # pixels above this in the original count as hand (background renders as 0)


class ConvAE(nn.Module):
    def __init__(self, latent=16, size=64, ch=(32, 64, 128, 256)):
        super().__init__()
        enc, c_in = [], 1
        for c in ch:
            enc += [nn.Conv2d(c_in, c, 4, 2, 1), nn.SiLU()]
            c_in = c
        self.enc = nn.Sequential(*enc, nn.Flatten())
        self.s, self.c = size // 2 ** len(ch), c_in
        self.fc_e = nn.Linear(self.c * self.s * self.s, latent)
        self.fc_d = nn.Linear(latent, self.c * self.s * self.s)
        dec, chs = [], list(ch[::-1])
        for i, c in enumerate(chs):
            last = i + 1 == len(chs)
            dec += [nn.ConvTranspose2d(c, 1 if last else chs[i + 1], 4, 2, 1)] + ([] if last else [nn.SiLU()])
        self.dec = nn.Sequential(*dec)
        self.cfg = dict(latent=latent, size=size, ch=list(ch))

    def encode(self, x):
        return self.fc_e(self.enc(x))

    def decode(self, z):
        return torch.sigmoid(self.dec(self.fc_d(z).view(-1, self.c, self.s, self.s)))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def to_float(u8):  # uint8 (B,H,W) on device -> float (B,1,H,W) in [0,1]
    return u8.float().div_(255.0).unsqueeze(1)


@torch.no_grad()
def eval_recon(model, imgs, bs=1024, mean_img=None):
    """Full-frame and foreground-only MSE of the AE (and of a constant mean image if given)."""
    tot = dict(mse=0.0, fg_mse=0.0, base_mse=0.0, base_fg_mse=0.0)
    n = fg_n = 0
    for i in range(0, len(imgs), bs):
        x = to_float(imgs[i:i + bs])
        xh, _ = model(x)
        fg = x > FG_THRESH
        tot["mse"] += ((xh - x) ** 2).sum().item()
        tot["fg_mse"] += (((xh - x) ** 2) * fg).sum().item()
        if mean_img is not None:
            tot["base_mse"] += ((mean_img - x) ** 2).sum().item()
            tot["base_fg_mse"] += (((mean_img - x) ** 2) * fg).sum().item()
        n += x.numel()
        fg_n += fg.sum().item()
    return {k: v / (fg_n if "fg" in k else n) for k, v in tot.items()}


@torch.no_grad()
def encode_all(model, imgs, bs=2048):
    return torch.cat([model.encode(to_float(imgs[i:i + bs])) for i in range(0, len(imgs), bs)]).cpu().numpy()


def ridge_r2(x, y, x_val, y_val, lam=1e-3):
    """Total R^2 of a ridge regression x -> y fit on train, scored on val (both standardized by train)."""
    mx, sx = x.mean(0), x.std(0) + 1e-8
    my, sy = y.mean(0), y.std(0) + 1e-8
    xs, ys = (x - mx) / sx, (y - my) / sy
    xs1 = np.c_[xs, np.ones(len(xs))]
    w = np.linalg.solve(xs1.T @ xs1 + lam * np.eye(xs1.shape[1]), xs1.T @ ys)
    xv = np.c_[(x_val - mx) / sx, np.ones(len(x_val))]
    yv = (y_val - my) / sy
    res = ((xv @ w - yv) ** 2).sum()
    return float(1 - res / ((yv - yv.mean(0)) ** 2).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--renders", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--side", required=True, choices=["right", "left"])
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    out = os.path.join(args.out_dir, f"ae_{args.side}")
    os.makedirs(out, exist_ok=True)

    with h5py.File(args.renders, "r") as f:
        imgs_np = f[args.side][()]
        qpos = f[f"qpos_{args.side}"][()]
        demo_index = f["demo_index"][()]
        size = imgs_np.shape[-1]
        render_meta = f.attrs["render_meta"]
    n_demos = demo_index.max() + 1
    rng = np.random.default_rng(args.seed)
    val_demos = np.sort(rng.choice(n_demos, int(round(args.val_frac * n_demos)), replace=False))
    is_val = np.isin(demo_index, val_demos)
    tr_idx, va_idx = np.where(~is_val)[0], np.where(is_val)[0]
    imgs = torch.from_numpy(imgs_np).to(dev)  # uint8 on GPU (~1.2 GB for 300k x 64 x 64)
    tr, va = imgs[tr_idx], imgs[va_idx]
    mean_img = to_float(tr[:50000]).mean(0, keepdim=True)
    for i in range(50000, len(tr), 50000):  # running mean over the whole train split
        chunk = to_float(tr[i:i + 50000])
        mean_img = mean_img + (chunk.mean(0, keepdim=True) - mean_img) * len(chunk) / (i + len(chunk))
    print(f"[{args.side}] train {len(tr)} frames / {n_demos - len(val_demos)} demos, val {len(va)} / {len(val_demos)} demos")

    model = ConvAE(args.latent, size).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = args.epochs * (len(tr) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best, hist, t0 = None, [], time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        tl = 0.0
        for i in range(0, len(tr) - args.batch + 1, args.batch):
            x = to_float(tr[perm[i:i + args.batch]])
            xh, _ = model(x)
            loss = ((xh - x) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item()
        model.eval()
        ev = eval_recon(model, va, mean_img=mean_img)
        hist.append(dict(epoch=ep, train_mse=tl / (len(tr) // args.batch), **ev))
        print(f"[{args.side}] ep {ep} train {hist[-1]['train_mse']:.5f} val {ev['mse']:.5f} fg {ev['fg_mse']:.4f} "
              f"(mean-img {ev['base_mse']:.5f} / fg {ev['base_fg_mse']:.4f}) {time.time() - t0:.0f}s", flush=True)
        if best is None or ev["mse"] < best["val"]["mse"]:
            best = dict(epoch=ep, val=ev, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best["state"])
    model.eval()
    ckpt = os.path.join(out, "best.pt")
    torch.save(dict(state_dict=best["state"], cfg=model.cfg, side=args.side, epoch=best["epoch"], val=best["val"],
                    val_demos=val_demos.tolist(), args=vars(args), renders=os.path.abspath(args.renders),
                    render_meta=render_meta), ckpt)
    ckpt_md5 = hashlib.md5(open(ckpt, "rb").read()).hexdigest()

    # ---- diagnostics on the best checkpoint
    z_all = encode_all(model, imgs)
    z_tr, z_va = z_all[tr_idx], z_all[va_idx]
    q_tr, q_va = qpos[tr_idx], qpos[va_idx]
    ev = np.linalg.svd(z_va - z_va.mean(0), compute_uv=False) ** 2
    u = np.linalg.svd(q_tr - q_tr.mean(0), full_matrices=False)[2][0]  # first principal axis of the joints
    pc1_tr, pc1_va = ((q_tr - q_tr.mean(0)) @ u)[:, None], ((q_va - q_tr.mean(0)) @ u)[:, None]
    diag = dict(
        n_params=n_params, best_epoch=best["epoch"], val=best["val"],
        latent_std=z_va.std(0).round(4).tolist(),
        latent_pca_cum_var=np.cumsum(ev / ev.sum()).round(4).tolist(),
        r2_joints_to_latent=ridge_r2(q_tr, z_tr, q_va, z_va),
        r2_joint_pc1_to_latent=ridge_r2(pc1_tr, z_tr, pc1_va, z_va),
        r2_latent_to_joints=ridge_r2(z_tr, q_tr, z_va, q_va),
        r2_joint_pc1_to_joints=ridge_r2(pc1_tr, q_tr, pc1_va, q_va),
        ckpt_md5=ckpt_md5, history=hist,
    )
    json.dump(diag, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    np.save(os.path.join(out, "latents.npy"), z_all.astype(np.float32))
    json.dump(dict(ckpt=ckpt, ckpt_md5=ckpt_md5, renders=os.path.abspath(args.renders), n=len(z_all), latent=args.latent),
              open(os.path.join(out, "latents_provenance.json"), "w"), indent=1)

    # reconstruction panel: 16 held-out frames spanning closure; rows = original / reconstruction / |diff|
    order = va_idx[np.argsort(qpos[va_idx].sum(1))]
    pick = order[np.linspace(0, len(order) - 1, 16).astype(int)]
    with torch.no_grad():
        x = to_float(imgs[pick])
        xh, _ = model(x)
        base = mean_img.expand_as(x)
    rows = [x, xh, (xh - x).abs() * 4, base]
    up = 3
    panel = np.concatenate([np.concatenate([np.kron((im[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                                                    np.ones((up, up), np.uint8)) for im in r], 1) for r in rows], 0)
    Image.fromarray(panel).save(os.path.join(out, "recon_panel.png"))
    print(f"[{args.side}] best epoch {best['epoch']} val mse {best['val']['mse']:.5f} fg {best['val']['fg_mse']:.4f} | "
          f"mean-img {best['val']['base_mse']:.5f} fg {best['val']['base_fg_mse']:.4f} | "
          f"R2 joints->z {diag['r2_joints_to_latent']:.3f} pc1->z {diag['r2_joint_pc1_to_latent']:.3f} "
          f"z->joints {diag['r2_latent_to_joints']:.3f} pc1->joints {diag['r2_joint_pc1_to_joints']:.3f}")


if __name__ == "__main__":
    main()
