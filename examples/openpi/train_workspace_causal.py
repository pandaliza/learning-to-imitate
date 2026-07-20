"""Stage-1 training of the CAUSAL workspace encoder (steer_intent / export design) on LIBERO.

Trains steer_intent's WorkspaceModel (causal AdaLN-Zero temporal encoder + salient-patch decoder) on
precomputed VL grids + salient-patch labels. Unlike examples/openpi/train_workspace.py (which trains the
LOCAL single-frame encoder), this feeds a CAUSAL WINDOW of `window` past+current grids per labeled frame
and supervises the decoder at the window's last step. window=1 degenerates to single-frame (fast smoke).

Reuses the local Hungarian matching + set-reconstruction loss (shape-compatible with the export decoder's
(recon [B,k,D], occ [B,k]) output). No Pi0.5 in the loop -> cheap. Saves workspace_stack.pt (encoder+cfg).

  python examples/openpi/train_workspace_causal.py \
      --vl-cache-dir .../vl_cache_goal_agentview --salient-dir .../salient_goal \
      --out .../workspace_causal_goal --window 8 --backbone-dim 2048 [--lang-table lang.npz]
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from steer_intent.dataset import CausalWindowDataset
from steer_intent.networks.wsm_model import WorkspaceModel, WSMConfig
from workspace_models.config import WorkspaceConfig
from workspace_models.losses import set_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vl-cache-dir", required=True)
    ap.add_argument("--salient-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=8)               # T causal frames (1 = single-frame)
    ap.add_argument("--backbone-dim", type=int, default=2048)      # VL patch dim (recon target dim)
    ap.add_argument("--dim", type=int, default=512)                # workspace token dim
    ap.add_argument("--k-slots", type=int, default=8)              # decoder slots; must >= label max_patches
    ap.add_argument("--max-patches", type=int, default=8)          # salient patches per frame (label side)
    ap.add_argument("--proprio-dim", type=int, default=0)          # 0 -> zeros (smoke); >0 needs a source
    ap.add_argument("--lang-dim", type=int, default=0)             # 0 -> no language; else needs --lang-table
    ap.add_argument("--lang-table", default=None)                  # npz {stem: [lang_dim]} (pool='lang' means)
    ap.add_argument("--max-t", type=int, default=1200)             # time-embedding table (>= longest episode)
    ap.add_argument("--feature-loss-weight", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=250)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    assert args.k_slots >= args.max_patches, "decoder slots must cover the label's max salient patches"

    os.makedirs(args.out, exist_ok=True)
    # export-design encoder/decoder config
    cfg = WSMConfig(dim=args.dim, backbone_dim=args.backbone_dim, k_slots=args.k_slots,
                    proprio_dim=max(args.proprio_dim, 1), lang_dim=max(args.lang_dim, 1),
                    max_t=args.max_t)
    model = WorkspaceModel(cfg).to(args.device).train()
    # local WorkspaceConfig is used ONLY for the matching/loss hyperparameters (dims are irrelevant there)
    lcfg = WorkspaceConfig(feature_loss_weight=args.feature_loss_weight)

    ds = CausalWindowDataset(args.vl_cache_dir, args.salient_dir, args.max_patches, args.window,
                             proprio_dim=cfg.proprio_dim, lang_dim=cfg.lang_dim, lang_table=args.lang_table)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    print(f"[data] {len(ds)} labeled frames, window={args.window}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = torch.optim.lr_scheduler.LinearLR(opt, 1e-3, 1.0, args.warmup_steps)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, args.steps - args.warmup_steps), args.lr * 0.1)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], [args.warmup_steps])

    def save(tag):
        torch.save({"encoder": model.encoder.state_dict(), "decoder": model.decoder.state_dict(),
                    "cfg": vars(cfg), "step": step, "window": args.window},
                   os.path.join(args.out, f"workspace_stack_{tag}.pt"))

    step, ema, best = 0, None, float("inf")
    while step < args.steps:
        for patches, proprio, lang, target, mask in loader:
            patches, proprio, lang = patches.to(args.device), proprio.to(args.device), lang.to(args.device)
            target, mask = target.to(args.device), mask.to(args.device)
            w = model.encode(patches, proprio, lang)             # (B, T, dim) causal
            recon, occ = model.decode(w[:, -1], lang[:, -1])     # supervise the last (labeled) frame
            losses = set_losses(recon, occ, target, mask, lcfg)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            lv = float(losses["total"])
            ema = lv if ema is None else 0.99 * ema + 0.01 * lv
            if step % 100 == 0:
                msg = " ".join(f"{k}={float(v):.4f}" for k, v in losses.items())
                print(f"step {step}: {msg} ema={ema:.4f}", flush=True)
            if step > 0 and step % args.save_every == 0:
                save(str(step))
                if ema < best:
                    best = ema
                    save("best")
            step += 1
            if step >= args.steps:
                break
    save("final")
    print(f"[done] {step} steps, best_ema={best:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
