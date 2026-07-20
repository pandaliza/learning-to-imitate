"""Stage-1 training of the workspace-model intent (current-frame).

Trains WorkspaceModel (encoder + DETR set-reconstruction decoder) on precomputed DINO grids +
VLM salient patch sets (no Pi0.5 in the loop -> cheap). Saves workspace_stack.pt (encoder + config).
Stage-2 then freezes the encoder and conditions Pi0.5 on the workspace token (M3-style).

  python examples/openpi/train_workspace.py \
      --vl-cache-dir .../vl_cache_goal_dinov2 --salient-dir .../workspace_salient_goal \
      --out .../workspace_stack_goal
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from workspace_models import WorkspaceModel, WorkspaceConfig
from workspace_models.dataset import WorkspaceGridDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vl-cache-dir", required=True)
    ap.add_argument("--salient-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feat-dim", type=int, default=768)
    ap.add_argument("--num-patches", type=int, default=196)
    ap.add_argument("--max-patches", type=int, default=8)
    ap.add_argument("--feature-loss-weight", type=float, default=1.0)   # Table 3 lambda_1
    ap.add_argument("--steps", type=int, default=5000)                  # Table 3 (CubeDrop)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=250)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = WorkspaceConfig(feat_dim=args.feat_dim, num_patches=args.num_patches,
                          max_patches=args.max_patches, feature_loss_weight=args.feature_loss_weight)
    model = WorkspaceModel(cfg).to(args.device).train()

    ds = WorkspaceGridDataset(args.vl_cache_dir, args.salient_dir, cfg.max_patches)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    print(f"[data] {len(ds)} labeled frames", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # linear warmup + cosine decay (Table 3: "linear warmup cosine decay")
    warm = torch.optim.lr_scheduler.LinearLR(opt, 1e-3, 1.0, args.warmup_steps)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, args.steps - args.warmup_steps), args.lr * 0.1)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], [args.warmup_steps])

    def save(tag):
        torch.save({"encoder": model.encoder.state_dict(), "cfg": vars(cfg), "step": step},
                   os.path.join(args.out, f"workspace_stack_{tag}.pt"))

    step, ema, best = 0, None, float("inf")
    while step < args.steps:
        for grid, target, mask in loader:
            grid, target, mask = grid.to(args.device), target.to(args.device), mask.to(args.device)
            _, losses = model(grid, target, mask)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # Table 3: grad clip 1.0
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
