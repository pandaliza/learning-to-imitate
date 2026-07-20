"""Precompute per-demo workspace latents w.npy from a FROZEN causal encoder (stage-1 output).

For each demo's VL grid ({stem}__{demo}.npy, (T,N,D)), run the frozen causal WorkspaceEncoder over the
full sequence -> w [T, dim], and save {out}/{stem}__{demo}.npy (w [T,dim] fp16). w is FROZEN and the
encoder never runs at stage-2 train time. The stage-2 LiberoDataset injects these per-demo arrays under
a "w" key (exactly like the VL-grid cache) so the SequenceSampler WINDOWS them with the frames: w_t is
the windowed w at the current frame and w_{t+1} the next frame -- because w was encoded CAUSALLY per
frame, the windowed slice needs no frame_indices/next_at bookkeeping.

  python examples/openpi/precompute_w.py \
      --vl-cache-dir .../vl_cache_goal_agentview --wsm-stack .../workspace_stack_best.pt \
      --lang-table .../lang_table_goal.npz --out .../w_cache_goal
"""
import argparse
import glob
import os

import numpy as np
import torch

from steer_intent.networks.wsm_model import WorkspaceModel, WSMConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vl-cache-dir", required=True)
    ap.add_argument("--wsm-stack", required=True)          # stage-1 checkpoint (encoder + cfg)
    ap.add_argument("--lang-table", default=None)          # {stem: [lang_dim]} npz; zeros if omitted
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-frames", type=int, default=512)  # chunk long demos to bound memory
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ck = torch.load(args.wsm_stack, map_location=args.device, weights_only=False)
    cfg = WSMConfig(**ck["cfg"])
    model = WorkspaceModel(cfg).to(args.device).eval()
    model.encoder.load_state_dict(ck["encoder"])
    for p in model.parameters():
        p.requires_grad_(False)
    lang_table = dict(np.load(args.lang_table)) if args.lang_table else {}
    print(f"[precompute_w] encoder step {ck.get('step')} dim={cfg.dim} lang_dim={cfg.lang_dim}", flush=True)

    files = sorted(glob.glob(os.path.join(args.vl_cache_dir, "*.npy")))
    for i, gp in enumerate(files):
        key = os.path.basename(gp)[:-4]                    # {stem}__{demo}
        stem = key.split("__")[0]
        out_path = os.path.join(args.out, key + ".npy")
        if os.path.exists(out_path):
            continue
        g = np.load(gp, mmap_mode="r")                     # (T, N, D)
        T = g.shape[0]
        lg = lang_table.get(stem)
        # The causal encoder emits w_t from frames <= t, so a single full-sequence pass gives every w_t
        # consistently (chunking would break causal context) -> run the whole demo in one forward.
        patches = torch.from_numpy(np.asarray(g, dtype=np.float32)).unsqueeze(0).to(args.device)
        proprio = torch.zeros(1, T, cfg.proprio_dim, device=args.device)
        lang = (torch.from_numpy(np.broadcast_to(lg, (T, cfg.lang_dim)).astype(np.float32)).unsqueeze(0).to(args.device)
                if lg is not None else torch.zeros(1, T, cfg.lang_dim, device=args.device))
        with torch.no_grad():
            w = model.encode(patches, proprio, lang)[0].to(torch.float16).cpu().numpy()   # (T, dim)
        np.save(out_path, w)
        if i % 20 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] {key}: w{w.shape}", flush=True)
    print(f"[done] {len(files)} demos -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
