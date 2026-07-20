"""Precompute per-frame patch-feature grids from an alternative vision encoder (DINOv2 /
DynaFLIP) for the slot-intent variants -- a drop-in replacement for the PaliGemma grids in
examples/openpi/precompute_vl_grids.py.

For every LIBERO-Goal agentview frame we run the chosen encoder and cache its patch-token grid
(n_patches, dim) in fp16, one .npy per (task_file, demo) keyed exactly like the PaliGemma cache
so libero_dataset.py's vl_cache_dir path consumes it unchanged. The slot encoder then runs slot
attention on these grids (slot_vl_dim = encoder dim).

  DINOv2  (facebook/dinov2-base):   last_hidden_state[:, 1:]  -> (256, 768)   [drop CLS]
  DynaFLIP (jlee-larr/dynaflip-base): patch features if exposed (see --encoder dynaflip)

Usage:
  python examples/openpi/precompute_encoder_grids.py --encoder dinov2 \
    --out /data/.../vl_cache_goal_dinov2 [--limit-demos 1]
"""

import argparse
import glob
import os

import h5py
import numpy as np
import torch


def _load_encoder(name, device):
    from transformers import AutoImageProcessor, AutoModel
    if name == "dinov2":
        repo = "facebook/dinov2-base"
        proc = AutoImageProcessor.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).to(device).eval()

        def featurize(frames_uint8):  # list of HWC uint8 -> (B, n_patch, dim) fp32
            inp = proc(images=frames_uint8, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp).last_hidden_state  # (B, 1+n_patch, dim)
            return out[:, 1:]  # drop CLS -> patch grid
        return featurize
    if name == "dynaflip":
        repo = "jlee-larr/dynaflip-base"
        proc = AutoImageProcessor.from_pretrained(repo, trust_remote_code=True)
        model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(device).eval()

        def featurize(frames_uint8):
            inp = proc(images=frames_uint8, return_tensors="pt").to(device)
            with torch.no_grad():
                v = model.vision_outputs(inp["pixel_values"])  # DynaFLIP patch tokens
            return v.last_hidden_state  # (B, num_patches, 768)
        return featurize
    raise ValueError(f"unknown encoder {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["dinov2", "dynaflip"])
    ap.add_argument("--dataset-glob",
                    default="/home/ldahiya/LIBERO/libero/datasets/libero_goal/*_demo.hdf5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit-demos", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    featurize = _load_encoder(args.encoder, args.device)
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(args.dataset_glob))
    assert files, f"no hdf5 matched {args.dataset_glob}"
    print(f"[{args.encoder}] {len(files)} task files -> {args.out}", flush=True)

    for f in files:
        stem = os.path.basename(f).replace("_demo.hdf5", "").replace(".hdf5", "")
        with h5py.File(f, "r") as h:
            demos = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[-1]))
            if args.limit_demos:
                demos = demos[: args.limit_demos]
            for d in demos:
                out_path = os.path.join(args.out, f"{stem}__{d}.npy")
                if os.path.exists(out_path):
                    continue
                agv = h[f"data/{d}/obs/agentview_rgb"][()]  # (T,H,W,3) uint8
                T = agv.shape[0]
                grids = None
                for s in range(0, T, args.batch):
                    e = min(s + args.batch, T)
                    g = featurize([agv[t] for t in range(s, e)]).to(torch.float16).cpu().numpy()
                    if grids is None:
                        grids = np.empty((T, g.shape[1], g.shape[2]), dtype=np.float16)
                    grids[s:e] = g
                np.save(out_path, grids)
                print(f"  {stem}__{d}: {T} frames -> grid{grids.shape} {grids.nbytes/1e6:.0f}MB", flush=True)
    print(f"[{args.encoder}] done", flush=True)


if __name__ == "__main__":
    main()
