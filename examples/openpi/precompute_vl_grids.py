"""Precompute frozen pi05_base VL agentview token grids for the VL-grounded slot-intent
(decoupled) variant.

For every LIBERO-Goal frame we run the FROZEN pi05_base PaliGemma prefix and cache the
agentview (base_0_rgb) post-PaliGemma token grid (n_tokens, width) in fp16. The slot
encoder consumes these grids as its z* input (replacing the from-scratch ResNet18); the
flow-map generator uses their mean as the current-obs conditioning. See slot_attention.py
(vl_input path) and libero_dataset.py (vl_cache_dir).

The grids are agentview-only (base_0_rgb = the first per-image token block of the prefix),
matching the existing slot encoder's slot_image_key="agentview_rgb". They are still
contextualised by the wrist view + language prompt via the prefix's full attention.

Loading mirrors train_pi05_cotrain.py exactly: load pi05_base weights (no LoRA keys) into a
LoRA config with strict=False so the LoRA adapters init to B=0 -> VL == pure base VL; run
fp32 (pi05_base overflows bf16). One .npy per (task_file, demo) -> resumable.

Usage (smoke test on 1 demo/task):
  python examples/openpi/precompute_vl_grids.py --pi05-weights /data/.../pi05_base_pytorch \
    --out /data/group_data/.../vl_cache_goal_agentview --limit-demos 1
Full run: drop --limit-demos.
"""

import argparse
import glob
import os

import h5py
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi05-config", default="pi05_base_nointent",
                    help="config for architecture+transforms (no-intent -> transform needs no intent key)")
    ap.add_argument("--pi05-weights", required=True, help="pi05_base_pytorch dir (model.safetensors)")
    ap.add_argument("--dataset-glob",
                    default="/home/ldahiya/LIBERO/libero/datasets/libero_goal/*_demo.hdf5")
    ap.add_argument("--out", required=True, help="cache dir (per-demo .npy of fp16 grids)")
    ap.add_argument("--batch", type=int, default=16, help="frames per VL forward")
    ap.add_argument("--limit-demos", type=int, default=0, help=">0 for a smoke test (per task)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import openpi.models.model as _model
    import openpi.training.config as _config
    import openpi.transforms as _transforms
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    import safetensors.torch

    pi05_config = _config.get_config(args.pi05_config)
    model = PI0Pytorch(pi05_config.model).to(args.device)
    # pi05_base has no LoRA keys; strict=False -> LoRA adapters init to B=0 -> VL == base VL.
    safetensors.torch.load_model(model, os.path.join(args.pi05_weights, "model.safetensors"), strict=False)
    model = model.float().eval()  # frozen; fp32 (pi05_base overflows bf16 on LIBERO)

    # Same transform chain as training (repack -> LiberoInputs -> Normalize -> tokenize/resize/pad).
    data_config = pi05_config.data.create(pi05_config.assets_dirs, pi05_config.model)
    tfm = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    action_horizon = int(pi05_config.model.action_horizon)

    def to_t(v):
        t = torch.as_tensor(v, device=args.device)
        return t.float() if t.is_floating_point() else t

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(args.dataset_glob))
    assert files, f"no hdf5 matched {args.dataset_glob}"
    print(f"[precompute] {len(files)} task files -> {args.out}", flush=True)

    for f in files:
        stem = os.path.basename(f).replace("_demo.hdf5", "").replace(".hdf5", "")
        prompt = stem.replace("_", " ")  # matches train_pi05_cotrain.py _task_langs
        with h5py.File(f, "r") as h:
            demos = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[-1]))
            if args.limit_demos:
                demos = demos[: args.limit_demos]
            for d in demos:
                out_path = os.path.join(args.out, f"{stem}__{d}.npy")
                if os.path.exists(out_path):
                    continue  # resumable
                agv = h[f"data/{d}/obs/agentview_rgb"][()]      # (T,H,W,3) uint8
                wr = h[f"data/{d}/obs/eye_in_hand_rgb"][()]     # (T,H,W,3) uint8
                T = agv.shape[0]
                grids = None
                for s in range(0, T, args.batch):
                    e = min(s + args.batch, T)
                    elems = [tfm({
                        "image": agv[t], "wrist_image": wr[t],
                        "state": np.zeros(8, np.float32),                  # VL prefix ignores state
                        "actions": np.zeros((action_horizon, 7), np.float32),
                        "prompt": prompt,
                    }) for t in range(s, e)]
                    coll = {k: (np.stack([el[k] for el in elems]) if not isinstance(elems[0][k], dict)
                                else {kk: np.stack([el[k][kk] for el in elems]) for kk in elems[0][k]})
                            for k in elems[0]}
                    obs = _model.Observation.from_dict({
                        k: (to_t(v) if not isinstance(v, dict) else {kk: to_t(vv) for kk, vv in v.items()})
                        for k, v in coll.items()})
                    with torch.no_grad():
                        grid = model.vl_image_features(obs, pool="none")   # (b, n_img, width)
                    n_per_img = grid.shape[1] // len(obs.images)           # agentview = first block
                    agv_grid = grid[:, :n_per_img].to(torch.float16).cpu().numpy()
                    if grids is None:
                        grids = np.empty((T, agv_grid.shape[1], agv_grid.shape[2]), dtype=np.float16)
                    grids[s:e] = agv_grid
                np.save(out_path, grids)
                print(f"  {stem}__{d}: {T} frames -> grid{grids.shape} {grids.nbytes/1e6:.0f}MB", flush=True)

    print("[precompute] done", flush=True)


if __name__ == "__main__":
    main()
