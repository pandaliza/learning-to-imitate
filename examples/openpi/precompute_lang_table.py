"""Precompute the per-task cond_lang table for the causal workspace encoder.

For each LIBERO task, run ONE PaliGemma prefix forward and pool the LANGUAGE tokens (masked mean,
vl_image_features pool='lang') -> a [width] vector. Saves {stem: [width]} to an npz. This is the
global task-language AdaLN condition the causal encoder consumes (broadcast over T at stage-1). Cheap:
one forward per task (language tokens are prompt-only; the image just fills the required obs slot).

  python examples/openpi/precompute_lang_table.py \
      --pi05-weights .../pi05_base_pytorch --out .../lang_table_goal.npz
"""
import argparse
import glob
import os

import h5py
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi05-config", default="pi05_base_nointent")
    ap.add_argument("--pi05-weights", required=True)
    ap.add_argument("--dataset-glob", default="/home/ldahiya/LIBERO/libero/datasets/libero_goal/*_demo.hdf5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import openpi.models.model as _model
    import openpi.training.config as _config
    import openpi.transforms as _transforms
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    import safetensors.torch

    pi05_config = _config.get_config(args.pi05_config)
    model = PI0Pytorch(pi05_config.model).to(args.device)
    safetensors.torch.load_model(model, os.path.join(args.pi05_weights, "model.safetensors"), strict=False)
    model = model.float().eval()

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

    files = sorted(glob.glob(args.dataset_glob))
    assert files, f"no hdf5 matched {args.dataset_glob}"
    table = {}
    for f in files:
        stem = os.path.basename(f).replace("_demo.hdf5", "").replace(".hdf5", "")
        prompt = stem.replace("_", " ")            # matches train_pi05_cotrain.py _task_langs
        with h5py.File(f, "r") as h:
            d = sorted(h["data"].keys())[0]
            agv = h[f"data/{d}/obs/agentview_rgb"][0]      # one frame just fills the obs slot
            wr = h[f"data/{d}/obs/eye_in_hand_rgb"][0]
        el = tfm({"image": agv, "wrist_image": wr, "state": np.zeros(8, np.float32),
                  "actions": np.zeros((action_horizon, 7), np.float32), "prompt": prompt})
        coll = {k: (v[None] if not isinstance(v, dict) else {kk: vv[None] for kk, vv in v.items()})
                for k, v in el.items()}
        obs = _model.Observation.from_dict({
            k: (to_t(v) if not isinstance(v, dict) else {kk: to_t(vv) for kk, vv in v.items()})
            for k, v in coll.items()})
        with torch.no_grad():
            lang = model.vl_image_features(obs, pool="lang")[0].to(torch.float32).cpu().numpy()  # [width]
        table[stem] = lang.astype(np.float16)
        print(f"  {stem}: lang[{lang.shape[0]}] |mean|={np.abs(lang).mean():.3f}  prompt='{prompt}'", flush=True)

    np.savez(args.out, **table)
    print(f"[done] {len(table)} tasks -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
