"""Diagnostic: load M4 (pi05_base_nointent PyTorch ckpt) exactly as eval_libero_intent does,
check for missing/unexpected keys, and run ONE forward on a plausible LIBERO obs to see whether
the policy produces sane actions (vs garbage/zero/NaN) -- isolates eval-path bug from real perf.
"""
import numpy as np
import torch

import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config
from openpi.training import checkpoints as _checkpoints

CK = "/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_m4_aux/14000"

cfg = _config.get_config("pi05_base_nointent")
print("model intent_dim:", getattr(cfg.model, "intent_dim", "<none>"))
dc = cfg.data.create(cfg.assets_dirs, cfg.model)
ns = _checkpoints.load_norm_stats(cfg.assets_dirs, dc.asset_id)
print("norm stats keys:", list(ns.keys()) if ns else ns)

# --- replicate create_trained_policy's PyTorch load + report key mismatch directly ---
import safetensors.torch as _st
model = cfg.model.load_pytorch(cfg, CK + "/model.safetensors")
sd_ck = _st.load_file(CK + "/model.safetensors")
sd_model = model.state_dict()
missing = [k for k in sd_model if k not in sd_ck]
unexpected = [k for k in sd_ck if k not in sd_model]
print(f"\nKEY CHECK: model has {len(sd_model)} params, ckpt has {len(sd_ck)}")
print(f"  missing in ckpt (model needs, not loaded): {len(missing)}")
for k in missing[:12]:
    print("    -", k, tuple(sd_model[k].shape))
print(f"  unexpected in ckpt (not used by model): {len(unexpected)}")
for k in unexpected[:12]:
    print("    +", k)

# --- run one forward via the policy wrapper (full transform pipeline, fp32 like eval) ---
policy = _policy_config.create_trained_policy(cfg, CK, norm_stats=ns)
policy._model = policy._model.float()
el = {
    "observation/image": (np.random.rand(256, 256, 3) * 255).astype(np.uint8),
    "observation/wrist_image": (np.random.rand(256, 256, 3) * 255).astype(np.uint8),
    "observation/state": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.02, -0.02], dtype=np.float32),
    "prompt": "open the middle drawer of the cabinet",
}
out = policy.infer(el)
a = np.asarray(out["actions"])
print(f"\nACTION OUT: shape={a.shape} range=[{a.min():.3f},{a.max():.3f}] mean={a.mean():.3f} "
      f"std={a.std():.3f} nan={bool(np.isnan(a).any())}")
print("first action vec:", np.round(a[0], 3))
print("SANE if range ~[-1,1] non-degenerate; BAD if all~0 / huge / NaN")
