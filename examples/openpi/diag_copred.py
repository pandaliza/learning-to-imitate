"""M10 co-prediction sanity checks (docs/co_prediction.md section 5). Run on a GPU node.

  A. per-token adaRMS: RMSNorm(x, cond_2d) == RMSNorm(x, cond_2d expanded to (B,S,w)) exactly.
  B. flag-off parity: a copred model called WITHOUT intent targets must produce the same action
     velocities as the plain pi05_base_nointent model from the same weights (existing path untouched).
  C. plumbing + grad flow: copred forward returns sane shapes; action loss alone puts gradient into
     intent_in_proj (variant J: action tokens attend to intent tokens); intent loss alone puts
     gradient into intent_out_proj.
  D. (post-training gate, needs --checkpoint of a TRAINED run + real data) clean-intent gain:
     action loss with tau_I=0 + GT intent substantially below tau_I=1. At init this is ~0 by
     construction (intent_out_proj zero-init, fresh in_proj) -- do not gate on it before training.

  python examples/openpi/diag_copred.py --pi05-weights .../pi05_base_pytorch [--fp32]
"""
import argparse

import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

ap = argparse.ArgumentParser()
ap.add_argument("--pi05-weights", required=True, help="pi05_base PyTorch params dir (model.safetensors)")
ap.add_argument("--copred-config", default="pi05_base_copred")
ap.add_argument("--baseline-config", default="pi05_base_nointent")
ap.add_argument("--device", default="cuda")
ap.add_argument("--fp32", action="store_true")
args = ap.parse_args()
dev = args.device

import safetensors.torch as _st

cfg_c = _config.get_config(args.copred_config)
cfg_b = _config.get_config(args.baseline_config)
H = int(cfg_c.model.action_horizon)
h = int(cfg_c.model.copred_h)
d_i = int(cfg_c.model.copred_intent_dim)

model_c = PI0Pytorch(cfg_c.model).to(dev)
model_b = PI0Pytorch(cfg_b.model).to(dev)
sd = _st.load_file(args.pi05_weights + "/model.safetensors")
miss_c, _ = model_c.load_state_dict(sd, strict=False)
miss_b, _ = model_b.load_state_dict(sd, strict=False)
print(f"[load] copred missing={len(miss_c)} (expect only intent_in/out_proj): {sorted(miss_c)[:6]}")
print(f"[load] baseline missing={len(miss_b)}")
if args.fp32:
    model_c, model_b = model_c.float(), model_b.float()
model_c.eval(), model_b.eval()

# --- A. per-token adaRMS equivalence on a real expert layer ---
ln = model_c.paligemma_with_expert.gemma_expert.model.layers[0].input_layernorm
w = ln.dense.in_features
x = torch.randn(2, h + H, ln.dense.out_features // 3, device=dev, dtype=next(ln.parameters()).dtype)
cond = torch.randn(2, w, device=dev, dtype=x.dtype)
y2d, g2d = ln(x, cond)
y3d, g3d = ln(x, cond[:, None, :].expand(-1, h + H, -1))
ok_a = torch.equal(y2d, y3d) and torch.equal(g2d.expand_as(g3d), g3d)  # 2-D gate is (B,1,d), 3-D is (B,S,d)
print(f"[A] adaRMS 2D==3D-expanded: {'PASS' if ok_a else 'FAIL'} "
      f"(dy={(y2d - y3d).abs().max().item():.2e})")

# --- shared synthetic observation through the real transform pipeline ---
def build_obs(cfg):
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    tfm = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    rng = np.random.RandomState(0)
    el = tfm({
        "image": (rng.rand(256, 256, 3) * 255).astype(np.uint8),
        "wrist_image": (rng.rand(256, 256, 3) * 255).astype(np.uint8),
        "state": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.02, -0.02], dtype=np.float32),
        "actions": rng.randn(cfg.model.action_horizon, 7).astype(np.float32) * 0.1,
        "prompt": "open the middle drawer of the cabinet",
    })
    def _t(v):
        t = torch.as_tensor(np.asarray(v)[None], device=dev)
        return t.float() if t.is_floating_point() else t
    obs = _model.Observation.from_dict({
        k: (_t(v) if not isinstance(v, dict) else {kk: _t(vv) for kk, vv in v.items()})
        for k, v in el.items()})
    return obs, _t(el["actions"])

obs, actions = build_obs(cfg_b)
noise = torch.randn_like(actions)
time = torch.full((1,), 0.5, device=dev)

# --- B. flag-off parity ---
with torch.no_grad():
    la = model_c(obs, actions, noise=noise, time=time)   # copred model, NO intent -> baseline path
    lb = model_b(obs, actions, noise=noise, time=time)
d = (la - lb).abs().max().item()
print(f"[B] flag-off parity copred-vs-baseline: {'PASS' if d == 0.0 else f'FAIL (max|d|={d:.2e})'}")

# --- C. copred plumbing + grad flow ---
itgt = torch.randn(1, h, d_i, device=dev) * 0.1
tau_i = torch.full((1,), 0.7, device=dev)
model_c.train()
al, il = model_c(obs, actions, noise=noise, time=time, intent_targets=itgt, intent_time=tau_i)
print(f"[C] shapes action_loss={tuple(al.shape)} intent_loss={tuple(il.shape)} "
      f"finite={bool(torch.isfinite(al).all() and torch.isfinite(il).all())}")
model_c.zero_grad(set_to_none=True)
al.mean().backward(retain_graph=True)
g_in = model_c.intent_in_proj.weight.grad
print(f"[C] action-loss grad reaches intent_in_proj (attention coupling): "
      f"{'PASS' if g_in is not None and g_in.abs().max() > 0 else 'FAIL'}")
model_c.zero_grad(set_to_none=True)
il.mean().backward()
g_out = model_c.intent_out_proj.weight.grad
print(f"[C] intent-loss grad reaches intent_out_proj: "
      f"{'PASS' if g_out is not None and g_out.abs().max() > 0 else 'FAIL'}")

# --- D. clean-intent delta (gate only on a TRAINED checkpoint) ---
model_c.eval()
with torch.no_grad():
    l_clean, _ = model_c(obs, actions, noise=noise, time=time,
                         intent_targets=itgt, intent_time=torch.zeros(1, device=dev))
    l_noise, _ = model_c(obs, actions, noise=noise, time=time,
                         intent_targets=itgt, intent_time=torch.ones(1, device=dev))
print(f"[D] action loss tau_I=0 {l_clean.mean().item():.5f} vs tau_I=1 {l_noise.mean().item():.5f} "
      f"(expect clean << noise AFTER training; ~equal at init)")

# --- inference schedules smoke test ---
with torch.no_grad():
    for s in ("s1", "s2", "s3"):
        a = model_c.sample_actions_copred(dev, obs, schedule=s)
        print(f"[sched {s}] actions shape={tuple(a.shape)} range=[{a.min():.3f},{a.max():.3f}] "
              f"nan={bool(torch.isnan(a).any())}")
