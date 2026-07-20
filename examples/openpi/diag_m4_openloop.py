"""Open-loop diagnostic: does M4 predict the DEMONSTRATION's ground-truth actions on real LIBERO
frames? Loads pi05_base_nointent + M4 ckpt exactly as the eval, builds the eval's 8-D state from a
demo hdf5 (ee_pos + ee_ori[axisangle] + gripper_states), runs policy.infer, compares the predicted
action chunk to the demo's GT actions. Low MSE => M4 learned (0 SR is a rollout/env issue); high
MSE => M4 genuinely didn't learn the actions. CPU to avoid the shared-debug-GPU OOM.
"""
import glob
import os

import h5py
import numpy as np
import torch

import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config
from openpi.training import checkpoints as _checkpoints

CK = os.environ.get(
    "DIAG_CK",
    "/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_m4_aux/14000")
cfg = _config.get_config("pi05_base_nointent")
dc = cfg.data.create(cfg.assets_dirs, cfg.model)
ns = _checkpoints.load_norm_stats(cfg.assets_dirs, dc.asset_id)
policy = _policy_config.create_trained_policy(cfg, CK, norm_stats=ns)
if isinstance(getattr(policy, "_model", None), torch.nn.Module):  # PyTorch ckpt (M4); JAX M0 skips
    policy._model = policy._model.float()
print(f"policy loaded from {CK}\n")

f = "/home/ldahiya/LIBERO/libero/datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5"
prompt = "open the middle drawer of the cabinet"
with h5py.File(f, "r") as h:
    o = h["data/demo_0/obs"]
    acts = h["data/demo_0/actions"][:]
    agv = o["agentview_rgb"][:]
    wr = o["eye_in_hand_rgb"][:]
    eep, eeo, grp = o["ee_pos"][:], o["ee_ori"][:], o["gripper_states"][:]
T = len(acts)
print(f"demo T={T}, prompt='{prompt}'\n")
for t in [0, T // 4, T // 2, 3 * T // 4]:
    state = np.concatenate([eep[t], eeo[t], grp[t]]).astype(np.float32)
    el = {"observation/image": agv[t], "observation/wrist_image": wr[t],
          "observation/state": state, "prompt": prompt}
    pred = np.asarray(policy.infer(el)["actions"])      # (H, 7)
    k = min(pred.shape[0], T - t)
    gt = acts[t:t + k]
    mse = float(((pred[:k] - gt) ** 2).mean())
    cos = float((pred[0] * acts[t]).sum() /
                (np.linalg.norm(pred[0]) * np.linalg.norm(acts[t]) + 1e-8))
    print(f"t={t:3d}  chunk_mse={mse:.4f}  step0_cos={cos:+.3f}")
    print(f"   pred[0]={np.round(pred[0],3)}")
    print(f"   gt  [0]={np.round(acts[t],3)}")
print("\nLOW mse + high cos => M4 learned the actions (0 SR is rollout/env, not the policy).")
print("HIGH mse / cos~0 => M4 genuinely did not learn.")
