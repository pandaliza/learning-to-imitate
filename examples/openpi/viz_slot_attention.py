"""Visualize the frozen VL slot encoder's per-slot spatial attention over LIBERO frames.

For each chosen frame: feed its cached PaliGemma token grid (256 = 16x16 patches) through
proj -> SlotAttention(return_attn) -> selector, then overlay each of the K slots' attention
(reshaped 16x16, upsampled) on the agentview image. The selector weight per slot is annotated;
the selected (argmax) slot — the one that becomes the intent — is boxed. Shows whether slots
bind to distinct objects and whether the selected slot tracks the manipulated object.
"""
import argparse
import glob
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mip.networks.slot_attention import SlotObjectEncoder

B = "/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi"
ap = argparse.ArgumentParser()
ap.add_argument("--task", default="open_the_middle_drawer_of_the_cabinet")
ap.add_argument("--demo", type=int, default=0)
ap.add_argument("--n-frames", type=int, default=4)
ap.add_argument("--num-slots", type=int, default=4)
ap.add_argument("--vl-dim", type=int, default=2048, help="slot-encoder input dim (PaliGemma 2048; DINOv2/DynaFLIP 768)")
ap.add_argument("--stack", default=f"{B}/ldahiya_checkpoints/slot_intent_vl/intent_stack_32000.pt")
ap.add_argument("--vl-cache", default=f"{B}/vl_cache_goal_agentview")
ap.add_argument("--hdf5-dir", default="/home/ldahiya/LIBERO/libero/datasets/libero_goal")
ap.add_argument("--out", default="figures/slot_attention.png")
args = ap.parse_args()
torch.manual_seed(0)  # slots are randomly initialized; fix for reproducible maps

# --- build + load the frozen slot encoder (soft selector; K from --num-slots) ---
se = SlotObjectEncoder(num_slots=args.num_slots, slot_dim=64, num_iters=3, obj_state_dim=6, slot_input_dim=64,
                       use_recon_decoder=True, use_soft_selector=True, vl_input=True, vl_dim=args.vl_dim)
se.load_state_dict(torch.load(args.stack, map_location="cpu", weights_only=False)["slot_encoder"], strict=False)
se.eval()

# --- data: cached VL grid (T,256,2048) + agentview frames (T,H,W,3) ---
grid = np.load(f"{args.vl_cache}/{args.task}__demo_{args.demo}.npy").astype(np.float32)  # (T,256,2048)
with h5py.File(f"{args.hdf5_dir}/{args.task}_demo.hdf5", "r") as h:
    imgs = h[f"data/demo_{args.demo}/obs/agentview_rgb"][:]  # (T,H,W,3) uint8
T = min(len(grid), len(imgs))
frames = np.linspace(0, T - 1, args.n_frames).astype(int)
K, P = args.num_slots, 16  # slots, 16x16 patch grid

fig, axes = plt.subplots(len(frames), K + 1, figsize=(3 * (K + 1), 3 * len(frames)))
for r, f in enumerate(frames):
    g = torch.from_numpy(grid[f])[None]                 # (1,256,2048)
    with torch.no_grad():
        feat = se.proj(g)                                # (1,256,64)
        slots, attn = se.slot_attention(feat, return_attn=True)  # slots (1,K,64), attn (1,K,256)
        w = se.selector(slots).softmax(dim=1)[0, :, 0]   # (K,) selector weights over slots
    attn = attn[0].reshape(K, P, P)                      # (K,16,16) per-slot spatial map
    attn = F.interpolate(attn[:, None], size=imgs.shape[1:3], mode="bilinear",
                         align_corners=False)[:, 0].numpy()
    img = np.flipud(imgs[f])                             # LIBERO agentview is stored upside-down
    sel = int(w.argmax())
    axes[r, 0].imshow(img); axes[r, 0].set_ylabel(f"frame {f}", fontsize=11)
    axes[r, 0].set_title("agentview" if r == 0 else ""); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    for k in range(K):
        a = axes[r, k + 1]
        a.imshow(img); a.imshow(np.flipud(attn[k]), cmap="jet", alpha=0.55)
        a.set_title(f"slot {k}  w={w[k]:.2f}" + ("  ★SELECTED" if k == sel else ""),
                    fontsize=10, color=("crimson" if k == sel else "black"))
        a.set_xticks([]); a.set_yticks([])
        if k == sel:
            for s in a.spines.values(): s.set_edgecolor("crimson"); s.set_linewidth(3)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.suptitle(f"VL slot-encoder attention — {args.task} (demo {args.demo})", fontsize=13)
plt.tight_layout()
plt.savefig(args.out, dpi=110)
print(f"saved {args.out}  (T={T}, frames={list(frames)})")
