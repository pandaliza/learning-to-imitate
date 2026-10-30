"""H2 figures: what the wipe policy sees. Panels = RGB + policy-vocabulary segmentation (128x128, nearest-exact,
exactly as PlaybackDataset / the evaluator build it) for
  (a) training playback demos (vocabulary applied to the stored per-step obs_info names),
  (b) policy rollouts saved by diag_rollout.py (--frames), per condition.
Colours: unknown=black, DamageableCounter=grey, dirt=orange, robot=blue, sponge=green.
"""
import glob
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw

OUT = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/wipe_diag/png"
PB = "/edrive1/lizad/oopsieverse_full_study/data/playback/wipe_countertop__train__v1.hdf5"
RUNS = "/edrive1/lizad/oopsieverse_full_study/runs/wipe_diag"
CLASS_TO_ID = {"unknown": 0, "DamageableCounter": 1, "dirt": 2, "robot": 3, "sponge": 4}
PAL = np.array([[0, 0, 0], [150, 150, 150], [255, 140, 0], [40, 90, 230], [40, 200, 60]], np.uint8)
CAMS = ["robot0_eye_in_hand", "robot0_agentview_right", "robot0_agentview_left"]


def resize_nearest_exact(seg, size=128):
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(seg.astype(np.float32))[None, None]
    return F.interpolate(x, size=(size, size), mode="nearest-exact")[0, 0].numpy().astype(np.int64)


def panel(rgb, seg, label):
    a = Image.fromarray(rgb).resize((256, 256))
    b = Image.fromarray(PAL[np.clip(seg, 0, 4)]).resize((256, 256), Image.NEAREST)
    im = Image.new("RGB", (512, 276), (255, 255, 255))
    im.paste(a, (0, 20))
    im.paste(b, (256, 20))
    ImageDraw.Draw(im).text((4, 4), label, fill=(0, 0, 0))
    return im


def grid(panels):
    w, h = panels[0].size
    im = Image.new("RGB", (w * len(panels), h), (255, 255, 255))
    for i, p in enumerate(panels):
        im.paste(p, (i * w, 0))
    return im


def counts(seg):
    return {n: int((seg == i).sum()) for n, i in CLASS_TO_ID.items()}


def demo_pngs(stats):
    f = h5py.File(PB, "r")
    for k in ["demo_0", "demo_10"]:
        g = f["data"][k]
        T = g["action"].shape[0]
        for t in [0, T // 2]:
            info = json.loads(g["info/obs_info"][t].decode())
            ps = []
            for cam in CAMS:
                m = {int(i): n for i, n in info["robot0"][cam]["seg_instance"].items()}
                lut = np.zeros(max(m) + 1, np.int64)
                for i, n in m.items():
                    lut[i] = CLASS_TO_ID.get(n, 0)
                raw = g[f"obs/cam/{cam}/seg"][t].astype(np.int64)
                seg = resize_nearest_exact(lut[np.clip(raw, 0, len(lut) - 1)])
                stats[f"train_playback/{k}/t{t}/{cam}"] = counts(seg)
                ps.append(panel(g[f"obs/cam/{cam}/rgb"][t], seg, f"TRAIN playback {k} t={t} {cam}"))
            grid(ps).save(os.path.join(OUT, f"train_playback_{k}_t{t:04d}.png"))


def rollout_pngs(stats):
    for fn in sorted(glob.glob(os.path.join(RUNS, "amix_s*", "*", "*__frames.npz"))):
        z = np.load(fn)
        run, cond = fn.split("/")[-3], fn.split("/")[-2]
        ep = os.path.basename(fn).replace("__frames.npz", "")
        ts = sorted({x.split("__")[0] for x in z.files})
        for t in [ts[0], ts[min(2, len(ts) - 1)], ts[-1]]:
            ps = []
            for cam in CAMS:
                seg = resize_nearest_exact(z[f"{t}__seg__{cam}"])
                stats[f"rollout/{run}/{cond}/{ep}/{t}/{cam}"] = counts(seg)
                ps.append(panel(z[f"{t}__rgb__{cam}"], seg, f"{run} {cond} {ep[-28:]} {t} {cam}"))
            grid(ps).save(os.path.join(OUT, f"rollout_{run}_{cond}_{ep}_{t}.png"))


def main():
    os.makedirs(OUT, exist_ok=True)
    stats = {}
    demo_pngs(stats)
    if len(sys.argv) < 2 or sys.argv[1] != "--demo-only":
        rollout_pngs(stats)
    json.dump(stats, open(os.path.join(os.path.dirname(OUT), "h2_seg_class_pixel_counts_128.json"), "w"), indent=1)
    print(len(stats), "panels")


if __name__ == "__main__":
    main()
