"""Build an intent sidecar for a DexMimicGen robomimic dataset (BC-RNN + intent runs).

The sidecar is a small HDF5 file whose demo groups link (h5py ExternalLink) to the original
dataset's obs / actions / action_dict / states, plus one new per-step dataset `intent`:
the poses of both end effectors (pos + quat, 14-D) at t+2, t+4, t+6, t+8, clipped to the
demo's last frame and concatenated (56-D, same target as MIP's hier_wp4 arm), min-max
normalized to [-1, 1] over all frames. Normalization stats for the intent and for the
predictor's low-dim inputs are stored as attrs on /data. The original file is not modified.

Usage: python make_intent_sidecar.py SRC.hdf5 DST.hdf5
"""

import sys

import h5py
import numpy as np

POSE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot1_eef_pos", "robot1_eef_quat"]
# predictor inputs: the official BC-RNN low-dim obs, in config order
LOWDIM_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
               "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"]
WAYPOINTS = [2, 4, 6, 8]


def wp_intent(poses):
    T = len(poses)
    idx = np.minimum(np.arange(T)[:, None] + np.array(WAYPOINTS)[None], T - 1)  # (T, W)
    return poses[idx].reshape(T, -1)


def main(src, dst):
    with h5py.File(src, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        intents, lowdims = {}, []
        for d in demos:
            obs = f["data"][d]["obs"]
            poses = np.concatenate([obs[k][()] for k in POSE_KEYS], -1).astype(np.float32)
            intents[d] = wp_intent(poses)
            lowdims.append(np.concatenate([obs[k][()] for k in LOWDIM_KEYS], -1).astype(np.float32))
        allint = np.concatenate(list(intents.values()))
        lo, hi = allint.min(0), allint.max(0)
        scale = np.maximum(hi - lo, 1e-6)
        alllow = np.concatenate(lowdims)

        with h5py.File(dst, "w") as g:
            gd = g.create_group("data")
            for k, v in f["data"].attrs.items():
                gd.attrs[k] = v
            gd.attrs["intent_lo"], gd.attrs["intent_hi"] = lo, hi
            gd.attrs["lowdim_lo"], gd.attrs["lowdim_hi"] = alllow.min(0), alllow.max(0)
            gd.attrs["lowdim_keys"] = LOWDIM_KEYS
            gd.attrs["intent_waypoints"] = WAYPOINTS
            gd.attrs["source"] = src
            if "mask" in f:
                f.copy("mask", g)
            for d in demos:
                sd, dd = f["data"][d], gd.create_group(d)
                for k, v in sd.attrs.items():
                    dd.attrs[k] = v
                for k in sd.keys():
                    dd[k] = h5py.ExternalLink(src, f"/data/{d}/{k}")
                dd.create_dataset("intent", data=(2 * (intents[d] - lo) / scale - 1).astype(np.float32))
    print(f"wrote {dst}: {len(demos)} demos, intent dim {allint.shape[1]}, {len(allint)} frames")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
