"""Stage 4: sidecar with wp4 wrist intent + future rendered-hand latents + future finger joints (control).

Per-step datasets added next to the ExternalLinks of the original demo groups (same layout as
examples/dexmg_bcrnn/make_intent_sidecar.py, whose `intent` is reproduced bit-for-bit):
  intent        (T, 56)  wp4: both eef poses at t+2,4,6,8, min-max normalized to [-1, 1]   (unchanged)
  hand_intent   (T, 128) concat_k [E_L(I^L_{t+k}), E_R(I^R_{t+k})], k in WAYPOINTS, standardized
  joint_intent  (T, 96)  concat_k [q_L_{t+k}, q_R_{t+k}] (12 measured joints per hand), standardized
Future indices are clipped to the demo's last frame exactly as wp4 does; labels never cross episodes.
Standardization uses mean/std over all frames of the policy training set (= all demos; there is no
filter key), with std floored at `std_floor`; the stats, the AE checkpoint md5s and the render/AE
provenance are stored as attrs on /data.

Usage: python make_hand_intent_sidecar.py --src D.hdf5 --renders hand_renders_64.hdf5 --ae_dir OUT --out DST.hdf5
         [--check_against existing_wp4_sidecar.hdf5]
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../dexmg_bcrnn"))
from make_intent_sidecar import LOWDIM_KEYS, POSE_KEYS, WAYPOINTS, wp_intent  # noqa: E402

HAND_ORDER = ["left", "right"]  # per offset: [z_L, z_R] as in the spec


def future(x, T):
    idx = np.minimum(np.arange(T)[:, None] + np.array(WAYPOINTS)[None], T - 1)
    return x[idx].reshape(T, -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--renders", required=True)
    p.add_argument("--ae_dir", required=True, help="dir holding ae_left/ and ae_right/ from train_hand_ae.py")
    p.add_argument("--out", required=True)
    p.add_argument("--std_floor", type=float, default=1e-3)
    p.add_argument("--check_against", default=None, help="existing wp4 sidecar; assert identical `intent`")
    args = p.parse_args()

    lat, prov = {}, {}
    for side in HAND_ORDER:
        lat[side] = np.load(os.path.join(args.ae_dir, f"ae_{side}", "latents.npy"))
        prov[side] = json.load(open(os.path.join(args.ae_dir, f"ae_{side}", "latents_provenance.json")))
    with h5py.File(args.renders, "r") as r:
        lengths = r["demo_lengths"][()]
        demo_names = [s.decode() for s in r["demo_names"][()]]
        qpos = {side: r[f"qpos_{side}"][()] for side in HAND_ORDER}
        assert os.path.abspath(r.attrs["source"]) == os.path.abspath(args.src), "renders built from another dataset"
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    for side in HAND_ORDER:
        assert len(lat[side]) == offsets[-1], f"{side} latents ({len(lat[side])}) != frames ({offsets[-1]})"

    with h5py.File(args.src, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        assert demos == demo_names, "demo order differs between dataset and renders"
        wp, hand, joint, lowdims = {}, {}, {}, []
        for i, d in enumerate(demos):
            obs, T = f["data"][d]["obs"], int(lengths[i])
            assert obs[POSE_KEYS[0]].shape[0] == T
            sl = slice(offsets[i], offsets[i + 1])
            wp[d] = wp_intent(np.concatenate([obs[k][()] for k in POSE_KEYS], -1).astype(np.float32))
            z = np.concatenate([lat[s][sl] for s in HAND_ORDER], -1)  # (T, 32): [z_L, z_R]
            q = np.concatenate([qpos[s][sl] for s in HAND_ORDER], -1)  # (T, 24): [q_L, q_R]
            hand[d], joint[d] = future(z, T), future(q, T)
            lowdims.append(np.concatenate([obs[k][()] for k in LOWDIM_KEYS], -1).astype(np.float32))

        allwp = np.concatenate(list(wp.values()))
        lo, hi = allwp.min(0), allwp.max(0)
        scale = np.maximum(hi - lo, 1e-6)
        alllow = np.concatenate(lowdims)
        stats = {}
        for name, dct in [("hand_intent", hand), ("joint_intent", joint)]:
            a = np.concatenate(list(dct.values()))
            mu, sd = a.mean(0), a.std(0)
            n_low = int((sd < args.std_floor).sum())
            stats[name] = dict(mean=mu, std=np.maximum(sd, args.std_floor), dim=a.shape[1], n_low_var_dims=n_low,
                               raw_std_min=float(sd.min()), raw_std_max=float(sd.max()))
            print(f"{name}: dim {a.shape[1]}, std range [{sd.min():.4f}, {sd.max():.4f}], {n_low} dims under floor")

        check = h5py.File(args.check_against, "r") if args.check_against else None
        with h5py.File(args.out, "w") as g:
            gd = g.create_group("data")
            for k, v in f["data"].attrs.items():
                gd.attrs[k] = v
            gd.attrs["intent_lo"], gd.attrs["intent_hi"] = lo, hi
            gd.attrs["lowdim_lo"], gd.attrs["lowdim_hi"] = alllow.min(0), alllow.max(0)
            gd.attrs["lowdim_keys"] = LOWDIM_KEYS
            gd.attrs["intent_waypoints"] = WAYPOINTS
            gd.attrs["source"] = args.src
            for name, s in stats.items():
                gd.attrs[f"{name}_mean"], gd.attrs[f"{name}_std"] = s["mean"].astype(np.float32), s["std"].astype(np.float32)
            gd.attrs["hand_intent_layout"] = json.dumps(dict(waypoints=WAYPOINTS, hand_order=HAND_ORDER,
                                                              latent_per_hand=lat["left"].shape[1]))
            gd.attrs["joint_intent_layout"] = json.dumps(dict(waypoints=WAYPOINTS, hand_order=HAND_ORDER, joints_per_hand=12))
            gd.attrs["provenance"] = json.dumps(dict(renders=os.path.abspath(args.renders), ae=prov,
                                                     std_floor=args.std_floor, n_low_var_dims={k: v["n_low_var_dims"] for k, v in stats.items()}))
            if "mask" in f:
                f.copy("mask", g)
            for d in demos:
                sd_, dd = f["data"][d], gd.create_group(d)
                for k, v in sd_.attrs.items():
                    dd.attrs[k] = v
                for k in sd_.keys():
                    dd[k] = h5py.ExternalLink(args.src, f"/data/{d}/{k}")
                wpn = (2 * (wp[d] - lo) / scale - 1).astype(np.float32)
                if check is not None:
                    assert np.array_equal(check["data"][d]["intent"][()], wpn), f"wp4 intent mismatch in {d}"
                dd.create_dataset("intent", data=wpn)
                for name, dct in [("hand_intent", hand), ("joint_intent", joint)]:
                    s = stats[name]
                    dd.create_dataset(name, data=((dct[d] - s["mean"]) / s["std"]).astype(np.float32))
        if check is not None:
            check.close()
            print("wp4 `intent` identical to", args.check_against)
    print(f"wrote {args.out}: {len(demos)} demos, {offsets[-1]} frames; hand_intent {stats['hand_intent']['dim']}-D, "
          f"joint_intent {stats['joint_intent']['dim']}-D")


if __name__ == "__main__":
    main()
