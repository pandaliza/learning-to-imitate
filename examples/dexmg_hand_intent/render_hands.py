"""Hand-only renders of the Inspire dexterous hands (DexMimicGen Drawer Cleanup, rendered-hand intent, stage 2).

Each hand is rendered on its own from robosuite's gripper MJCF (the actual meshes and kinematics used in the
task), with the palm fixed at the model origin, a neutral grey material, no sites, a black background and one
fixed oblique camera per hand. Finger configurations come from the dataset's measured `robot{0,1}_gripper_qpos`
(12 hinge joints in the MJCF's joint order, i.e. the simulator state, not the 6 commanded values).

Camera framing is calibrated once per hand on the union bounding box of the visual geoms over a sample of
demo configurations, then kept fixed; nothing is cropped per frame.

Usage:
  # contact sheet of 20 representative configurations per hand (k-means over joint space)
  python render_hands.py --dataset D.hdf5 --out_dir OUT --sheet
  # full cache: OUT/hand_renders_64.hdf5 with right/left (N,64,64) uint8, qpos_right/left, demo_index, t
  python render_hands.py --dataset D.hdf5 --out_dir OUT
"""

import argparse
import hashlib
import json
import os
import time

import h5py
import mujoco
import numpy as np
from PIL import Image, ImageDraw

HANDS = {  # side -> (dataset robot prefix, robosuite gripper xml, camera azimuth)
    "right": ("robot0", "inspire_right_hand.xml", 45.0),
    "left": ("robot1", "inspire_left_hand.xml", -45.0),
}
ELEVATION = -30.0
GREY = 0.75
VISUAL_GROUP = 1  # robosuite puts visual meshes in group 1, collision meshes in group 0


def gripper_xml(side):
    import robosuite

    return os.path.join(os.path.dirname(robosuite.__file__), "models/assets/grippers", HANDS[side][1])


class HandRenderer:
    def __init__(self, side, size=64, margin=1.1):
        self.side, self.size = side, size
        self.xml = gripper_xml(side)
        self.model = mujoco.MjModel.from_xml_path(self.xml)
        self.model.geom_matid[:] = -1  # the hand's materials are black; use a neutral grey instead
        self.model.geom_rgba[:] = [GREY, GREY, GREY, 1.0]
        self.data = mujoco.MjData(self.model)
        self.vis = np.where(self.model.geom_group == VISUAL_GROUP)[0]
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth, self.cam.elevation = HANDS[side][2], ELEVATION
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[VISUAL_GROUP] = 1
        self.renderer = mujoco.Renderer(self.model, size, size)
        self.margin = margin
        self.calibrated = False

    def calibrate(self, qpos, n_render=1500, fill=0.85, hires=256):
        """Fix lookat/distance so the union silhouette of the hand over `qpos` (N,12) fills `fill` of the frame.

        Pass 1 frames generously from the geometric bounding box; two refinement passes render the sample
        poses at `hires`, take the union of their silhouettes, and re-center / re-scale the camera on it.
        """
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        rb = self.model.geom_rbound[self.vis][:, None]
        for q in qpos:
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            p = self.data.geom_xpos[self.vis]
            lo, hi = np.minimum(lo, (p - rb).min(0)), np.maximum(hi, (p + rb).max(0))
        fovy = np.deg2rad(self.model.vis.global_.fovy)
        self.cam.lookat[:] = (lo + hi) / 2
        self.cam.distance = float(1.5 * np.linalg.norm(hi - lo) / 2 / np.tan(fovy / 2))
        self.calibrated = True
        sel = qpos[np.linspace(0, len(qpos) - 1, min(n_render, len(qpos))).astype(int)]
        big = mujoco.Renderer(self.model, hires, hires)
        for _ in range(2):
            mask = np.zeros((hires, hires), bool)
            for q in sel:
                self.data.qpos[:] = q
                mujoco.mj_forward(self.model, self.data)
                big.update_scene(self.data, self.cam, self.opt)
                mask |= big.render().max(-1) > 8
            ys, xs = np.where(mask)
            cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
            extent = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
            c = big.scene.camera[0]
            fwd, up = np.array(c.forward), np.array(c.up)
            right = np.cross(fwd, up)
            per_px = 2 * self.cam.distance * np.tan(fovy / 2) / hires  # world units per pixel at the lookat plane
            self.cam.lookat[:] = self.cam.lookat + right * (cx - hires / 2) * per_px - up * (cy - hires / 2) * per_px
            self.cam.distance = float(self.cam.distance * extent / (fill * hires))
        big.close()
        return dict(lookat=self.cam.lookat.tolist(), distance=self.cam.distance, fill=fill, n_calib_poses=len(sel),
                    union_bbox_px=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], hires=hires)

    def set_camera(self, lookat, distance):
        self.cam.lookat[:] = lookat
        self.cam.distance = distance
        self.calibrated = True

    def render(self, q):
        assert self.calibrated
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.cam, self.opt)
        return self.renderer.render().mean(-1).astype(np.uint8)  # grayscale (size, size)

    def params(self):
        return dict(
            side=self.side, size=self.size, azimuth=self.cam.azimuth, elevation=self.cam.elevation,
            lookat=self.cam.lookat.tolist(), distance=float(self.cam.distance), grey=GREY, margin=self.margin,
            xml=self.xml, xml_md5=hashlib.md5(open(self.xml, "rb").read()).hexdigest(),
            joints=[self.model.joint(i).name for i in range(self.model.njnt)],
            joint_range=self.model.jnt_range.tolist(), mujoco=mujoco.__version__,
        )


def kmeans_representatives(x, k, seed=0, iters=50):
    """Indices of the samples nearest to k-means centroids of standardized x (N,D)."""
    rng = np.random.default_rng(seed)
    z = (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)
    c = z[rng.choice(len(z), k, replace=False)]
    for _ in range(iters):
        a = np.argmin(((z[:, None] - c[None]) ** 2).sum(-1), 1)
        c = np.stack([z[a == j].mean(0) if np.any(a == j) else c[j] for j in range(k)])
    return np.array([np.argmin(((z - cj) ** 2).sum(-1)) for cj in c])


def load_qpos(f, robot, demos, stride=1):
    return np.concatenate([f[f"data/{d}/obs/{robot}_gripper_qpos"][::stride] for d in demos]).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--sheet", action="store_true", help="only write contact sheets + checks")
    p.add_argument("--n_sheet", type=int, default=20)
    p.add_argument("--calib_stride", type=int, default=10, help="calibrate on every n-th demo, all frames")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    f = h5py.File(args.dataset, "r")
    demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
    lengths = np.array([f[f"data/{d}"].attrs["num_samples"] for d in demos])
    N = int(lengths.sum())
    meta = {}

    renderers = {}
    for side, (robot, _, _) in HANDS.items():
        r = HandRenderer(side, args.size)
        calib = r.calibrate(load_qpos(f, robot, demos[:: args.calib_stride]))
        renderers[side] = r
        meta[side] = dict(camera=r.params(), calibration=calib)
        print(f"[{side}] calibrated: lookat={np.round(calib['lookat'], 3).tolist()} dist={calib['distance']:.3f}")

    if args.sheet:
        checks = {}
        for side, (robot, _, _) in HANDS.items():
            r = renderers[side]
            q = load_qpos(f, robot, demos[::5], stride=3)
            idx = kmeans_representatives(q, args.n_sheet)
            idx = idx[np.argsort(q[idx].sum(1))]  # order by overall closure
            ims = np.stack([r.render(q[i]) for i in idx])
            # checks: articulation changes pixels; nothing touches the 1-px border
            diffs = np.abs(ims[1:].astype(int) - ims[:-1].astype(int)).mean((1, 2))
            border = np.concatenate([ims[:, 0], ims[:, -1], ims[:, :, 0], ims[:, :, -1]], 1).max()
            fg = (ims > 8).mean()
            checks[side] = dict(mean_abs_diff_between_consecutive=diffs.round(2).tolist(), max_border_pixel=int(border),
                                foreground_fraction=round(float(fg), 3), closure_sum=q[idx].sum(1).round(2).tolist())
            cols = 10
            up = 3
            rows = []
            for s in range(0, len(ims), cols):
                row = [np.kron(im, np.ones((up, up), np.uint8)) for im in ims[s:s + cols]]
                row += [np.zeros_like(row[0])] * (cols - len(row))
                rows.append(np.concatenate(row, 1))
            sheet = Image.fromarray(np.concatenate(rows, 0))
            d = ImageDraw.Draw(sheet)
            for j, i in enumerate(idx):
                d.text((j % cols * args.size * up + 2, j // cols * args.size * up + 2), f"{q[i].sum():.1f}", fill=255)
            path = os.path.join(args.out_dir, f"contact_sheet_{side}_{args.size}.png")
            sheet.save(path)
            print(f"[{side}] wrote {path}; border max={border} fg={fg:.3f} min consecutive diff={diffs.min():.2f}")
        json.dump(dict(meta=meta, checks=checks), open(os.path.join(args.out_dir, "contact_sheet_checks.json"), "w"), indent=1)
        return

    out = os.path.join(args.out_dir, f"hand_renders_{args.size}.hdf5")
    t0 = time.time()
    with h5py.File(out, "w") as g:
        g.attrs["source"] = os.path.abspath(args.dataset)
        g.attrs["render_meta"] = json.dumps(meta)
        g.create_dataset("demo_index", data=np.repeat(np.arange(len(demos)), lengths))
        g.create_dataset("t", data=np.concatenate([np.arange(n) for n in lengths]))
        g.create_dataset("demo_names", data=np.array(demos, dtype="S"))
        g.create_dataset("demo_lengths", data=lengths)
        for side, (robot, _, _) in HANDS.items():
            r = renderers[side]
            ims = g.create_dataset(side, (N, args.size, args.size), dtype=np.uint8, chunks=(256, args.size, args.size))
            qs = g.create_dataset(f"qpos_{side}", (N, 12), dtype=np.float32)
            k = 0
            for di, d in enumerate(demos):
                q = f[f"data/{d}/obs/{robot}_gripper_qpos"][()].astype(np.float32)
                ims[k:k + len(q)] = np.stack([r.render(qi) for qi in q])
                qs[k:k + len(q)] = q
                k += len(q)
                if di % 100 == 0:
                    print(f"[{side}] demo {di}/{len(demos)} frames {k}/{N} {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {out} ({N} frames x 2 hands, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
