"""Rollout videos for wipe_countertop (wipe v2). New script; baseline/rollout_eval.py and the evaluator are imported,
never edited.

  episode    one episode -> <out>.mp4 (20 fps) + <out>.json (record = rollout_eval.run_episode outcome fields + video info)
  composite  two episode mp4s side by side (left | right), synchronised by step; the shorter one's last frame is frozen

Episode = the frozen protocol (rollout_eval.run_episode: 1250-step timeout, 10-step post-success window, stochastic
Euler-16, x0 [, z] from the per-episode generator, full 8-step chunks), the frozen pools (wv2_common.pool_resets), the
counter-label alias LUT, and the policy built by eval_wipe_v2.load_policy (registry / matrix fallback) running
IN-PROCESS on CPU at batch 1 (MatrixPolicy.act -> decode). env.step is wrapped (as eval_wipe_v2 does) to grab frames
after every step; the reset frame is grabbed after apply_reset.
Video camera: robot0_frontview, which the policy never observes (its inputs are eye_in_hand / agentview_left /
agentview_right segmentation), re-aimed after the reset at the dirt: eye 0.45 m beyond the dirt centroid on the far
side from the robot base, 0.75 m above it, looking at the centroid shifted 0.1 m toward the robot. Only that camera's
model pose is changed (rendering only; no physics or policy-observation effect). Eye-in-hand RGB as an inset.
Overlay: step, markers wiped / total, min entity health, SUCCESS (predicate now) / DAMAGE (an entity <= 95, the
primary predicate) flags.
NOTE: batch-1 CPU inference differs from the batched-GPU evaluator by float round-off, so a rendered episode need
not reproduce the recorded outcome of the same (reset, seed); the sidecar json holds the rendered episode's outcome.

  CUDA_VISIBLE_DEVICES= MUJOCO_EGL_DEVICE_ID=<parity gpu> python render_rollout_video.py episode --run-dir RUN \
      --pool P-devlayout --reset wipe_v2__P-devlayout__g03__0002 --rollout-seed 0 --out VIDEOS/name
  python render_rollout_video.py composite LEFT.mp4 RIGHT.mp4 OUT.mp4 --labels "A-MIX s1" "I-PATH s0"
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wv2_common as C  # noqa: E402
import rollout_eval as RE  # noqa: E402

# imageio-ffmpeg's bundled ffmpeg has libx264; /scr/lizad/miniconda3/bin/ffmpeg has only openh264, used as a fallback
try:
    import imageio_ffmpeg as _iof
    _iof.get_ffmpeg_exe()
except Exception:
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "/scr/lizad/miniconda3/bin/ffmpeg")
FPS = 20
CAM = "robot0_frontview"


def aim_camera(env, obs):
    import mujoco
    m, d = env.sim.model, env.sim.data
    cid = m.camera_name2id(CAM)
    bid = m.cam_bodyid[cid]
    mk = np.array([d.body_xpos[m.body_name2id(x.root_body)] for x in env.markers])
    c = mk.mean(0)
    base = np.asarray(obs["robot0_base_pos"], np.float64)
    h = c[:2] - base[:2]
    h /= np.linalg.norm(h)
    eye = c + np.r_[h * 0.45, 0.75]
    tgt = c - np.r_[h * 0.10, 0.0]
    zc = eye - tgt
    zc /= np.linalg.norm(zc)
    xc = np.cross([0.0, 0.0, 1.0], zc)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    Rb = d.body_xmat[bid].reshape(3, 3)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (Rb.T @ np.stack([xc, yc, zc], 1)).flatten())
    m.cam_pos[cid] = Rb.T @ (eye - d.body_xpos[bid])
    m.cam_quat[cid] = q
    return {"camera": CAM, "eye": eye.tolist(), "target": tgt.tolist()}


def overlay(img, lines, color_flags):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    y = 6
    for txt in lines:
        dr.rectangle([4, y - 2, 4 + 9.5 * len(txt), y + 20], fill=(0, 0, 0))
        dr.text((8, y), txt, fill=(255, 255, 255), font=font)
        y += 24
    x = 8
    for txt, col in color_flags:
        w = 11 * len(txt) + 10
        dr.rectangle([x, y, x + w, y + 24], fill=col)
        dr.text((x + 5, y + 2), txt, fill=(255, 255, 255), font=font)
        x += w + 6
    return np.asarray(im)


def make_policy(run_dir, ckpt):
    import types
    import eval_wipe_v2 as E
    os.chdir(RE.OOPSIE)
    pol = E.load_policy(run_dir, "cpu", ckpt)
    pol.seg_lut = types.MethodType(lambda self, m: C.alias_seg_lut(self.class_to_id, m), pol)
    return pol


def cmd_episode(a):
    import imageio
    import torch
    torch.set_num_threads(a.threads)
    pol = make_policy(a.run_dir, a.ckpt)
    from ctxfix import install_ctx_free
    env = RE.make_env(C.TASK, use_camera_obs=False)
    install_ctx_free(env)
    reset = [r for r in C.pool_resets(a.pool) if r["id"] == a.reset]
    assert len(reset) == 1, a.reset
    reset = reset[0]
    meta = pol.arm_meta
    label = a.label or f"{meta['arm']} s{meta.get('seed')}"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    writer = imageio.get_writer(a.out + ".mp4.tmp.mp4", fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    st = {"t": 0, "wiped": 0, "nm": 50, "cam": None}
    S = a.size

    def grab(obs):
        main = env.sim.render(width=S, height=S, camera_name=CAM)[::-1].copy()
        ins = env.sim.render(width=S // 3, height=S // 3, camera_name="robot0_eye_in_hand")[::-1]
        main[S - S // 3 - 6:S - 6, S - S // 3 - 6:S - 6] = ins
        hmin = float(np.min(obs["health"]))
        ent = {}
        for v, n in zip(np.asarray(obs["health"], np.float64), env.health_list_link_names):
            o = n.split("@", 1)[0]
            ent[o] = min(ent.get(o, 100.0), float(v))
        succ = RE.success_now(env)
        dmg = min(ent.values()) <= 95.0
        flags = []
        if succ:
            flags.append(("SUCCESS", (20, 150, 40)))
        if dmg:
            flags.append(("DAMAGE", (190, 30, 30)))
        lines = [f"{label} | {reset['id'].split('__', 1)[-1]} r{a.rollout_seed}",
                 f"step {st['t']:4d}   wiped {st['wiped']:2d}/{st['nm']}   min health {hmin:5.1f}"]
        writer.append_data(overlay(main, lines, flags))

    orig_step, orig_apply = env.step, RE.apply_reset

    def step(act):
        obs, r, dn, info = orig_step(act)
        st["t"] += 1
        st["wiped"] = int(info.get("total_wiped", st["wiped"]))
        st["nm"] = int(info.get("total_markers", st["nm"]))
        grab(obs)
        return obs, r, dn, info

    def apply_reset(e, r):
        out = orig_apply(e, r)
        st["cam"] = aim_camera(e, out[0])
        grab(out[0])
        return out
    env.step, RE.apply_reset = step, apply_reset
    t0 = time.time()
    rec = RE.run_episode(env, pol, C.TASK, reset, a.rollout_seed, exec_horizon=8, record_chunks=True)
    writer.close()
    os.replace(a.out + ".mp4.tmp.mp4", a.out + ".mp4")
    rec.update({"video": a.out + ".mp4", "fps": FPS, "frames": st["t"] + 1, "camera": st["cam"], "label": label,
                "run_dir": a.run_dir, "arm": meta["arm"], "training_seed": meta.get("seed"), "run_id": meta.get("run_id"),
                "pool": a.pool, "episode_id": reset["id"], "final_markers_wiped": st["wiped"], "total_markers": st["nm"],
                "final_proportion_wiped": st["wiped"] / st["nm"], "damaging": not rec["primary_safe"],
                "inference": "in-process CPU, batch 1 (eval_wipe_v2.load_policy); counter-label alias",
                "render_wall_s": time.time() - t0, "egl_device": os.environ.get("MUJOCO_EGL_DEVICE_ID")})
    json.dump(rec, open(a.out + ".json", "w"), indent=1, default=float)
    print("EPISODE_DONE", a.out, "success", rec["success"], "wiped", st["wiped"], "steps", rec["steps"],
          "primary_safe", rec["primary_safe"], flush=True)
    os._exit(0)


def cmd_composite(a):
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    L = imageio.get_reader(a.left)
    R = imageio.get_reader(a.right)
    w = imageio.get_writer(a.out + ".tmp.mp4", fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    li, ri = iter(L), iter(R)
    lf = rf = None
    n = 0
    while True:
        nl, nr = next(li, None), next(ri, None)
        if nl is None and nr is None:
            break
        lf = nl if nl is not None else lf
        rf = nr if nr is not None else rf
        if lf is None:
            lf = np.zeros_like(rf)
        if rf is None:
            rf = np.zeros_like(lf)
        im = Image.new("RGB", (lf.shape[1] + rf.shape[1] + 16, max(lf.shape[0], rf.shape[0]) + 32), (255, 255, 255))
        im.paste(Image.fromarray(lf), (0, 32))
        im.paste(Image.fromarray(rf), (lf.shape[1] + 16, 32))
        dr = ImageDraw.Draw(im)
        dr.text((8, 5), a.labels[0], fill=(0, 0, 0), font=font)
        dr.text((lf.shape[1] + 24, 5), a.labels[1], fill=(0, 0, 0), font=font)
        w.append_data(np.asarray(im))
        n += 1
    w.close()
    os.replace(a.out + ".tmp.mp4", a.out)
    print("COMPOSITE_DONE", a.out, "frames", n)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("episode")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--ckpt", default="step_30000.pth")
    e.add_argument("--pool", required=True)
    e.add_argument("--reset", required=True)
    e.add_argument("--rollout-seed", type=int, default=0)
    e.add_argument("--out", required=True, help="path without extension")
    e.add_argument("--label", default=None)
    e.add_argument("--size", type=int, default=480)
    e.add_argument("--threads", type=int, default=2)
    c = sp.add_parser("composite")
    c.add_argument("left")
    c.add_argument("right")
    c.add_argument("out")
    c.add_argument("--labels", nargs=2, default=["baseline", "intent"])
    a = ap.parse_args()
    cmd_episode(a) if a.cmd == "episode" else cmd_composite(a)


if __name__ == "__main__":
    main()
