"""Phase A smoke test (spec section 4) for pick_egg and wipe_counter.

Checks: construction, reset, controller stepping, success detection, health updates,
offscreen rendering (one RGB + one segmentation frame). Writes
outputs/oopsieverse_full_study/env/smoke_test.json and PNG frames.
Run with MUJOCO_GL=egl.
"""

import json
import os
import time
import traceback

import numpy as np

from common import OUT, add_oopsie_paths

add_oopsie_paths()
from robosuite.controllers import load_composite_controller_config  # noqa: E402
from envs.registry import EnvironmentRegistry  # noqa: E402
from damagesim.utils.misc_utils import robocasa_check_success  # noqa: E402


def make(env_name, **over):
    cfg = EnvironmentRegistry.get(env_name)
    kw = dict(robots=cfg.robot, controller_configs=load_composite_controller_config(robot=cfg.robot),
              translucent_robot=False, has_renderer=False, has_offscreen_renderer=True,
              render_camera=cfg.camera_name, ignore_done=True, use_camera_obs=False,
              render_segmentation=False, control_freq=cfg.control_freq,
              # EGL on GPUs 0 and 2 of this host returns corrupt frames; 1 and 3-7 render correctly (see smoke_test.json)
              render_gpu_device_id=int(os.environ.get("OV_RENDER_GPU", "3")))
    kw.update(over)
    return cfg, cfg.damageable_class(**kw)


def run(env_name, outdir):
    r = {"env_name": env_name}
    t0 = time.time()
    cfg, env = make(env_name)
    r["construct_s"] = round(time.time() - t0, 2)
    r["robot"] = cfg.robot
    t0 = time.time()
    obs, info = env.reset()
    r["reset_s"] = round(time.time() - t0, 2)
    r["action_dim"] = int(env.action_dim)
    r["model_timestep"] = float(env.sim.model.opt.timestep)
    r["control_freq"] = env.control_freq
    r["substeps_per_control_step"] = int(round(env.control_timestep / env.model_timestep))
    r["state_dim"] = int(env.sim.get_state().flatten().shape[0])
    r["nq_nv"] = [int(env.sim.model.nq), int(env.sim.model.nv)]
    r["health_list_link_names"] = list(env.health_list_link_names)
    r["health_at_reset"] = obs["health"].tolist()
    r["obs_keys"] = sorted(obs.keys())
    r["success_at_reset"] = robocasa_check_success(env)

    # controller stepping: +x arm delta should move the EEF in the commanded (base-frame) direction
    eef0 = obs["robot0_base_to_eef_pos"].copy() if "robot0_base_to_eef_pos" in obs else obs["robot0_eef_pos"].copy()
    a = np.zeros(env.action_dim)
    a[0] = 0.5
    a[6] = -1.0  # gripper open
    t0 = time.time()
    for _ in range(10):
        obs, rew, done, info = env.step(a)
    r["step_ms"] = round((time.time() - t0) / 10 * 1000, 1)
    eef1 = obs["robot0_base_to_eef_pos"] if "robot0_base_to_eef_pos" in obs else obs["robot0_eef_pos"]
    r["eef_delta_after_10_plus_x_steps"] = (eef1 - eef0).tolist()
    r["controller_moves_eef_plus_x"] = bool((eef1 - eef0)[0] > 0.01)
    r["damage_info_keys"] = sorted(info.get("damage_info", {}).keys())

    # success detection
    if env_name == "pick_egg":
        bid = env.obj_body_id["egg"]
        jn = [j for j in env.objects["egg"].joints][0]
        qadr = env.sim.model.get_joint_qpos_addr(jn)
        st = env.sim.get_state()
        q = env.sim.data.qpos.copy()
        q[qadr[0] + 2] += 0.12
        env.sim.data.qpos[:] = q
        env.sim.forward()
        r["success_after_lifting_egg_12cm"] = robocasa_check_success(env)
        env.sim.set_state(st)
        env.sim.forward()
        r["success_after_restore"] = robocasa_check_success(env)
        # health update: drop the egg from 30 cm with zero robot action and check the mechanical evaluator fires
        h_before = float(env.objects["egg"].health) if hasattr(env.objects["egg"], "health") else None
        q = env.sim.data.qpos.copy()
        q[qadr[0] + 2] += 0.30
        env.sim.data.qpos[:] = q
        env.sim.forward()
        z = np.zeros(env.action_dim)
        hist = []
        for _ in range(20):
            obs, _, _, info = env.step(z)
            di = info["damage_info"].get("egg", {}).get("egg_main", {}).get("mechanical", {})
            hist.append([float(obs["health"][0]), float(di.get("impact_forces", 0.0)), float(di.get("damage", 0.0))])
        r["egg_drop_30cm_health_impact_damage_per_step"] = hist
        r["health_update_fires"] = bool(min(h[0] for h in hist) < 100.0)
        r["egg_health_before_drop"] = h_before
    else:
        r["n_markers"] = len(env.markers)
        r["wiped_after_reset"] = len(env.wiped_markers)
        # success detection: mark all markers wiped through the env's own bookkeeping list
        saved = list(env.wiped_markers)
        env.wiped_markers = list(env.markers)
        r["success_with_all_markers_wiped"] = robocasa_check_success(env)
        env.wiped_markers = saved
        r["success_after_restore"] = robocasa_check_success(env)
        # health update: press the gripper down into the counter for 40 steps
        a = np.zeros(env.action_dim)
        a[2] = -1.0
        a[6] = 1.0
        hist = []
        for _ in range(40):
            obs, _, _, info = env.step(a)
            hist.append([float(np.min(obs["health"])), int(info.get("total_wiped", -1))])
        r["press_down_min_health_and_wiped_per_step"] = hist
        r["health_update_fires"] = bool(min(h[0] for h in hist) < 100.0)

    # rendering: official camera-obs path (RGB + class segmentation), one reset, offscreen EGL
    try:
        import imageio
        _, env2 = make(env_name, render_segmentation=True, use_camera_obs=True, camera_names=[cfg.camera_name],
                       camera_widths=256, camera_heights=256)
        o2, i2 = env2.reset()
        rgb = o2[f"{cfg.camera_name}_image"]
        seg = o2[f"{cfg.camera_name}_segmentation_class"]
        p = os.path.join(outdir, f"smoke_{env_name}_{cfg.camera_name}.png")
        imageio.imwrite(p, rgb)
        imageio.imwrite(p.replace(".png", "_seg.png"), (seg.squeeze().astype(np.float32) * (255.0 / max(1, seg.max()))).astype(np.uint8))
        classes = i2["obs_info"]["robot0"][cfg.camera_name]["seg_instance"]
        r["render"] = {"ok": True, "render_gpu_device_id": env2.render_gpu_device_id, "rgb_shape": list(rgb.shape),
                       "rgb_mean": float(rgb.mean()), "n_unique_colors": int(len(np.unique(rgb.reshape(-1, 3), axis=0))),
                       "seg_classes_visible": sorted(str(classes.get(int(c), classes.get(str(int(c)), c))) for c in np.unique(seg)),
                       "png": os.path.relpath(p, OUT)}
        env2.close()
    except Exception as e:
        r["render"] = {"ok": False, "error": repr(e), "tb": traceback.format_exc()[-1500:]}
    env.close()
    return r


def main():
    outdir = os.path.join(OUT, "env")
    os.makedirs(outdir, exist_ok=True)
    res = {"MUJOCO_GL": os.environ.get("MUJOCO_GL")}
    for name in ["pick_egg", "wipe_counter"]:
        try:
            res[name] = run(name, outdir)
        except Exception as e:
            res[name] = {"error": repr(e), "tb": traceback.format_exc()}
        print(name, json.dumps({k: v for k, v in res[name].items() if "per_step" not in k and k != "obs_keys"}, default=str)[:3000])
    with open(os.path.join(outdir, "smoke_test.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
