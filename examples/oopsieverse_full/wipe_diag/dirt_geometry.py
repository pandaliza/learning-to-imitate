"""Dirt-layout geometry, demos vs eval_dev bank resets (CPU only: no renderer, no segmentation).
Bank resets use the canonical procedure (unset_ep_meta, env.rng = default_rng(seed), reset); the sponge position is
checked against the bank's obj_pos. Demo layouts come from the stored ep_meta. Output: marker xy (50x2), sponge xy,
base xy for each, plus summary stats of the dirt path relative to the sponge start."""
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.dirname(HERE)
for p in (os.path.join(FULL, "matrix"), os.path.join(FULL, "baseline"), FULL, HERE):
    sys.path.insert(0, p)
import rollout_eval as RE  # noqa: E402
import rollout_eval_matrix as REM  # noqa: E402
from wipe_common import TELEOP  # noqa: E402


def make_env_cpu():
    from damagesim.robosuite.io_utils import patch_gripper_for_position_control
    patch_gripper_for_position_control()
    from robosuite.controllers import load_composite_controller_config
    from envs.registry import EnvironmentRegistry
    cfg = EnvironmentRegistry.get("wipe_counter")
    return cfg.damageable_class(robots=cfg.robot, controller_configs=load_composite_controller_config(robot=cfg.robot),
                                translucent_robot=False, has_renderer=False, has_offscreen_renderer=False,
                                render_camera=cfg.camera_name, ignore_done=True, use_camera_obs=False,
                                render_segmentation=False, control_freq=cfg.control_freq)


def demo_geom():
    out = {}
    for cond in ("with", "without"):
        with h5py.File(TELEOP.format(cond=cond), "r") as f:
            for d in f["data"]:
                g = f["data"][d]
                em = json.loads(g.attrs["ep_meta"])
                eef0 = g["obs/robot0_eef_pos"][0].tolist() if "robot0_eef_pos" in g["obs"] else None
                out[f"{cond}/{d}"] = {"markers": em["dirt_marker_positions"], "base_xy": em["init_robot_base_pos"][:2],
                                      "eef0": eef0, "T": int(g["actions"].shape[0])}
    return out


def bank_geom(n):
    os.chdir(RE.OOPSIE)
    env = make_env_cpu()
    out = {}
    for r in REM.dev_resets("wipe_countertop")[:n]:
        env.unset_ep_meta()
        env.rng = np.random.default_rng(int(r["seed"]))
        env.reset()
        sp = env.sim.data.body_xpos[env.obj_body_id["sponge"]].tolist()
        arm = env.robots[0].arms[0]
        eef = env.sim.data.site_xpos[env.robots[0].eef_site_id[arm]].tolist()
        em = env.get_ep_meta()
        out[r["id"]] = {"markers": em["dirt_marker_positions"], "base_xy": em["init_robot_base_pos"][:2], "eef0": eef,
                        "sponge": sp, "sponge_vs_bank_maxabs": float(np.max(np.abs(np.asarray(sp) - np.asarray(r["obj_pos"]))))}
    return out


def main():
    res = {"demos": demo_geom(), "bank_eval_dev": bank_geom(int(sys.argv[2]))}
    json.dump(res, open(sys.argv[1], "w"))
    print("ok", len(res["demos"]), len(res["bank_eval_dev"]),
          "max sponge diff vs bank", max(v["sponge_vs_bank_maxabs"] for v in res["bank_eval_dev"].values()))


if __name__ == "__main__":
    main()
