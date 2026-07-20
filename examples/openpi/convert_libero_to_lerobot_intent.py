"""Convert local LIBERO HDF5 demos -> LeRobot dataset with a slot-intent column.

Reads the same HDF5 files a MIP `flow_intent` task config points at, computes a
per-frame intent vector with the trained MIP flow map (the deployable p(z|s)
generator), and writes a LeRobot dataset matching openpi's libero schema plus an
extra `intent` feature. The resulting dataset trains `pi05_libero_intent`.

Schema (mirrors external/openpi/examples/libero/convert_libero_data_to_lerobot.py,
plus `intent`):
  image       (128,128,3) uint8   <- agentview_rgb
  wrist_image (128,128,3) uint8   <- eye_in_hand_rgb
  state       (8,)        float32 <- ee_states(6) + gripper_states(2)
  actions     (7,)        float32
  intent      (D,)        float32 <- MIP flow map (D = 6 mean / 64 slot)

Usage (slot, once Stage-1 is trained):
  python examples/openpi/convert_libero_to_lerobot_intent.py \
    --task-config libero_goal_suite_image_slot_intent \
    --ckpt /data/.../libero-goal-suite-image/slot_intent/models/model_best.pt \
    --repo-id ldahiya/libero_goal_slot_intent

Dry run (existing 6D mean ckpt):
  python examples/openpi/convert_libero_to_lerobot_intent.py \
    --task-config libero_goal_suite_image_flow_intent \
    --ckpt /data/.../libero-goal-suite-image/flow_intent/models/model_best.pt \
    --repo-id ldahiya/libero_goal_mean_intent

Set HF_LEROBOT_HOME to control the output location.
"""

import argparse
import os
import shutil

import h5py
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

from mip.pi05_intent import IntentGenerator


def _prompt_from_path(path: str) -> str:
    name = os.path.basename(path)
    for suf in ("_demo.hdf5", ".hdf5"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return name.replace("_", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-config", required=True, help="MIP task config the ckpt was trained with")
    ap.add_argument("--ckpt", default=None, help="MIP flow_intent agent checkpoint (ResNet generator)")
    ap.add_argument("--repo-id", required=True, help="output LeRobot dataset repo id")
    ap.add_argument("--config-dir", default="examples/configs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-demos", type=int, default=-1, help="limit demos per file (debug); -1 = all")
    ap.add_argument("--intent-batch", type=int, default=256, help="frames per intent forward pass")
    # M3 (VL-grounded, decoupled): intent comes from the Stage-1 VL generator (vl_proj + flow map)
    # applied to the cached frozen-pi05_base VL agentview-grid mean -- NOT the ResNet IntentGenerator.
    ap.add_argument("--m3-vl-stack", default=None, help="M3: intent_stack_*.pt (vl_proj + intent_flow_map)")
    ap.add_argument("--vl-cache-dir", default=None, help="M3: dir of per-demo VL grid .npy (precompute_vl_grids)")
    args = ap.parse_args()

    img_key, wrist_key = "agentview_rgb", "eye_in_hand_rgb"
    m3 = bool(args.m3_vl_stack)
    if m3:  # M3: VL generator on cached frozen-pi05_base VL grids
        from mip.pi05_intent import CotrainIntentModule
        assert args.vl_cache_dir, "--m3-vl-stack requires --vl-cache-dir"
        cot = CotrainIntentModule(args.task_config, args.config_dir, device=args.device, vl_obs=True)
        cot.vl_proj(torch.zeros(1, int(cot.cfg.task.slot_vl_dim), device=args.device))  # materialize LazyLinear
        _st = torch.load(args.m3_vl_stack, map_location=args.device, weights_only=False)
        cot.vl_proj.load_state_dict(_st["vl_proj"])
        cot.agent.intent_flow_map.load_state_dict(_st["intent_flow_map"])
        cot.eval()
        intent_dim = cot.intent_dim
        dataset_paths = [os.path.expanduser(p) for p in cot.cfg.task.dataset_paths]
        print(f"[convert-M3] intent_dim={intent_dim}, VL stack step {_st.get('step')}, {len(dataset_paths)} files")
    else:
        assert args.ckpt, "ResNet mode requires --ckpt"
        gen = IntentGenerator(args.task_config, args.ckpt, args.config_dir, device=args.device)
        intent_dim = gen.intent_dim
        To = gen.obs_steps
        dataset_paths = [os.path.expanduser(p) for p in gen.cfg.task.dataset_paths]
        print(f"[convert] intent_dim={intent_dim}, obs_steps={To}, {len(dataset_paths)} files")

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=10,
        features={
            "image": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            "intent": {"dtype": "float32", "shape": (intent_dim,), "names": ["intent"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    n_episodes = 0
    for path in dataset_paths:
        prompt = _prompt_from_path(path)
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys())
            if args.max_demos > 0:
                demos = demos[: args.max_demos]
            for demo_key in demos:
                obs = f[f"data/{demo_key}/obs"]
                ee = obs["ee_states"][()].astype(np.float32)        # (T,6)
                grip = obs["gripper_states"][()].astype(np.float32)  # (T,2)
                joint = obs["joint_states"][()].astype(np.float32)   # (T,7)
                agv = obs[img_key][()]                               # (T,128,128,3) uint8
                wrist = obs[wrist_key][()]
                act = f[f"data/{demo_key}/actions"][()].astype(np.float32)[:, :7]  # (T,7)
                T = act.shape[0]

                state8 = np.concatenate([ee, grip], axis=-1)[:T]              # (T,8) -> Pi0.5

                if m3:
                    # M3: per-frame intent = VL generator on the cached frozen-pi05_base VL
                    # agentview-grid mean (the same signal eval taps live). One sample per frame.
                    stem = os.path.basename(path).replace("_demo.hdf5", "").replace(".hdf5", "")  # == precompute key
                    grid = np.load(os.path.join(args.vl_cache_dir, f"{stem}__{demo_key}.npy"))  # (T,256,2048) fp16
                    vlm = grid[:T].astype(np.float32).mean(axis=1)                               # (T,2048)
                    with torch.no_grad():
                        intents = np.concatenate(
                            [cot.vl_sample_intent(torch.from_numpy(vlm[i : i + args.intent_batch]).to(args.device))
                                .cpu().numpy()
                             for i in range(0, T, args.intent_batch)],
                            axis=0,
                        )  # (T, intent_dim)
                else:
                    # ResNet generator (M1): intent from state+image obs windows.
                    state15 = np.concatenate([ee, grip, joint], axis=-1)[:T]  # (T,15) -> flow map
                    sw = IntentGenerator.stack_windows(state15, To)           # (T,To,15)
                    agv_chw = np.moveaxis(agv[:T], -1, 1)                     # (T,3,128,128)
                    wrist_chw = np.moveaxis(wrist[:T], -1, 1)
                    iw = {
                        img_key: IntentGenerator.stack_windows(agv_chw, To),
                        wrist_key: IntentGenerator.stack_windows(wrist_chw, To),
                    }
                    intents = np.concatenate(
                        [
                            gen.intent_for_windows(
                                sw[i : i + args.intent_batch],
                                {k: v[i : i + args.intent_batch] for k, v in iw.items()},
                            )
                            for i in range(0, T, args.intent_batch)
                        ],
                        axis=0,
                    )  # (T, intent_dim)

                for t in range(T):
                    dataset.add_frame(
                        {
                            "image": agv[t],
                            "wrist_image": wrist[t],
                            "state": state8[t],
                            "actions": act[t],
                            "intent": intents[t],
                            "task": prompt,
                        }
                    )
                dataset.save_episode()
                n_episodes += 1
        print(f"[convert] {os.path.basename(path)}: done ({n_episodes} episodes total)")

    print(f"[convert] wrote {n_episodes} episodes to {output_path}")


if __name__ == "__main__":
    main()
