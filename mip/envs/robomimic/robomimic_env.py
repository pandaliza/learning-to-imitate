"""This file contains the functions to create the environment."""

import collections
import io
import os
import sys

import gymnasium as gym
import numpy as np
from loguru import logger

from mip.config import TaskConfig
from mip.env_utils import MultiStepWrapper, VideoRecorder, VideoRecordingWrapper

# DexMimicGen tasks (https://github.com/NVlabs/dexmimicgen); same robomimic-format datasets.
DEXMG_ENV_NAMES = [
    "threading", "three_piece_assembly", "box_cleanup", "drawer_cleanup",
    "lift_tray", "coffee", "pouring", "can_sort",
]
ROBOMIMIC_ENV_NAMES = ["can", "lift", "square", "tool_hang", "transport", *DEXMG_ENV_NAMES]


def match_dataset_scene(inner_env, dataset_path):
    """Rebuild a DexMimicGen env's scene so its physics match the scenes the dataset was recorded in.

    The installed robosuite/dexmimicgen build the scene differently from the versions that generated
    the DexMimicGen datasets: (1) robosuite's base.xml sets `inertiagrouprange="0 0"`, so visual
    geoms no longer add mass (dex-hand palm 0.10 vs 0.50 kg, mug 0.056 vs 0.279 kg), and (2) each
    dexterous hand is mounted on the wrist rotated by 120 deg. Demo replays from the stored
    model_file are unaffected; fresh env.reset() scenes (i.e. every eval episode) are. This copies
    the dataset's hand-mount orientations, drops the inertia-group override, and reloads the scene.
    `hard_reset` is turned off so later resets keep this scene and only re-sample object placements.
    """
    import re

    import h5py

    with h5py.File(dataset_path, "r") as f:
        demo_xml = f["data"][sorted(f["data"].keys())[0]].attrs["model_file"]
    xml = inner_env.model.get_xml()
    xml = re.sub(r'\s*inertiagrouprange="[^"]*"', "", xml, count=1)
    n_fixed = 0
    for name in sorted(set(re.findall(r'<body[^>]*name="(gripper\d+_[^"]*)"', xml))):
        m_demo = re.search(r'<body[^>]*name="%s"[^>]*quat="([^"]*)"' % re.escape(name), demo_xml)
        if m_demo is None:
            continue
        xml, n = re.subn(
            r'(<body[^>]*name="%s"[^>]*quat=")[^"]*(")' % re.escape(name),
            lambda m: m.group(1) + m_demo.group(1) + m.group(2),
            xml,
        )
        n_fixed += n
    inner_env.hard_reset = False
    inner_env.reset_from_xml_string(xml)
    logger.info(f"match_dataset_scene: reloaded scene ({n_fixed} gripper body orientations from dataset)")


def make_env(task_config: TaskConfig, idx, render=False, seed=None):
    if task_config.env_name in ROBOMIMIC_ENV_NAMES:
        return make_robomimic_env(task_config, idx, render, seed=seed)
    else:
        raise ValueError(f"Environment {task_config.env_name} not supported")


def make_vec_env(task_config: TaskConfig, seed=None):
    # Suppress output by redirecting stdout temporarily
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()  # Redirect stdout to a string buffer
    # Use SyncVectorEnv for image-based tasks (rendering contexts can't be pickled)
    # or when num_envs=1 or save_video=True
    if (
        task_config.num_envs == 1
        or task_config.save_video
        or task_config.obs_type == "image"
    ):
        vnc_env_class = gym.vector.SyncVectorEnv
    else:
        vnc_env_class = gym.vector.AsyncVectorEnv
    if task_config.env_name in ROBOMIMIC_ENV_NAMES:
        try:
            envs = vnc_env_class(
                [
                    make_robomimic_env(task_config, idx, task_config.save_video, seed=seed)
                    for idx in range(task_config.num_envs)
                ],
            )
        finally:
            sys.stdout = original_stdout  # Restore stdout
        return envs
    else:
        raise ValueError(f"Environment {task_config.env_name} not supported")


def make_robomimic_env(task_config: TaskConfig, idx, render=False, seed=None):
    from mip.envs.robomimic.robomimic_image_wrapper import (
        RobomimicImageWrapper,
    )
    from mip.envs.robomimic.robomimic_lowdim_wrapper import (
        RobomimicLowdimWrapper,
    )

    def thunk():
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.obs_utils as ObsUtils

        def create_robomimic_env(
            env_meta, obs_keys=None, shape_meta=None, enable_render=True
        ):
            if task_config.obs_type == "state":
                ObsUtils.initialize_obs_modality_mapping_from_dict(
                    {"low_dim": obs_keys}
                )
            else:  # image observation
                modality_mapping = collections.defaultdict(list)
                for key, attr in shape_meta["obs"].items():
                    modality_mapping[attr.get("type", "low_dim")].append(key)
                ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=False,
                # For state tasks, enable offscreen rendering only when requested.
                # For image tasks, this must remain enabled to produce camera obs.
                render_offscreen=enable_render,
                use_image_obs=enable_render
                if task_config.obs_type == "image"
                else False,
            )
            if is_dexmg:
                # robomimic's EnvRobosuite reads di["object-state"] unconditionally; DexMimicGen
                # envs do not emit it, so supply an empty one.
                inner = env.env
                orig_get_obs = inner._get_observations

                def _get_obs_with_object(*args, **kwargs):
                    di = orig_get_obs(*args, **kwargs)
                    di.setdefault("object-state", np.zeros(0))
                    return di

                inner._get_observations = _get_obs_with_object
                if getattr(task_config, "match_dataset_scene", False):
                    match_dataset_scene(inner, dataset_path)
            return env

        # Get dataset path (either from explicit path or HuggingFace download)
        if hasattr(task_config, "dataset_repo") and hasattr(
            task_config, "dataset_filename"
        ):
            from huggingface_hub import hf_hub_download

            dataset_path = hf_hub_download(
                repo_id=task_config.dataset_repo,
                filename=task_config.dataset_filename,
                repo_type="dataset",
            )
        elif hasattr(task_config, "dataset_path"):
            dataset_path = os.path.expanduser(task_config.dataset_path)
        else:
            raise ValueError(
                "Either dataset_repo/dataset_filename or dataset_path must be provided"
            )

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        is_dexmg = task_config.env_name in DEXMG_ENV_NAMES
        if is_dexmg:
            import dexmimicgen  # noqa: F401  registers the TwoArm* envs with robosuite

            # DexMimicGen envs do not accept env_lang, and their datasets carry no object obs.
            env_meta["env_kwargs"].pop("env_lang", None)
            env_meta["env_kwargs"]["use_object_obs"] = False
            # The dataset pins render_gpu_device_id=0, and EGL ignores CUDA_VISIBLE_DEVICES, so
            # every concurrent run would render on physical GPU 0. Honor MUJOCO_EGL_DEVICE_ID.
            if os.environ.get("MUJOCO_EGL_DEVICE_ID"):
                env_meta["env_kwargs"]["render_gpu_device_id"] = int(os.environ["MUJOCO_EGL_DEVICE_ID"])
        if task_config.obs_type == "image":
            # disable object state observation for image mode
            env_meta["env_kwargs"]["use_object_obs"] = False
        abs_action = task_config.abs_action
        if abs_action:
            env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
        if getattr(task_config, "abs_control", False):
            # robosuite 1.5 composite configs nest arm controllers under body_parts and select
            # absolute control with input_type; control_delta (set above) is a no-op there.
            for part in env_meta["env_kwargs"]["controller_configs"].get("body_parts", {}).values():
                if part.get("type", "").startswith("OSC"):
                    part["input_type"] = "absolute"

        if task_config.obs_type == "state":
            env = create_robomimic_env(
                env_meta=env_meta,
                obs_keys=task_config.obs_keys,
                enable_render=render,
            )
            env = RobomimicLowdimWrapper(
                env=env,
                obs_keys=task_config.obs_keys,
                init_state=None,
                render_hw=(256, 256),
                render_camera_name=getattr(task_config, "render_camera_name", "agentview"),
            )
        else:  # image observation
            env = create_robomimic_env(
                env_meta=env_meta, shape_meta=task_config.shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            env.env.hard_reset = False
            env = RobomimicImageWrapper(
                env=env,
                shape_meta=task_config.shape_meta,
                init_state=None,
                render_obs_key=task_config.render_obs_key,
            )

        video_recoder = VideoRecorder.create_h264(
            fps=10,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=22,
            thread_type="FRAME",
            thread_count=1,
        )
        file_path = None if not render else "results/video.mp4"
        env = VideoRecordingWrapper(
            env, video_recoder, file_path=file_path, steps_per_render=2
        )
        env = MultiStepWrapper(
            env,
            n_obs_steps=task_config.obs_steps,
            n_action_steps=task_config.act_steps,
            max_episode_steps=task_config.max_episode_steps,
        )
        if seed is not None:
            env.seed(seed + idx)
            logger.info(f"Env seed: {seed + idx}")
        return env

    return thunk
