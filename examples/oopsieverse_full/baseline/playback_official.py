"""Run the OFFICIAL RoboCasa playback (scripts/playback_robocasa.py, path-portable copy) with the
collection-time continuous gripper applied (metric contract: physics_and_controller.gripper_semantics).

Only three things differ from the released script:
  1. path-only remap of collector asset paths (../playback_portability.diff), and
  2. damagesim.robosuite.io_utils.patch_gripper_for_position_control() is called before env
     construction, exactly as teleop_robocasa.py:656-658 did at collection time (see
     playback_gripper_headless.diff in this directory).
  3. has_renderer=False (headless host; onscreen viewer only, see _force_headless).
All CLI arguments are forwarded unchanged to the official main(). Resolution defaults to the
official 256x256 (the learning dataset resizes seg to 128 with nearest-exact).

Usage: python playback_official.py --playback --input X.hdf5 --output Y.hdf5 --env pick_egg --demo_ids 3
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # examples/oopsieverse_full (path_remap, portable playback)
os.environ.setdefault("PYNPUT_BACKEND", "dummy")  # io_utils imports pynput; headless host

import playback_robocasa_portable as pb  # noqa: E402  (inserts oopsieverse paths)
from damagesim.robosuite.io_utils import patch_gripper_for_position_control  # noqa: E402



def _force_headless():
    """[HEADLESS] The released script builds the env with has_renderer=True (onscreen MuJoCo viewer,
    needs X11/GLFW). This host is headless, so we force has_renderer=False. Offscreen EGL rendering
    (has_offscreen_renderer=True, cameras, segmentation) is unchanged; the onscreen viewer is never
    used by playback_episode, so physics, obs and damage are unaffected."""
    import dataclasses
    from envs.registry import EnvironmentRegistry
    orig_get = EnvironmentRegistry.get.__func__

    def get(cls, name):
        cfg = orig_get(cls, name)
        base = cfg.damageable_class

        def make(*a, **kw):
            kw["has_renderer"] = False
            return base(*a, **kw)
        return dataclasses.replace(cfg, damageable_class=make)
    EnvironmentRegistry.get = classmethod(get)


if __name__ == "__main__":
    patch_gripper_for_position_control()  # [GRIPPER] collection-time semantics
    _force_headless()  # [HEADLESS]
    pb.main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
