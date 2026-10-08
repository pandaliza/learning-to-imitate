"""Path-only portability shim for OopsieVerse teleop HDF5s.

The stored ``model_file`` XML and ``ep_meta`` JSON carry absolute asset paths from the
collector's machine (``/Users/arnavbalaji/Desktop/Arnav_UT_CS/oopsieverse/externals/{robocasa,robosuite}/...``).
This rewrites exactly that prefix to the pinned local clones. Nothing else in the
string is touched (no physics, camera, texture or evaluation change). The function
asserts that no foreign absolute path survives.
"""

import os
import re

COLLECTOR_PREFIX = "/Users/arnavbalaji/Desktop/Arnav_UT_CS/oopsieverse/externals/"
LOCAL_EXTERNALS = os.environ.get("OOPSIEVERSE_EXTERNALS", "/edrive1/lizad/oopsieverse_full_study/src/")


def remap_asset_paths(s):
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    out = s.replace(COLLECTOR_PREFIX, LOCAL_EXTERNALS)
    leftover = re.findall(r"/Users/[^\"'\s<>]+", out)
    assert not leftover, f"unmapped collector paths remain: {leftover[:3]}"
    return out
