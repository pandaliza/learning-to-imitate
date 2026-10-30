"""Shared helpers for the wipe_countertop diagnosis (read-only use of data and official code)."""
import json
import os
import sys
import zlib

import h5py
import numpy as np

FULL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FULL not in sys.path:
    sys.path.insert(0, FULL)
from path_remap import remap_asset_paths  # noqa: E402

STORAGE = "/edrive1/lizad/oopsieverse_full_study"
TELEOP = os.path.join(STORAGE, "data/oopsieverse-demos/paper_experiments/teleop_data/wipe_countertop/{cond}_live_feedback.hdf5")
VOCAB_RUN = os.path.join(STORAGE, "runs/matrix/amix_wipe_countertop_v1_s0")
OUT = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/wipe_diag"


def demo_reset(cond, demo):
    """Reset dict for baseline rollout_eval.apply_reset that restores a demo's initial state exactly as the
    official playback / study replay do: set_ep_meta(ep_meta) -> reset() -> reset_from_xml_string(model) ->
    sim.reset() -> set_state(states[0]). Seed only feeds env.rng (overridden by ep_meta/xml/state) and the
    per-episode torch noise generator."""
    with h5py.File(TELEOP.format(cond=cond), "r") as f:
        g = f["data"][demo]
        ep_meta = json.loads(remap_asset_paths(g.attrs["ep_meta"]))
        xml = remap_asset_paths(g.attrs["model_file"])
        s0 = g["states"][0].tolist()
        n = int(g["actions"].shape[0])
    seed = zlib.crc32(f"wipe_diag/{cond}/{demo}".encode()) & 0x7FFFFFFF
    return {"id": f"demo__{cond}__{demo}", "seed": seed, "family": f"demo/{cond}/{demo}", "ep_meta": ep_meta,
            "model_xml": xml, "state": s0, "demo_len": n}


def demo_actions(cond, demo):
    with h5py.File(TELEOP.format(cond=cond), "r") as f:
        return f["data"][demo]["actions"][:]


def split_demos():
    """(train, dev, test) lists of (cond, demo) from the frozen splits (cond in with/without)."""
    sp = json.load(open("/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/roster/splits_v1.json"))
    t = sp["tasks"]["wipe_countertop"] if "tasks" in sp else sp["wipe_countertop"]
    return t
