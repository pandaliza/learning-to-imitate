"""Shared paths and helpers for the OopsieVerse full study (Phase A).

Runs inside the isolated env /edrive1/lizad/oopsieverse_full_study/env/oopsie.
Read-only access to the cached HF teleop HDF5s.
"""

import os
import sys

STORAGE = "/edrive1/lizad/oopsieverse_full_study"
SRC = os.path.join(STORAGE, "src")
OOPSIE_ROOT = os.path.join(SRC, "oopsieverse")
REPO = "/scr/lizad/personal/learning-to-imitate"
OUT = os.path.join(REPO, "outputs/oopsieverse_full_study")

HF_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--ut-robin-lab--oopsieverse-demos/snapshots/"
    "c0ec29395fdcf898de2aed8dc0b3e786dc2a37d3/paper_experiments/teleop_data"
)

# HF folder name -> oopsiebench registry env name
TASKS = {"pick_egg": "pick_egg", "wipe_countertop": "wipe_counter"}
CONDITIONS = ["with", "without"]

ZERO_TOL = 1e-4  # health units; float32 health is exactly 100.0 when undamaged
MILD_MAX = 5.0   # pre-registered: 0 < worst-entity loss <= 5 is "mild"


def teleop_path(task, cond):
    return os.path.join(HF_SNAPSHOT, task, f"{cond}_live_feedback.hdf5")


def add_oopsie_paths():
    for p in (OOPSIE_ROOT, os.path.join(OOPSIE_ROOT, "oopsiebench")):
        if p not in sys.path:
            sys.path.insert(0, p)


def demo_sort_key(name):
    return int(name.split("_")[1])


def object_final_health(health_row, link_names):
    """Object health = min over its links (damagesim/core/damageable_mixin.py:129-134)."""
    objs = {}
    for v, n in zip(health_row, link_names):
        o = n.split("@", 1)[0]
        objs[o] = min(objs.get(o, 100.0), float(v))
    return objs


def stratum(worst_loss):
    if worst_loss <= ZERO_TOL:
        return "zero"
    if worst_loss <= MILD_MAX:
        return "mild"
    return "substantial"
