"""Reconcile OopsieVerse safe/damaging counts under each candidate safety definition.

Reads stored ``obs/health`` from the cached teleop HDF5s; no simulator. Definitions:

- spec_no_damage: no tracked link ever loses health (spec §7 "no damage event",
  operationalised as any health < 100 because the benchmark event threshold is not stored).
- paper_all_gt95: every object's final health > 95 (paper wording, "above 95").
- all_ge95: every object's final health >= 95 (the ">5 loss" column of the audit).
- code_mean_ge95: mean over objects of final object health >= 95, objects = link-name
  prefix before "@" with min over links (oopsieverse playback_robocasa.py:426-453,
  damagesim/utils/misc_utils.py:203-219).

Counts are of the health condition only. Success is stored for wipe_countertop; for
pick_egg it is not stored, and all 60 demos satisfy the height proxy.
"""

import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path.home() / (
    ".cache/huggingface/hub/datasets--ut-robin-lab--oopsieverse-demos/snapshots/"
    "c0ec29395fdcf898de2aed8dc0b3e786dc2a37d3/paper_experiments/teleop_data"
)
OUT = Path("outputs/oopsieverse_audit/safety_definition_reconciliation.json")
TASKS = ["pick_egg", "wipe_countertop"]
SPLITS = ["without_live_feedback", "with_live_feedback"]


def object_final_health(health: np.ndarray, link_names) -> dict[str, float]:
    final = {}
    for idx, name in enumerate(link_names):
        name = name.decode() if isinstance(name, bytes) else str(name)
        obj = name.split("@", 1)[0]
        final[obj] = min(final.get(obj, np.inf), float(health[-1, idx]))
    return final


def classify(demo) -> dict[str, bool]:
    health = np.asarray(demo["obs/health"])
    objs = object_final_health(health, demo.attrs["health_list_link_names"])
    vals = np.array(list(objs.values()))
    return {
        "spec_no_damage": bool((health >= 100.0).all()),
        "paper_all_gt95": bool((vals > 95.0).all()),
        "all_ge95": bool((vals >= 95.0).all()),
        "code_mean_ge95": bool(vals.mean() >= 95.0),
        "n_objects": len(vals),
    }


def main():
    result = {}
    for task in TASKS:
        result[task] = {}
        for split in SPLITS:
            with h5py.File(ROOT / task / f"{split}.hdf5", "r") as f:
                rows = [classify(f["data"][k]) for k in sorted(f["data"].keys())]
            counts = {"n": len(rows), "n_objects": sorted({r["n_objects"] for r in rows})}
            for d in ("spec_no_damage", "paper_all_gt95", "all_ge95", "code_mean_ge95"):
                safe = sum(r[d] for r in rows)
                counts[d] = {"no_damage_or_healthy": safe, "damaging": len(rows) - safe}
            result[task][split] = counts
    OUT.write_text(json.dumps(result, indent=2))
    for task, splits in result.items():
        for split, c in splits.items():
            summary = ", ".join(
                f"{d} {c[d]['no_damage_or_healthy']}/{c[d]['damaging']}"
                for d in ("spec_no_damage", "paper_all_gt95", "all_ge95", "code_mean_ge95")
            )
            print(f"{task:16s} {split:22s} n={c['n']} objects={c['n_objects']}: {summary}")


if __name__ == "__main__":
    main()
