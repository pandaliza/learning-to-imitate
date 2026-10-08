"""Checkpoint inventory + historical-result audit for the spatial-robustness study (spec sections 3, 5, 7).

Run ON THE CLUSTER from the repo checkout that produced the historical logs (it needs the checkpoint
filesystem and, for training-config extraction, the openpi submodule):

  python examples/openpi/oops/audit_existing_runs.py --out-dir docs/oops_workshop/audit \
      --logs-dir /home/ldahiya/max_vla/much-ado-about-noising/logs

Writes:
  checkpoint_inventory.csv   one row per candidate checkpoint (identity, hash, step, training config facts)
  historical_results.csv     every historical perturbed/SR JSON, with harness era + verification status
  audit_report.md            what is usable and why

Harness-era rule (verified on the libero_goal MJCF, see docs/oops_workshop/evidence_report.md): before commit
5f06c5d (2026-08-03 18:59 -0400) eval_libero_perturbed.py used qpos ADDRESSES as FLAT-state indices. On every
eligible task that added dx to a gripper-finger joint or a neighbouring object's quaternion component, applied
dy to the target's world-x, and read [junk, x, y] as the object's xyz for validity/reach. Results from that era
are labelled INVALID_PERTURBATION and are never paper evidence. Post-fix results are aggregates only (no
per-episode records, no state hashes) and are labelled UNVERIFIED_AGGREGATE.
"""

import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import oops_common as oc  # noqa: E402
from eval_spatial_shifts import B, PRESETS  # noqa: E402

FIX_TIME = dt.datetime(2026, 8, 3, 18, 59, 5, tzinfo=dt.timezone(dt.timedelta(hours=-4)))

CANDIDATES = [
    # (label, config_name, checkpoint_dir, sidecar, family, note)
    ("M0_control", "pi05_base_nointent", PRESETS["M0"]["checkpoint_dir"], None, "pi05_base", "primary baseline"),
    ("M1_frozen_resnet_intent", "pi05_base_intent", PRESETS["M1"]["checkpoint_dir"], PRESETS["M1"]["intent_ckpt"],
     "pi05_base", "primary intent arm"),
    ("M3_frozen_vl_intent", "pi05_base_intent_m3", PRESETS["M3"]["checkpoint_dir"], PRESETS["M3"]["intent_stack"],
     "pi05_base", "secondary intent arm (+ frozen pi05_base VL tap)"),
    ("libero_family_control_35k", "pi05_libero_nointent",
     "/data/user_data/ldahiya/openpi-checkpoints/pi05_libero_nointent/nointent_goal_run1/35000", None,
     "pi05_libero", "different base: do not mix"),
    ("libero_family_intent_35k", "pi05_libero_intent",
     "/data/user_data/ldahiya/openpi-checkpoints/pi05_libero_intent/slot_intent_goal_run1/35000",
     PRESETS["M1"]["intent_ckpt"], "pi05_libero", "different base: do not mix"),
]


def config_facts(name):
    """Pull training facts from the openpi TrainConfig (needs external/openpi checked out)."""
    try:
        sys.path.insert(0, str(REPO / "external/openpi/src"))
        import openpi.training.config as _config
        cfg = _config.get_config(name)
    except Exception as e:
        return {"config_error": repr(e)[:200]}
    m = cfg.model
    keep = {}
    for f in ("action_dim", "action_horizon", "max_token_len", "paligemma_variant", "action_expert_variant",
              "pi05", "intent_dim", "dtype", "discrete_state_input"):
        if hasattr(m, f):
            keep[f"model.{f}"] = getattr(m, f)
    for f in ("batch_size", "num_train_steps", "seed", "exp_name", "pytorch_weight_path", "ema_decay"):
        if hasattr(cfg, f):
            keep[f] = getattr(cfg, f)
    for f in ("weight_loader", "lr_schedule", "optimizer", "freeze_filter", "data"):
        if hasattr(cfg, f):
            keep[f] = repr(getattr(cfg, f))[:400]
    keep["base_checkpoint_inferred"] = ("pi05_libero" if "pi05_libero" in keep.get("weight_loader", "") else
                                        "pi05_base" if "pi05_base" in keep.get("weight_loader", "") else "UNRESOLVED")
    return keep


def inventory(out_dir, extra, full_hash):
    rows = []
    for label, cfg_name, ck, side, family, note in CANDIDATES + extra:
        r = {"label": label, "config_name": cfg_name, "checkpoint_dir": ck, "family_from_name": family, "note": note,
             "exists": os.path.isdir(ck), "step": pathlib.Path(ck).name, "run_id": pathlib.Path(ck).parent.name}
        if r["exists"]:
            fp, parts = oc.dir_fingerprint(ck, full=full_hash)
            r["checkpoint_hash"] = fp
            r["files"] = ";".join(p.split("=")[0] for p in parts)[:600]
            r["has_norm_stats"] = any("norm_stats" in p for p in parts)
            r["format"] = "pytorch" if any(p.startswith("model.safetensors") for p in parts) else "jax/orbax"
            r["mtime"] = dt.datetime.fromtimestamp(os.path.getmtime(ck)).isoformat()
            sib = sorted(p.name for p in pathlib.Path(ck).parent.iterdir() if p.name.isdigit())
            r["sibling_steps"] = ",".join(sib)
        if side:
            r["intent_sidecar"] = side
            r["intent_sidecar_exists"] = os.path.exists(side)
            if os.path.exists(side):
                r["intent_sidecar_hash"] = oc.file_fingerprint(side, full=full_hash)
        r.update({f"cfg.{k}": v for k, v in config_facts(cfg_name).items()})
        rows.append(r)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (not k[0].islower() or k.startswith("cfg."), k))
    with open(out_dir / "checkpoint_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return rows


def classify_json(p):
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    mtime = dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc)
    base = {"file": str(p), "mtime_utc": mtime.isoformat(), "config": d.get("config"), "checkpoint": d.get("checkpoint"),
            "intent": d.get("intent")}
    rows = []
    if "results" in d and "deltas" in d:  # eval_libero_perturbed schema (aggregate only)
        era = "pre_fix_hardcoded_index" if mtime < FIX_TIME else "post_fix_runtime_index"
        status = "INVALID_PERTURBATION" if era.startswith("pre") else "UNVERIFIED_AGGREGATE"
        for dk, per in d["results"].items():
            for tk, v in per.items():
                if tk.startswith("_") or not isinstance(v, dict):
                    continue
                rows.append({**base, "kind": "perturbed", "harness_era": era, "status": status, "delta": dk,
                             "task_id": tk, "task": v.get("task"), "succ": v.get("succ"), "ep": v.get("ep"),
                             "sr": v.get("sr"), "n_valid": v.get("n_valid"), "sr_valid": v.get("sr_valid"),
                             "per_episode_records": False})
    elif "mean_sr" in d and "per_task" in d:  # eval_libero_intent schema
        rows.append({**base, "kind": "clean_sr", "harness_era": "native", "status": "AGGREGATE_ONLY",
                     "succ": str(d.get("total", "")).split("/")[0], "ep": str(d.get("total", "")).split("/")[-1],
                     "sr": d.get("mean_sr"), "per_episode_records": False,
                     "num_trials_per_task": d.get("num_trials_per_task")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--logs-dir", action="append", default=[], help="dirs with historical *.json / *.out (repeatable)")
    ap.add_argument("--extra-checkpoint", action="append", default=[],
                    help="label=config_name=checkpoint_dir (repeatable)")
    ap.add_argument("--full-hash", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    extra = [tuple(e.split("=", 2)) + (None, "extra", "") for e in args.extra_checkpoint]
    inv = inventory(out, extra, args.full_hash)

    hist, outs = [], []
    for ld in args.logs_dir:
        for p in sorted(pathlib.Path(ld).rglob("*.json")):
            hist.extend(classify_json(p) or [])
        for p in sorted(pathlib.Path(ld).rglob("*.out")):
            try:
                txt = p.read_text(errors="ignore")
            except Exception:
                continue
            if "eval_libero_perturbed" in txt or "SR|valid" in txt:
                outs.append({"file": str(p), "has_runtime_flat_idx": "flat_idx=" in txt,
                             "n_cell_lines": len(re.findall(r"\[task \d+\] δ=", txt))})
    cols = ["file", "kind", "harness_era", "status", "config", "checkpoint", "intent", "delta", "task_id", "task",
            "succ", "ep", "sr", "n_valid", "sr_valid", "per_episode_records", "num_trials_per_task", "mtime_utc"]
    with open(out / "historical_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(hist)
    (out / "historical_slurm_outs.json").write_text(json.dumps(outs, indent=2))

    n_pre = len({h["file"] for h in hist if h.get("harness_era") == "pre_fix_hardcoded_index"})
    n_post = len({h["file"] for h in hist if h.get("harness_era") == "post_fix_runtime_index"})
    lines = ["# Historical-run audit (generated)", "",
             f"Generated {dt.datetime.now().isoformat()} by examples/openpi/oops/audit_existing_runs.py", "",
             "## Checkpoints", "", "| label | exists | step | format | hash | base (from config) |", "|---|---|---|---|---|---|"]
    for r in inv:
        lines.append(f"| {r['label']} | {r['exists']} | {r['step']} | {r.get('format', '')} | "
                     f"`{str(r.get('checkpoint_hash', ''))[:30]}` | {r.get('cfg.base_checkpoint_inferred', 'UNRESOLVED')} |")
    lines += ["", "## Historical perturbation results", "",
              f"- {n_pre} perturbed JSON files predate the index fix -> **INVALID_PERTURBATION** (not evidence).",
              f"- {n_post} perturbed JSON files postdate the fix -> UNVERIFIED_AGGREGATE (no per-episode records, "
              "no state hashes, validity from the old post-hoc rule).",
              f"- {sum(o['has_runtime_flat_idx'] for o in outs)}/{len(outs)} perturbed slurm logs print a runtime "
              "`flat_idx=` line (post-fix harness).", "",
              "Historical clean-SR JSONs are aggregate-only; they may be quoted as historical aggregates with their "
              "file path, never used for per-episode or paired analysis."]
    (out / "audit_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
