"""OopsieVerse full study, phase B: task catalog + demonstration data catalog.

Read-only. No simulator is imported; damage params are parsed from the pinned
sources with ``ast`` and demos are read with h5py.

Usage:
    .venv/bin/python -I examples/oopsieverse_full/catalog/inventory_tasks_and_data.py \
        --src <pinned oopsieverse checkout> \
        --data /edrive1/lizad/oopsieverse_full_study/data/oopsieverse-demos \
        --out outputs/oopsieverse_full_study/catalog

Outputs (in --out):
    task_catalog.json / .csv        every task exposed by the pinned benchmark
    data_catalog.json               every file in the HF dataset revision
    episode_inventory.csv           one row per episode (RoboCasa: health classes + metadata)
    episode_inventory_b1k.csv       one row per BEHAVIOR-1K episode (counts/fields only)
    duplicates.json                 cross-file duplicate trajectories
"""

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

REPO_ID = "ut-robin-lab/oopsieverse-demos"
REVISION = "c0ec29395fdcf898de2aed8dc0b3e786dc2a37d3"
TOL = 1e-6  # health-loss numerical tolerance for "strict zero loss"
MILD = 5.0  # mild: 0 < max loss <= 5 ; substantial: > 5

# Released paper-experiment task folder -> registry task name / backend
PAPER_DIRS = {
    "pick_egg": ("robocasa", "pick_egg"),
    "wipe_countertop": ("robocasa", "wipe_counter"),
    "add_firewood": ("behavior1k", "add_firewood"),
    "shelve_item": ("behavior1k", "shelve_item"),
}
FEEDBACK = {"without_live_feedback": "nofb", "with_live_feedback": "fb", "all_data": "all"}

# Paper usage (from arXiv 2606.31993v1 via WebFetch summary, plus docs/learning.md).
# IL study (Table C.2): shelve_item 90, pick_egg 90, add_firewood 60, pour_water 60, wipe_counter 60.
# RL: shelve cereal box (DSRL), move cup of water (finetune), place plate (from scratch).
PAPER_IL = {"shelve_item", "pick_egg", "add_firewood", "pour_water", "wipe_counter"}
PAPER_RL = {"shelve_item", "place_plate", "pour_water"}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def arr_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def literal_assign(path, name):
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def mech_kind(m):
    """Classify a mechanical param block into impact / quasistatic modes."""
    out = []
    if not m:
        return out
    if m.get("impact_damage_sensitivity", 0) > 0:
        out.append("impact")
    if m.get("qs_damage_sensitivity", 0) > 0:
        out.append("quasistatic")
    return out


# ───────────────────────────── task catalog ─────────────────────────────

def robocasa_tasks(src):
    reg = literal_assign(src / "oopsiebench/envs/registry.py", "_ROBOCASA_ENVS")
    robot_over = literal_assign(src / "oopsiebench/envs/registry.py", "_ROBOT_OVERRIDES")
    dmg_obj = literal_assign(src / "damagesim/robosuite/params/damage_params.py", "DAMAGEABLE_OBJECTS")
    params = literal_assign(src / "damagesim/robosuite/params/damage_params.py", "OBJECT_PARAMS")
    rows = []
    for env_name, module, cls, dcls in reg:
        robot = robot_over.get(env_name, "PandaOmron")
        cfg = dmg_obj.get(env_name, dmg_obj["default"])
        ents = []
        for cat in cfg.get("categories", []):
            if cat == "agent":
                p = params[robot]["mechanical"]
                modes = set()
                for ov in p.get("link_config_overrides", {}).values():
                    modes.update(mech_kind(ov))
                modes.update(mech_kind(p))
                ents.append({"entity": f"robot:{robot}", "kind": "robot", "evaluators": params[robot]["damage_evaluators"],
                             "mechanical_modes": sorted(modes)})
            else:
                p = params.get(cat, params["fixture_default"])
                modes = set(mech_kind(p["mechanical"]))
                for ov in p["mechanical"].get("link_config_overrides", {}).values():
                    modes.update(mech_kind(ov))
                ents.append({"entity": f"fixture:{cat}", "kind": "fixture", "evaluators": p["damage_evaluators"],
                             "mechanical_modes": sorted(modes), "params": p["mechanical"]})
        for nm in cfg.get("names", []):
            base = nm.rsplit("_", 1)[0] if nm.rsplit("_", 1)[-1].isdigit() else nm
            key = nm if nm in params else (base if base in params else "default")
            p = params[key]
            ents.append({"entity": f"object:{nm}", "kind": "object", "param_key": key,
                         "evaluators": p["damage_evaluators"], "mechanical_modes": mech_kind(p["mechanical"]),
                         "params": p["mechanical"]})
        # success predicate source
        text = (src / f"oopsiebench/envs/robocasa/{module}.py").read_text()
        m = re.search(r"def _check_success\(self\):", text)
        line = text[: m.start()].count("\n") + 1 if m else None
        mechs = sorted({mm for e in ents for mm in e["mechanical_modes"]})
        rows.append(dict(
            task=env_name, backend="robocasa", module=f"oopsiebench/envs/robocasa/{module}.py",
            env_class=cls, robot=robot,
            damage_evaluators=sorted({ev for e in ents for ev in e["evaluators"]}),
            damage_mechanisms=mechs,
            entities=ents,
            entities_with_nonzero_sensitivity=[e["entity"] for e in ents if e["mechanical_modes"]],
            entities_with_zero_sensitivity=[e["entity"] for e in ents if not e["mechanical_modes"]],
            success_predicate_source=f"oopsiebench/envs/robocasa/{module}.py:{line} _check_success",
            in_paper_il=env_name in PAPER_IL, in_paper_rl=env_name in PAPER_RL,
        ))
    return rows


def b1k_tasks(src):
    dmg_obj = literal_assign(src / "damagesim/omnigibson/params/damage_params.py", "DAMAGEABLE_OBJECTS")
    params = literal_assign(src / "damagesim/omnigibson/params/damage_params.py", "PARAMS")
    rows = []
    for f in sorted((src / "oopsiebench/envs/behavior1k").glob("*.py")):
        if f.stem in ("__init__", "base", "default", "spatial_checks"):
            continue
        text = f.read_text()
        robot = re.search(r'ROBOT_TYPE = "(\w+)"', text).group(1)
        cfg = dmg_obj.get(f.stem)
        ents = []
        if cfg is None:
            ents.append({"entity": "default-mode (all non-skipped categories)", "kind": "default"})
        else:
            for cat in cfg.get("categories", []) + cfg.get("names", []):
                key = cat if cat in params else "default"
                p = params[key]
                if cat == "agent":
                    mech = p["mechanical"]
                    modes = set(mech_kind(mech))
                    for ov in mech.get("part_config_overrides", {}).values():
                        modes.update(mech_kind(ov))
                    ents.append({"entity": f"robot:{robot}", "kind": "robot", "evaluators": p["damage_evaluators"],
                                 "mechanical_modes": sorted(modes)})
                else:
                    ents.append({"entity": f"{'name' if cat in cfg.get('names', []) else 'category'}:{cat}",
                                 "kind": "object", "param_key": key, "evaluators": p["damage_evaluators"],
                                 "mechanical_modes": mech_kind(p.get("mechanical"))})
        evs = sorted({ev for e in ents for ev in e.get("evaluators", [])})
        hazards = []
        if "heatSource" in text or "heat_source" in text.lower() or "_burner_on" in text:
            hazards.append("heat_source_in_scene")
        # water: an actual particle system used by the task (not a mention in a comment)
        if (re.search(r'get_system\(\s*"water"|_water_system|ContainedParticles|TRANSITION_SYSTEMS = \("water"', text)
                or ("faucet" in f.stem and "enable_transition_rules=True" in text)):
            hazards.append("water_in_scene")
        if re.search(r"target_objects_temperature=\[\s*f?\"", text):
            hazards.append("temperature_tracked_in_viz")
        mechs = sorted({mm for e in ents for mm in e.get("mechanical_modes", [])})
        active = list(mechs)
        if "heat_source_in_scene" in hazards and "thermal" in evs:
            active.append("thermal")
        if "water_in_scene" in hazards and "electrical" in evs:
            active.append("fluid/electrical")
        m = re.search(r"def task_completion_check\(env\):", text)
        line = text[: m.start()].count("\n") + 1 if m else None
        hdr = re.search(r"Damage\s*:\s*(.+)", text[:600])
        rows.append(dict(
            task=f.stem, backend="behavior1k", module=f"oopsiebench/envs/behavior1k/{f.name}", robot=robot,
            use_gpu_dynamics="use_gpu_dynamics=True" in text,
            enable_transition_rules="enable_transition_rules=True" in text,
            damage_evaluators=evs, damage_mechanisms=active, scene_hazards=hazards,
            module_docstring_damage=hdr.group(1).strip() if hdr else None,
            entities=ents,
            success_predicate_source=(f"oopsiebench/envs/behavior1k/{f.name}:{line} task_completion_check"
                                      if line else None),
            init_state_pkl=bool(re.search(r'^INIT_STATE_PATH = "', text, re.M)),
            in_paper_il=f.stem in PAPER_IL, in_paper_rl=f.stem in PAPER_RL,
        ))
    return rows


# ───────────────────────────── data catalog ─────────────────────────────

def collector_proxy(ep_meta):
    """Home-directory user embedded in asset paths: a collector-machine proxy, not a person ID."""
    for oc in ep_meta.get("object_cfgs", []):
        p = oc.get("obj_groups") or oc.get("info", {}).get("mjcf_path") or ""
        m = re.match(r"/(?:Users|home)/([^/]+)/", str(p))
        if m:
            return m.group(1)
    return "unknown"


def object_final_health(health, names):
    out = {}
    for i, n in enumerate(names):
        obj = n.split("@")[0]
        out[obj] = min(out.get(obj, np.inf), float(health[-1, i]))
    return out


def robocasa_episode(path, rel, task_dir, fb, name, d):
    T = len(d["actions"])
    has_health = "obs/health" in d
    info_keys = sorted(d["info"].keys()) if "info" in d else []
    ep_meta = json.loads(d.attrs["ep_meta"]) if "ep_meta" in d.attrs else {}
    row = dict(
        file=rel, task_dir=task_dir, feedback=fb, demo=name, T=T,
        num_samples_attr=int(d.attrs.get("num_samples", -1)),
        action_dim=int(d["actions"].shape[1]), state_dim=int(d["states"].shape[1]),
        has_health=has_health, has_damage_info="damage_info" in info_keys,
        has_task_success_field=("task_success" in info_keys) or ("task_completion" in d),
        has_progress_field="proportion_wiped" in info_keys,
        has_ep_meta="ep_meta" in d.attrs, has_model_file="model_file" in d.attrs,
        info_keys="|".join(info_keys),
        rewards_nonzero=int((d["rewards"][:] != 0).sum()), dones_any=bool(d["dones"][:].any()),
        layout_id=ep_meta.get("layout_id"), style_id=ep_meta.get("style_id"),
        lang=ep_meta.get("lang"), collector_proxy=collector_proxy(ep_meta),
        init_base_pos=json.dumps([round(x, 6) for x in ep_meta.get("init_robot_base_pos", [])]),
        init_base_ori=json.dumps([round(x, 6) for x in ep_meta.get("init_robot_base_ori", [])]),
        dirt_hash=(hashlib.sha256(json.dumps(ep_meta["dirt_marker_positions"]).encode()).hexdigest()[:12]
                   if "dirt_marker_positions" in ep_meta else ""),
        object_names="|".join(oc.get("name", "") for oc in ep_meta.get("object_cfgs", [])),
        model_file_hash=hashlib.sha256(d.attrs["model_file"].encode()).hexdigest()[:16] if "model_file" in d.attrs else "",
        actions_hash=arr_hash(d["actions"][:]), state0_hash=arr_hash(d["states"][0]),
    )
    # initial object / eef configuration from stored obs
    obs = d["obs"]
    row["init_eef_pos"] = json.dumps(np.round(obs["robot0_eef_pos"][0].astype(float), 4).tolist())
    objpos = {}
    for k in obs.keys():
        if k.endswith("_pos") and not k.startswith("robot0") and "_to_" not in k and k != "eef_pos":
            objpos[k[:-4]] = np.round(obs[k][0].astype(float), 4).tolist()
    row["init_obj_pos"] = json.dumps(objpos)
    # stored success
    if "task_success" in info_keys:
        ts = [str(x) for x in d["info/task_success"][:]]
        row["stored_success_final"] = ts[-1] in ("True", "b'True'", "1", "true")
        row["stored_success_any"] = any(t in ("True", "b'True'", "1", "true") for t in ts)
        row["success_basis"] = "stored info/task_success"
    elif "proportion_wiped" in info_keys:
        pw = np.array([float(x.decode() if isinstance(x, bytes) else x) for x in d["info/proportion_wiped"][:]])
        row["stored_success_final"] = bool(pw[-1] >= 1.0)
        row["stored_success_any"] = bool((pw >= 1.0).any())
        row["final_proportion_wiped"] = float(pw[-1])
        row["success_basis"] = "stored info/proportion_wiped (== _check_success: all markers wiped)"
    else:
        row["stored_success_final"] = None
        row["stored_success_any"] = None
        row["success_basis"] = "not stored"
    # pick_egg diagnostic height proxy (NOT a success label)
    if "egg_pos" in obs:
        z = obs["egg_pos"][:, 2]
        row["proxy_lift_m"] = float(z[-1] - z[0])
    if has_health:
        h = d["obs/health"][:].astype(np.float64)
        names = [str(n) for n in d.attrs["health_list_link_names"]]
        loss = h[0][None, :] - h.min(0)[None, :]
        loss = loss[0]
        worst = float(loss.max())
        row.update(
            n_health_entities=len(names),
            initial_health_all_100=bool(np.allclose(h[0], 100.0)),
            max_health_loss=worst,
            worst_entity=names[int(loss.argmax())] if worst > TOL else "",
            damaged_entities="|".join(n for n, l in zip(names, loss) if l > TOL),
            health_class=("zero" if worst <= TOL else "mild" if worst <= MILD else "substantial"),
        )
        ofh = object_final_health(h, names)
        vals = np.array(list(ofh.values()))
        row.update(
            objects="|".join(ofh), object_final_health=json.dumps({k: round(v, 3) for k, v in ofh.items()}),
            paper_all_gt95=bool((vals > 95).all()), code_mean_ge95=bool(vals.mean() >= 95.0),
            strict_no_loss=bool(worst <= TOL),
        )
        below = (h < h[0][None, :] - TOL).any(1)
        row["first_damage_step"] = int(np.argmax(below)) if below.any() else ""
    else:
        row.update(health_class="not_stored")
    return row


def b1k_episode(rel, task_dir, fb, name, d, fattrs):
    T = len(d["action"])
    names = str(d.attrs.get("health_list_link_names", ""))
    cfg = json.loads(fattrs["config"]) if "config" in fattrs else {}
    sf = json.loads(fattrs["scene_file"]) if "scene_file" in fattrs else {}
    st = d["state"]
    return dict(
        file=rel, task_dir=task_dir, feedback=fb, demo=name, T=T,
        num_samples_attr=int(d.attrs.get("num_samples", -1)),
        action_dim=int(d["action"].shape[1]), state_dim=int(st.shape[1]),
        state_size_unique="|".join(str(x) for x in sorted(set(d["state_size"][:].tolist()))),
        datasets="|".join(sorted(k for k in d.keys())),
        has_health=("obs" in d and "health" in d["obs"]) or "health" in d,
        has_health_list_link_names="health_list_link_names" in d.attrs,
        n_health_link_names=len(re.findall(r"'([^']+)'", names)),
        reward_nonzero=int((d["reward"][:] != 0).sum()),
        terminated_any=bool(d["terminated"][:].any()), truncated_any=bool(d["truncated"][:].any()),
        init_metadata_members=len(d["init_metadata"]) if "init_metadata" in d else -1,
        transitions=str(d.attrs.get("transitions", "")),
        task_type=(cfg.get("task") or {}).get("type"),
        scene_model=(cfg.get("scene") or {}).get("scene_model"),
        robot_type=",".join(r.get("type", "") for r in cfg.get("robots", [])),
        og_version=(sf.get("versions") or {}).get("omnigibson", {}).get("version"),
        og_git_hash=(sf.get("versions") or {}).get("omnigibson", {}).get("git_hash"),
        assets_version=(sf.get("versions") or {}).get("behavior-1k-assets", {}).get("version"),
        physics_frequency=(cfg.get("env") or {}).get("physics_frequency"),
        action_frequency=(cfg.get("env") or {}).get("action_frequency"),
        actions_hash=arr_hash(d["action"][:]), state0_hash=arr_hash(st[0]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hf-meta", type=Path, help="optional JSON of HF file sizes/lfs sha256")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    src_commit = subprocess.run(["git", "-C", str(a.src), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()

    # ── tasks
    tasks = robocasa_tasks(a.src) + b1k_tasks(a.src)
    # ── data
    hf = json.loads(a.hf_meta.read_text()) if a.hf_meta else {}
    files, rc_rows, b1k_rows = [], [], []
    for p in sorted(a.data.rglob("*.hdf5")):
        rel = str(p.relative_to(a.data))
        if rel.startswith(".cache"):
            continue
        digest = sha256(p)
        parts = rel.split("/")
        if parts[0] == "paper_experiments":
            task_dir, fb = parts[2], FEEDBACK[p.stem]
            backend, task = PAPER_DIRS[task_dir]
            source = "paper_experiments"
        else:
            backend, source = parts[0], f"{parts[0]}/teleop"
            m = re.match(r"(.+)_(safe|unsafe)$", p.stem)
            task, fb = m.group(1), f"label:{m.group(2)}"
            task_dir = task
        f = h5py.File(p, "r")
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        rows = []
        for n in demos:
            d = f["data"][n]
            if backend == "robocasa":
                r = robocasa_episode(p, rel, task_dir, fb, n, d)
            else:
                r = b1k_episode(rel, task_dir, fb, n, d, f["data"].attrs)
            r.update(task=task, backend=backend, source=source, file_sha256=digest[:16])
            rows.append(r)
        (rc_rows if backend == "robocasa" else b1k_rows).extend(rows)
        Ts = [r["T"] for r in rows]
        frec = dict(
            path=rel, backend=backend, task=task, source=source, condition=fb, sha256=digest,
            hf_lfs_sha256_match=(hf.get(rel, {}).get("lfs_sha256") == digest) if hf else None,
            size_bytes=p.stat().st_size, n_episodes=len(rows),
            T_min=min(Ts), T_median=float(np.median(Ts)), T_max=max(Ts), T_total=int(sum(Ts)),
            action_dim=sorted({r["action_dim"] for r in rows}),
            file_attrs=sorted(f["data"].attrs.keys()),
            has_health=sorted({bool(r["has_health"]) for r in rows}),
        )
        if backend == "robocasa":
            frec.update(
                has_damage_info=sorted({r["has_damage_info"] for r in rows}),
                has_task_success_field=sorted({r["has_task_success_field"] for r in rows}),
                has_progress_field=sorted({r["has_progress_field"] for r in rows}),
                has_ep_meta=all(r["has_ep_meta"] for r in rows), has_model_file=all(r["has_model_file"] for r in rows),
                info_keys=sorted({r["info_keys"] for r in rows}),
                health_class_counts=dict(Counter(r["health_class"] for r in rows)),
                stored_success_final_counts=dict(Counter(str(r["stored_success_final"]) for r in rows)),
                paper_all_gt95_counts=dict(Counter(str(r.get("paper_all_gt95")) for r in rows)),
                code_mean_ge95_counts=dict(Counter(str(r.get("code_mean_ge95")) for r in rows)),
                collector_proxy=dict(Counter(r["collector_proxy"] for r in rows)),
                layout_style=dict(Counter(f"{r['layout_id']}/{r['style_id']}" for r in rows)),
                n_unique_base_pose=len({r["init_base_pos"] for r in rows}),
            )
        else:
            frec.update(
                reward_nonzero_episodes=sum(r["reward_nonzero"] > 0 for r in rows),
                terminated_any_episodes=sum(r["terminated_any"] for r in rows),
                task_type=sorted({str(r["task_type"]) for r in rows}),
                og_git_hash=sorted({str(r["og_git_hash"]) for r in rows}),
                scene_model=sorted({str(r["scene_model"]) for r in rows}),
                robot_type=sorted({str(r["robot_type"]) for r in rows}),
                n_unique_state0=len({r["state0_hash"] for r in rows}),
            )
        files.append(frec)
        f.close()

    # ── duplicates across files (identical action sequences)
    by_act = defaultdict(list)
    for r in rc_rows + b1k_rows:
        by_act[(r["backend"], r["actions_hash"])].append(f"{r['file']}#{r['demo']}")
    dup_groups = [v for v in by_act.values() if len(v) > 1]
    # classify: all_data <-> split file (expected) vs anything else
    unexpected = [g for g in dup_groups if not (len(g) == 2 and any("all_data" in x for x in g))]
    within_split = defaultdict(list)
    for r in rc_rows + b1k_rows:
        if r["source"] == "paper_experiments" and r["feedback"] != "all":
            within_split[(r["task_dir"], r["actions_hash"])].append(f"{r['feedback']}/{r['demo']}")
    within_dups = [v for v in within_split.values() if len(v) > 1]
    all_data_cov = {}
    for td in PAPER_DIRS:
        alls = {r["actions_hash"] for r in rc_rows + b1k_rows if r["task_dir"] == td and r["feedback"] == "all"}
        splits = {r["actions_hash"] for r in rc_rows + b1k_rows if r["task_dir"] == td and r["feedback"] in ("nofb", "fb")}
        all_data_cov[td] = dict(n_all=len(alls), n_split=len(splits), all_minus_split=len(alls - splits),
                                split_minus_all=len(splits - alls))
    dups = dict(identical_action_groups=len(dup_groups), unexpected_groups=unexpected,
                within_paper_split_duplicates=within_dups, all_data_vs_splits=all_data_cov)

    # ── write
    meta = dict(hf_repo=REPO_ID, hf_revision=REVISION, pinned_source_commit=src_commit,
                data_root=str(a.data), tolerance=TOL, mild_threshold=MILD)
    (a.out / "task_catalog.json").write_text(json.dumps(dict(meta=meta, tasks=tasks), indent=1, default=str))
    with open(a.out / "task_catalog.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "backend", "robot", "damage_mechanisms", "damage_evaluators", "entities",
                    "zero_sensitivity_entities", "success_predicate_source", "in_paper_il", "in_paper_rl"])
        for t in tasks:
            w.writerow([t["task"], t["backend"], t["robot"], "+".join(t["damage_mechanisms"]),
                        "+".join(t["damage_evaluators"]),
                        "|".join(f"{e['entity']}[{'/'.join(e.get('mechanical_modes', []))}]" for e in t["entities"]),
                        "|".join(t.get("entities_with_zero_sensitivity", [])),
                        t["success_predicate_source"], t["in_paper_il"], t["in_paper_rl"]])
    (a.out / "data_catalog.json").write_text(json.dumps(dict(meta=meta, files=files), indent=1, default=str))
    (a.out / "duplicates.json").write_text(json.dumps(dups, indent=1))
    for rows, name in ((rc_rows, "episode_inventory.csv"), (b1k_rows, "episode_inventory_b1k.csv")):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(a.out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"tasks={len(tasks)} files={len(files)} robocasa_eps={len(rc_rows)} b1k_eps={len(b1k_rows)}")
    print("dups:", json.dumps(dups)[:2000])


if __name__ == "__main__":
    main()
