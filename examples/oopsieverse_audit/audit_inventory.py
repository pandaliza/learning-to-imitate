"""Stage 0B inventory: hashes, schema, demo_inventory.csv, stored-field consistency checks, partitions.

Usage: .venv/bin/python examples/oopsieverse_audit/audit_inventory.py
"""

import csv
import json
from collections import Counter, defaultdict

import h5py
import numpy as np

from common import (
    B1K_TASKS,
    OUT,
    ROBOCASA_TASKS,
    SNAPSHOT,
    SPLIT_TAG,
    SPLITS,
    demo_names,
    fmt,
    h5path,
    load_task,
    sha256,
)

COLUMNS = [
    "task", "simulator_version", "dataset_hash", "source_split", "episode_id",
    "length", "task_success", "any_damage", "safe_success",
    "first_damage_step", "damaged_entities", "initial_health", "final_health",
    "observation_keys", "action_schema", "playback_version", "replay_verified",
    # extra, clearly-labelled diagnostic columns (not part of the spec schema)
    "outcome_class", "proxy_task_success", "proxy_basis", "max_health_loss",
    "final_proportion_wiped", "session_id", "group_id", "partition",
]

ROBOCASA_ACTION_SCHEMA = "12-d float64: [0:6] eef delta pose, [6] gripper (-1 open .. ~0 closed, analog), [7:12] base/torso (held 0/-1)"
B1K_ACTION_SCHEMA = "7-d float32: [0:6] IK eef delta pose, [6] gripper"


def schema_of(path):
    out = {"file_attrs": {}, "data_attrs": {}, "demo_datasets": {}, "demo_attrs": [], "n_demos": 0}
    with h5py.File(path, "r") as f:
        out["file_attrs"] = list(f.attrs.keys())
        out["data_attrs"] = list(f["data"].attrs.keys())
        names = demo_names(f)
        out["n_demos"] = len(names)
        out["demo_ids"] = names
        d = f["data"][names[0]]

        def vis(n, o):
            if isinstance(o, h5py.Dataset):
                out["demo_datasets"][n] = [list(o.shape[1:]), str(o.dtype)]
            else:
                out["demo_datasets"][n + "/"] = "group"

        d.visititems(vis)
        out["demo_attrs"] = list(d.attrs.keys())
        if "health_list_link_names" in d.attrs:
            v = d.attrs["health_list_link_names"]
            out["health_list_link_names"] = [str(x) for x in v] if not isinstance(v, str) else v
        if "config" in f["data"].attrs:
            cfg = json.loads(f["data"].attrs["config"])
            out["b1k_task_type"] = cfg.get("task", {}).get("type")
            out["b1k_damage_params"] = [
                {"name": o.get("name"), "damage_params": o.get("damage_params")}
                for o in cfg.get("objects", [])
                if o.get("damage_params")
            ]
        # B1K: check whether any stored field could carry success/health
        if "reward" in d:
            agg = Counter()
            for n in names:
                dd = f["data"][n]
                agg["reward_nonzero"] += int(np.any(dd["reward"][:] != 0))
                agg["terminated_any"] += int(np.any(dd["terminated"][:]))
                agg["truncated_any"] += int(np.any(dd["truncated"][:]))
                agg["transitions_nonempty"] += int(str(dd.attrs.get("transitions", "{}")) not in ("{}", ""))
                agg["init_metadata_nonempty"] += int(len(dd["init_metadata"].keys()) > 0)
                agg["steps"] += int(dd["action"].shape[0])
            out["b1k_field_scan"] = dict(agg)
    return out


def union_find(n):
    p = list(range(n))

    def find(i):
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    def union(i, j):
        p[find(i)] = find(j)

    return find, union


def build_groups(eps):
    """Reset-family/session/near-duplicate grouping (outcome-free)."""
    n = len(eps)
    find, union = union_find(n)
    notes = {}
    # 1. session: identical init_robot_base_pos within condition; if a condition has a single
    #    base pose for all episodes it carries no session info -> conservative pseudo-sessions of
    #    5 consecutive episode indices (the block size observed where sessions are detectable).
    for split in SPLITS:
        idx = [i for i, e in enumerate(eps) if e["split"] == split]
        keys = {i: tuple(np.round(eps[i]["base_xy"], 6)) for i in idx}
        uniq = sorted(set(keys.values()))
        if len(uniq) > 1:
            notes[split] = f"{len(uniq)} distinct init_robot_base_pos values -> used as session ids"
            for i in idx:
                eps[i]["session_id"] = f"{SPLIT_TAG[split]}-base{uniq.index(keys[i])}"
        else:
            notes[split] = "single init_robot_base_pos for whole condition -> pseudo-sessions of 5 consecutive demos"
            for i in idx:
                eps[i]["session_id"] = f"{SPLIT_TAG[split]}-blk{eps[i]['idx'] // 5}"
    by_sess = defaultdict(list)
    for i, e in enumerate(eps):
        by_sess[e["session_id"]].append(i)
    for members in by_sess.values():
        for j in members[1:]:
            union(members[0], j)
    # 1b. reset family (wipe_countertop): identical dirt-marker layout stored in ep_meta
    fam = defaultdict(list)
    for i, e in enumerate(eps):
        if "dirt" in e:
            fam[(e["split"], tuple(np.round(e["dirt"].ravel(), 6)))].append(i)
            e["reset_family"] = None
    for k, (key, members) in enumerate(sorted(fam.items(), key=lambda kv: kv[1][0])):
        notes[f"reset_family_{k}"] = f"{SPLIT_TAG[key[0]]}: {len(members)} demos share one dirt_marker_positions layout"
        for j in members:
            eps[j]["reset_family"] = f"{SPLIT_TAG[key[0]]}-dirt{k}"
        for j in members[1:]:
            union(members[0], j)
    # 2. near-duplicate initial state or identical trajectories (any condition)
    dup_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = eps[i], eps[j]
            d_obj = np.linalg.norm(a["obj0_world"] - b["obj0_world"])
            d_eef = np.linalg.norm(a["eef0_world"] - b["eef0_world"])
            d_dirt = np.linalg.norm(a["dirt_centroid"] - b["dirt_centroid"]) if "dirt_centroid" in a else 0.0
            same_state = a["states"].shape == b["states"].shape and np.allclose(a["states"][0], b["states"][0])
            same_act = a["actions"].shape == b["actions"].shape and np.allclose(a["actions"], b["actions"])
            if (d_obj < 0.005 and d_eef < 0.005 and d_dirt < 0.005) or same_state or same_act:
                dup_pairs.append((a["episode_id"], b["episode_id"], float(d_obj), float(d_eef), bool(same_state), bool(same_act)))
                union(i, j)
    roots = {}
    for i, e in enumerate(eps):
        r = find(i)
        roots.setdefault(r, f"g{len(roots):02d}")
        e["group_id"] = roots[r]
    # nearest-neighbour spacing of initial object position (reset-family check)
    P = np.array([e["obj0_world"][:2] for e in eps])
    D = np.linalg.norm(P[:, None] - P[None], axis=-1) + np.eye(n) * 9
    nn = D.min(1)
    return notes, dup_pairs, dict(nn_min=float(nn.min()), nn_median=float(np.median(nn)), nn_max=float(nn.max()),
                                  obj_x_range=[float(P[:, 0].min()), float(P[:, 0].max())],
                                  obj_y_range=[float(P[:, 1].min()), float(P[:, 1].max())])


def assign_partitions(eps, seed=0, n_trials=20000, frac=(0.6, 0.2, 0.2)):
    """Group-level stratified split. Strata = condition x (damaging/safe). Outcome labels are used only for
    stratification balance (standard), never for grouping."""
    groups = sorted({e["group_id"] for e in eps})
    strata = sorted({(e["split"], e["any_damage"]) for e in eps})
    G = np.zeros((len(groups), len(strata)))
    for e in eps:
        G[groups.index(e["group_id"]), strata.index((e["split"], e["any_damage"]))] += 1
    tot = G.sum(0)
    target = np.array(frac)[:, None] * tot[None]
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in range(n_trials):
        a = rng.choice(3, size=len(groups), p=frac)
        C = np.stack([G[a == k].sum(0) for k in range(3)])
        cost = ((C - target) ** 2).sum()
        # held-out splits must contain at least one safe and one damaging episode if the task has them
        for k in (1, 2):
            for dmg in (False, True):
                cols = [s for s, (sp, dd) in enumerate(strata) if dd == dmg]
                if tot[cols].sum() > 0 and C[k, cols].sum() == 0:
                    cost += 1e3
        if cost < best_cost:
            best, best_cost = a, cost
    names = ("train", "dev", "test")
    for e in eps:
        e["partition"] = names[best[groups.index(e["group_id"])]]
    return float(best_cost)


def lag_check(eps):
    """Which observation difference do actions[t, :3] best track? lag 0 => obs[t+1]-obs[t] (obs pre-action)."""
    res = {}
    for L in (-1, 0, 1):
        cs = []
        for e in eps:
            a = e["actions"][:, :3]
            dx = np.diff(e["eef"], axis=0)  # dx[t] = eef[t+1]-eef[t]
            T = len(dx)
            t = np.arange(1, T - 1)
            for k in range(3):
                x, y = a[t, k], dx[t + L, k]
                if x.std() > 1e-6 and y.std() > 1e-9:
                    cs.append(np.corrcoef(x, y)[0, 1])
        res[L] = float(np.mean(cs))
    return res


def quat_flip_count(e):
    q = e["quat"]
    return int((np.sum(q[1:] * q[:-1], axis=1) < 0).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes, schema = {}, {}
    for task in list(ROBOCASA_TASKS) + list(B1K_TASKS):
        for split in SPLITS:
            p = h5path(task, split)
            hashes[f"{task}/{split}"] = {"path": str(p.resolve()), "sha256": sha256(p), "bytes": p.stat().st_size}
            schema[f"{task}/{split}"] = schema_of(p)
    extra_files = sorted(str(x.relative_to(SNAPSHOT)) for x in SNAPSHOT.rglob("*.hdf5")
                         if x.name not in ("with_live_feedback.hdf5", "without_live_feedback.hdf5"))
    for x in extra_files:
        p = SNAPSHOT / x
        hashes[x] = {"path": str(p.resolve()), "sha256": sha256(p), "bytes": p.stat().st_size, "note": "not used"}

    rows, summary = [], {"hashes": hashes, "extra_files_in_snapshot": extra_files, "tasks": {}}
    all_eps = {}
    for task in ROBOCASA_TASKS:
        eps = load_task(task)
        all_eps[task] = eps
        notes, dups, spacing = build_groups(eps)
        cost = assign_partitions(eps)
        lag = lag_check(eps)
        t = {
            "n_episodes": len(eps),
            "per_split": dict(Counter(e["split"] for e in eps)),
            "health_monotone_all": all(e["health_monotone"] for e in eps),
            "health_eq_prev_minus_damageinfo_maxerr": max(e["health_dmg_max_abs_err"] for e in eps),
            "first_damage_health_eq_info": all(e["first_damage_step"] == e["first_damage_step_info"] for e in eps),
            "initial_health_all_100": all(np.allclose(e["initial_health"], 100) for e in eps),
            "dones_any": sum(int(e["dones"].any()) for e in eps),
            "rewards_nonzero": sum(int(np.any(e["rewards"] != 0)) for e in eps),
            "extra_damage_info_entities": sorted({x for e in eps for x in e["extra_damage_entities"]}),
            "quat_sign_flips_total": sum(quat_flip_count(e) for e in eps),
            "action_obs_lag_corr": lag,
            "base_yaw_values": sorted({round(e["base_yaw"], 4) for e in eps}),
            "session_notes": notes,
            "near_duplicate_pairs": dups,
            "init_obj_spacing": spacing,
            "partition_cost": cost,
            "lengths": {sp: [int(min(x)), int(np.median(x)), int(max(x))] for sp in SPLITS
                        for x in [[e["T"] for e in eps if e["split"] == sp]]},
            "damaged_entity_counts": dict(Counter(n for e in eps for n in e["damaged_entities"])),
            "damage_sensitivity": {
                thr: {sp: sum(int(e["max_health_loss"] > thr) for e in eps if e["split"] == sp) for sp in SPLITS}
                for thr in (0.0, 1.0, 5.0, 10.0)
            },
            "severe_health0": {sp: sum(int(e["min_health"].min() <= 0) for e in eps if e["split"] == sp) for sp in SPLITS},
        }
        if task == "pick_egg":
            t["proxy_lift_m_range"] = [min(e["proxy_lift_m"] for e in eps), max(e["proxy_lift_m"] for e in eps)]
        else:
            t["final_proportion_wiped_values"] = sorted({round(e["final_proportion_wiped"], 4) for e in eps})
            t["first_full_wipe_is_last_step"] = sum(int(e["first_full_wipe_step"] == e["T"] - 1) for e in eps)
            t["rewards_eq_prev_proportion_wiped_maxerr"] = max(float(np.abs(e["rewards"][1:] - e["proportion_wiped"][:-1]).max()) for e in eps)
            t["max_reward"] = max(float(e["rewards"].max()) for e in eps)
        # class counts
        cc = defaultdict(Counter)
        for e in eps:
            cc[e["split"]][e["outcome_class"]] += 1
            cc[e["split"]]["damaging"] += int(e["any_damage"])
            cc[e["split"]]["no_damage"] += int(not e["any_damage"])
            cc[e["split"]]["proxy:" + e["outcome_class_proxy"]] += 1
            cc["all"][e["outcome_class"]] += 1
            cc["all"]["damaging"] += int(e["any_damage"])
            cc["all"]["no_damage"] += int(not e["any_damage"])
            cc["all"]["proxy:" + e["outcome_class_proxy"]] += 1
        t["class_counts"] = {k: dict(v) for k, v in cc.items()}
        pc = defaultdict(Counter)
        for e in eps:
            for key in (e["partition"], f"{e['partition']}/{SPLIT_TAG[e['split']]}"):
                pc[key]["n"] += 1
                pc[key]["damaging"] += int(e["any_damage"])
                pc[key]["no_damage"] += int(not e["any_damage"])
                pc[key][e["outcome_class_proxy"]] += 1
        t["partition_counts"] = {k: dict(v) for k, v in sorted(pc.items())}
        t["n_groups"] = len({e["group_id"] for e in eps})
        summary["tasks"][task] = t

        h = hashes
        for e in eps:
            rows.append({
                "task": task,
                "simulator_version": "unknown",
                "dataset_hash": h[f"{task}/{e['split']}"]["sha256"],
                "source_split": e["split"],
                "episode_id": e["episode_id"],
                "length": e["T"],
                "task_success": fmt(e["task_success"]),
                "any_damage": fmt(e["any_damage"]),
                "safe_success": fmt(e["safe_success"]),
                "first_damage_step": fmt(e["first_damage_step"]),
                "damaged_entities": ";".join(e["damaged_entities"]),
                "initial_health": json.dumps([round(float(x), 2) for x in e["initial_health"]]),
                "final_health": json.dumps([round(float(x), 2) for x in e["final_health"]]),
                "observation_keys": ";".join(e["obs_keys"]),
                "action_schema": ROBOCASA_ACTION_SCHEMA,
                "playback_version": "unknown",
                "replay_verified": "False",
                "outcome_class": e["outcome_class"],
                "proxy_task_success": fmt(e["proxy_task_success"]) if task == "pick_egg" else "",
                "proxy_basis": f"egg z rise at final step = {e['proxy_lift_m']:.3f} m (>=0.05 m); NOT a stored success label" if task == "pick_egg" else "",
                "max_health_loss": round(e["max_health_loss"], 3),
                "final_proportion_wiped": round(e["final_proportion_wiped"], 4) if task == "wipe_countertop" else "",
                "session_id": e["session_id"],
                "group_id": e["group_id"],
                "partition": e["partition"],
            })

    for task in B1K_TASKS:
        for split in SPLITS:
            with h5py.File(h5path(task, split), "r") as f:
                for n in demo_names(f):
                    d = f["data"][n]
                    rows.append({
                        "task": task, "simulator_version": "unknown", "dataset_hash": hashes[f"{task}/{split}"]["sha256"],
                        "source_split": split, "episode_id": f"{task}/{SPLIT_TAG[split]}/{n}", "length": int(d["action"].shape[0]),
                        "task_success": "unknown", "any_damage": "unknown", "safe_success": "unknown",
                        "first_damage_step": "unknown", "damaged_entities": "unknown", "initial_health": "unknown",
                        "final_health": "unknown", "observation_keys": "state(flat OmniGibson dump);state_size",
                        "action_schema": B1K_ACTION_SCHEMA, "playback_version": "unknown (raw teleop dump; needs OmniGibson playback)",
                        "replay_verified": "False", "outcome_class": "unknown", "proxy_task_success": "", "proxy_basis": "",
                        "max_health_loss": "unknown", "final_proportion_wiped": "", "session_id": "", "group_id": "", "partition": "not_partitioned",
                    })
        summary["tasks"][task] = {"per_split": {sp: schema[f"{task}/{sp}"]["n_demos"] for sp in SPLITS},
                                  "field_scan": {sp: schema[f"{task}/{sp}"].get("b1k_field_scan") for sp in SPLITS}}

    with open(OUT / "demo_inventory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    splits = {}
    for task, eps in all_eps.items():
        splits[task] = {
            "grouping_rule": "union of (session: identical init_robot_base_pos within condition, else 5-consecutive-demo "
                             "pseudo-session), (reset family: identical ep_meta dirt_marker_positions, wipe only) and "
                             "(near-duplicate initial state: obj & eef init within 5 mm [and dirt centroid within 5 mm], "
                             "identical states[0], or identical actions)",
            "stratification": "condition x any_damage, group-level, seeded random search (seed 0, 20000 trials), 60/20/20",
            "groups": {g: sorted(e["episode_id"] for e in eps if e["group_id"] == g) for g in sorted({e["group_id"] for e in eps})},
            "partitions": {p: sorted((e["episode_id"] for e in eps if e["partition"] == p), key=lambda s: (s.split("/")[1], int(s.split("_")[-1])))
                           for p in ("train", "dev", "test")},
            "counts": summary["tasks"][task]["partition_counts"],
        }
    with open(OUT / "splits.json", "w") as fh:
        json.dump(splits, fh, indent=1)
    with open(OUT / "schema_and_checks.json", "w") as fh:
        json.dump({"schema": schema, **summary}, fh, indent=1, default=str)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("near_duplicate_pairs",)} for k, v in summary["tasks"].items()}, indent=1, default=str))
    for task in ROBOCASA_TASKS:
        print(task, "near-dup pairs:", len(summary["tasks"][task]["near_duplicate_pairs"]))
    print(json.dumps({k: v["sha256"] for k, v in hashes.items()}, indent=1))


if __name__ == "__main__":
    main()
