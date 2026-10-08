"""Analyze replay runs: divergence statistics and replayed labels.

Inputs: <storage>/runs/replay/{force,open}_{released,teleop}/*.npz (from replay.py),
        replay_sample.json (pre-registered), stored teleop HDF5s (read-only).
Outputs (repo): outputs/oopsieverse_full_study/replay/{replay_divergence.json, replay_labels.csv,
        replay_episode_stats.csv}. Stored labels are never modified.
"""

import csv
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, STORAGE, TASKS, CONDITIONS, teleop_path, demo_sort_key, object_final_health, stratum  # noqa: E402

RUN = os.path.join(STORAGE, "runs", "replay")
STRICT_TOL = 1e-3       # health units; operational tolerance for "no loss" (see metric_contract.json)
PAPER_THR = 95.0        # paper: "all object health remaining above 95" (strict >)
CODE_THR = 95.0         # code: mean over objects of final health >= 95 (misc_utils.py:203-225)
TARGET = {"pick_egg": "egg", "wipe_countertop": "sponge"}


def obj_series(H, names):
    objs = {}
    for k, n in enumerate(names):
        o = n.split("@", 1)[0]
        objs[o] = H[:, k] if o not in objs else np.minimum(objs[o], H[:, k])
    return objs


def preds(final):
    v = list(final.values())
    return dict(paper=all(x > PAPER_THR for x in v), code=float(np.mean(v)) >= CODE_THR,
                strict=all(x >= 100.0 - STRICT_TOL for x in v))


def damage_steps(tab, k_links):
    return set(np.nonzero((tab[:, k_links, 0] > 0).any(axis=1))[0].tolist())


def episode_stats(z, mode):
    names = [str(n) for n in z["link_names"]]
    st_tab = z["st_tab"][1:]
    rp_tab = z["rp_tab"]
    st_h = z["st_health"]
    rp_h = z["rp_health"]
    n = len(rp_h)
    out = {}
    # per-step damage and force agreement (all links)
    out["max_abs_step_damage_diff"] = float(np.nanmax(np.abs(st_tab[:, :, 0] - rp_tab[:, :, 0])))
    out["max_abs_health_diff"] = float(np.max(np.abs(st_h[1:] - rp_h)))
    m = st_tab[:, :, 1] > 0.05
    out["impact_ratio_median"] = float(np.median(rp_tab[:, :, 1][m] / st_tab[:, :, 1][m])) if m.any() else None
    out["max_abs_qs_diff_N"] = float(np.nanmax(np.abs(st_tab[:, :, 2] - rp_tab[:, :, 2])))
    allk = list(range(len(names)))
    a, b = damage_steps(st_tab, allk), damage_steps(rp_tab, allk)
    out["damage_steps_stored"] = len(a)
    out["damage_steps_replay"] = len(b)
    out["damage_step_jaccard"] = (len(a & b) / len(a | b)) if (a | b) else 1.0
    out["first_damage_step_stored"] = min(a) + 1 if a else None
    out["first_damage_step_replay"] = min(b) + 1 if b else None
    # final object health: stored vs replay (absolute, official playback convention) and anchored
    fs = object_final_health(st_h[-1], names)
    fr = object_final_health(rp_h[-1], names)
    anchored = st_h[0] - np.cumsum(rp_tab[:, :, 0], axis=0)[-1]
    fa = object_final_health(np.maximum(anchored, 0.0), names)
    out["final_obj_health_stored"] = fs
    out["final_obj_health_replay"] = fr
    out["final_obj_health_replay_anchored"] = fa
    out["max_abs_final_obj_health_diff"] = max(abs(fs[o] - fr[o]) for o in fs)
    out["labels_agree"] = {k: preds(fs)[k] == preds(fr)[k] for k in ["paper", "code", "strict"]}
    # state / obs divergence
    se = z["rp_state_err"]
    out["state_err_median"] = float(np.median(se))
    out["state_err_max"] = float(se.max())
    out["state_err_final"] = float(se[-1])
    out["qpos_err_final"] = float(z["rp_qpos_err"][-1])
    out["qpos_err_max"] = float(z["rp_qpos_err"].max())
    for k in ["robot0_eef_pos", "object-state"]:
        e = z[f"obs_err__{k}"]
        out[f"{k}_err_final"] = float(e[-1])
        out[f"{k}_err_max"] = float(e.max())
    obj_pos_err = z["obs_err__object-state"]  # norm over the 14-D object-state (pos, quat, rel pos, rel quat)
    over = np.nonzero(obj_pos_err > 0.01)[0]
    out["first_step_objstate_err_gt_1cm"] = int(over[0]) + 1 if len(over) else None
    out["success_final"] = bool(z["rp_success"][-1])
    out["success_ever"] = bool(z["rp_success"].any())
    out["first_success_step"] = int(np.argmax(z["rp_success"])) + 1 if z["rp_success"].any() else None
    out["reward_max_abs_diff"] = float(np.max(np.abs(z["rp_reward"] - z["st_rewards"][1:])))
    out["wiped_final"] = int(z["rp_wiped"][-1])
    out["n_steps"] = n
    return out


def summarize(rows, keys):
    s = {"n": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
        if len(v):
            s[k] = dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(v.max()))
    for k in ["paper", "code", "strict"]:
        s[f"label_agree_{k}"] = f"{sum(r['labels_agree'][k] for r in rows)}/{len(rows)}"
    s["success_final"] = f"{sum(r['success_final'] for r in rows)}/{len(rows)}"
    return s


def counter_surface_z():
    """Counter top height for pick_egg from the pinned env + stored model (same fixed layout in all demos)."""
    from replay import get_env, setup_episode
    env = get_env("pick_egg")
    vals = []
    for cond in CONDITIONS:
        with h5py.File(teleop_path("pick_egg", cond), "r") as f:
            setup_episode(env, f["data"]["demo_0"])
            vals.append(float(env.counter.pos[2] + env.counter.height / 2))
    assert abs(vals[0] - vals[1]) < 1e-9, vals
    return vals[0]


def main():
    sample = json.load(open(os.path.join(OUT, "replay", "replay_sample.json")))
    in_sample = {(t, r["cond"], r["demo"]): r["stratum"] for t, rs in sample["sample"].items() for r in rs}
    runs = [d for d in ["force_teleop", "open_teleop", "force_released", "open_released"] if os.path.isdir(os.path.join(RUN, d))]
    stats = {}
    for d in runs:
        mode = d.split("_")[0]
        for fn in sorted(os.listdir(os.path.join(RUN, d))):
            t, c, demo = fn[:-4].split("__")
            stats[(d, t, c, demo)] = episode_stats(np.load(os.path.join(RUN, d, fn)), mode)

    keys = ["max_abs_step_damage_diff", "max_abs_health_diff", "impact_ratio_median", "max_abs_qs_diff_N",
            "damage_step_jaccard", "max_abs_final_obj_health_diff", "state_err_median", "state_err_max",
            "state_err_final", "qpos_err_final", "robot0_eef_pos_err_final", "object-state_err_final",
            "first_step_objstate_err_gt_1cm", "reward_max_abs_diff"]
    div = {"strict_tol": STRICT_TOL, "tables": {}}
    for d in runs:
        for t in TASKS:
            for subset in ["sample", "all"]:
                rows = [v for (dd, tt, c, demo), v in stats.items() if dd == d and tt == t and (subset == "all" or (t, c, demo) in in_sample)]
                if rows:
                    div["tables"][f"{d}/{t}/{subset}"] = summarize(rows, keys)
            for stt in ["zero", "mild", "substantial"]:
                rows = [v for (dd, tt, c, demo), v in stats.items() if dd == d and tt == t and in_sample.get((t, c, demo)) == stt]
                if rows:
                    div["tables"][f"{d}/{t}/sample_{stt}"] = summarize(rows, keys)

    # pick_egg height threshold variants from stored/replayed egg z
    cz = counter_surface_z()
    div["pick_egg_counter_surface_z"] = cz

    # labels (primary replay = force_teleop: official state-forcing playback with collection-time gripper)
    label_rows = []
    for t in TASKS:
        for c in CONDITIONS:
            with h5py.File(teleop_path(t, c), "r") as f:
                for demo in sorted(f["data"], key=demo_sort_key):
                    g = f["data"][demo]
                    names = [n.decode() if isinstance(n, bytes) else str(n) for n in g.attrs["health_list_link_names"]]
                    key = ("force_teleop", t, c, demo)
                    if key not in stats:
                        continue
                    s = stats[key]
                    z = np.load(os.path.join(RUN, "force_teleop", f"{t}__{c}__{demo}.npz"))
                    st_final = object_final_health(g["obs/health"][-1], names)
                    fr = s["final_obj_health_replay"]
                    tgt = TARGET[t]
                    rob = [o for o in fr if o != tgt][0]
                    tab = z["rp_tab"]
                    tgt_k = [k for k, n in enumerate(names) if n.startswith(tgt + "@")]
                    rp_objs = obj_series(z["rp_health"], names)
                    avg_env_health_time = float(np.mean(np.mean(np.stack(list(rp_objs.values())), axis=0)))
                    succ = s["success_final"]
                    pr = preds(fr)
                    ps = preds(st_final)
                    row = dict(task=t, cond=c, demo=demo, T=int(g["actions"].shape[0]),
                               in_sample=(t, c, demo) in in_sample, sample_stratum=in_sample.get((t, c, demo), ""),
                               stored_stratum=stratum(max(100 - v for v in st_final.values())),
                               official_success=succ, official_success_ever=s["success_ever"],
                               first_success_step=s["first_success_step"],
                               success_released_playback=stats.get(("force_released", t, c, demo), {}).get("success_final"),
                               success_open_loop_teleop_gripper=stats.get(("open_teleop", t, c, demo), {}).get("success_final"),
                               **{f"final_health_{tgt}": round(fr[tgt], 4), f"final_health_{rob}": round(fr[rob], 4),
                                  f"stored_final_health_{tgt}": round(st_final[tgt], 4), f"stored_final_health_{rob}": round(st_final[rob], 4)},
                               worst_entity_loss=round(max(100 - v for v in fr.values()), 4),
                               stored_worst_entity_loss=round(max(100 - v for v in st_final.values()), 4),
                               env_health_code_mean_final=round(float(np.mean(list(fr.values()))), 4),
                               avg_env_health_over_time=round(avg_env_health_time, 4),
                               paper_rule_all_obj_gt95=pr["paper"], code_rule_mean_ge95=pr["code"], strict_no_loss=pr["strict"],
                               primary_safe_success=succ and pr["paper"], secondary_safe_success=succ and pr["strict"],
                               compat_safe_success=succ and pr["code"],
                               stored_paper_rule=ps["paper"], stored_code_rule=ps["code"], stored_strict=ps["strict"],
                               max_contact_force_N=round(float(np.nanmax(tab[:, :, 3])), 3),
                               max_contact_force_target_N=round(float(np.nanmax(tab[:, tgt_k, 3])), 3),
                               max_impact_force_target_N=round(float(np.nanmax(tab[:, tgt_k, 1])), 3),
                               max_qs_force_N=round(float(np.nanmax(tab[:, :, 2])), 3),
                               first_damage_step=s["first_damage_step_replay"],
                               replay_vs_stored_final_health_absdiff=round(s["max_abs_final_obj_health_diff"], 4))
                    if t == "pick_egg":
                        ez = g["obs/egg_pos"][:, 2].astype(float) - cz
                        row["stored_egg_height_final_m"] = round(float(ez[-1]), 4)
                        row["stored_success_h005_final"] = bool(ez[-1] >= 0.05)
                        row["stored_success_h010_final"] = bool(ez[-1] >= 0.10)
                        row["stored_success_h010_ever"] = bool((ez >= 0.10).any())
                    else:
                        row["wiped_final"] = s["wiped_final"]
                        row["stored_reward_final"] = float(g["rewards"][-1])
                    q = []
                    if s["success_final"] != s["success_ever"]:
                        q.append("success_not_final")
                    row["quarantine"] = ";".join(q)
                    label_rows.append(row)
    os.makedirs(os.path.join(OUT, "replay"), exist_ok=True)
    cols = []
    for r in label_rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(os.path.join(OUT, "replay", "replay_labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(label_rows)
    # class counts
    counts = {}
    for t in TASKS:
        for c in CONDITIONS + ["all"]:
            rs = [r for r in label_rows if r["task"] == t and (c == "all" or r["cond"] == c)]
            if not rs:
                continue
            cc = {"n": len(rs), "official_success": sum(r["official_success"] for r in rs)}
            for rule, col in [("primary_paper", "paper_rule_all_obj_gt95"), ("secondary_strict", "strict_no_loss"), ("compat_code", "code_rule_mean_ge95")]:
                cc[rule] = dict(safe_success=sum(r["official_success"] and r[col] for r in rs),
                                damaging_success=sum(r["official_success"] and not r[col] for r in rs),
                                safe_failure=sum((not r["official_success"]) and r[col] for r in rs),
                                damaging_failure=sum((not r["official_success"]) and not r[col] for r in rs))
            for rule, rcol, scol in [("paper", "paper_rule_all_obj_gt95", "stored_paper_rule"),
                                     ("code", "code_rule_mean_ge95", "stored_code_rule"),
                                     ("strict", "strict_no_loss", "stored_strict")]:
                cc[f"label_flip_vs_stored_{rule}"] = sum(r[rcol] != r[scol] for r in rs)
            if t == "pick_egg":
                cc["proxy_success_h005_stored_final"] = sum(r["stored_success_h005_final"] for r in rs)
                cc["proxy_success_and_official"] = sum(r["stored_success_h005_final"] and r["official_success"] for r in rs)
                cc["success_h010_final_stored"] = sum(r["stored_success_h010_final"] for r in rs)
            counts[f"{t}/{c}"] = cc
    div["class_counts_force_teleop"] = counts
    div["episodes"] = {"|".join(k): v for k, v in stats.items()}
    with open(os.path.join(OUT, "replay", "replay_divergence.json"), "w") as f:
        json.dump(div, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in div["tables"].items() if k.endswith("/sample") or k.endswith("/all")}, indent=1, default=float)[:12000])
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
