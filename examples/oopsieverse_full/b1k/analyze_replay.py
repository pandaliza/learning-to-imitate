"""Summarise official-playback output HDF5s (health, damage events, thermal, success).

Usage: python -I analyze_replay.py --task shelve_item --playback out.hdf5 [--metrics m.json] --out summary.json
Reads only; no simulator import.
"""
import argparse
import json

import h5py
import numpy as np

TOL = 1e-6


def summarise_demo(g, task):
    out = {"datasets": sorted(g.keys())}
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in g.attrs.get("health_list_link_names", [])]
    out["n_health_links"] = len(names)
    if "obs" in g and "health" in g["obs"]:
        h = np.asarray(g["obs"]["health"], dtype=np.float64)  # (T+1, L)
        out["T_recorded"] = int(h.shape[0])
        loss_final = h[0] - h[-1]
        loss_max = h[0] - h.min(axis=0)
        per_link = {names[i]: round(float(loss_max[i]), 4) for i in np.where(loss_max > TOL)[0]}
        per_obj = {}
        for i, n in enumerate(names):
            o = n.split("@")[0]
            per_obj[o] = min(per_obj.get(o, 100.0), float(h[-1, i]))
        out["initial_health_min"] = float(h[0].min())
        out["final_health_per_object"] = {k: round(v, 3) for k, v in per_obj.items()}
        out["damaged_links_maxloss"] = per_link
        out["worst_entity_loss"] = float(loss_max.max()) if loss_max.size else 0.0
        out["health_increase_anywhere"] = bool((np.diff(h, axis=0) > TOL).any())
        dec = (np.diff(h, axis=0) < -TOL)
        out["damage_event_steps"] = int(dec.any(axis=1).sum())
        out["damage_event_link_steps"] = int(dec.sum())
        out["first_damage_step"] = int(np.argmax(dec.any(axis=1))) if dec.any() else None
        # predicates (operational, for the report only; not the frozen metric contract)
        obj_final = list(per_obj.values())
        out["paper_style_all_objects_gt95"] = bool(all(v > 95 for v in obj_final)) if obj_final else None
        out["code_style_mean_ge95"] = bool(np.mean(obj_final) >= 95) if obj_final else None
        out["strict_no_loss"] = bool(out["worst_entity_loss"] <= TOL)
    if "task_completion" in g:
        tc = np.asarray(g["task_completion"]).reshape(-1).astype(bool)
        out["task_completion_final"] = bool(tc[-1]) if tc.size else None
        out["task_completion_any"] = bool(tc.any())
        out["first_success_step"] = int(np.argmax(tc)) if tc.any() else None
    if "info" in g and "damage_info" in g["info"]:
        di = g["info"]["damage_info"]
        tmax, tlinks, mech_max = {}, set(), {}
        for i in range(len(di)):
            d = json.loads(di[i].decode() if isinstance(di[i], bytes) else di[i])
            for obj, links in (d or {}).items():
                if not isinstance(links, dict):
                    continue
                for link, ev in links.items():
                    if not isinstance(ev, dict):
                        continue
                    th_ = ev.get("thermal") or {}
                    t = th_.get("temperature")
                    if t is not None:
                        try:
                            t = float(t)
                            k = f"{obj}@{link}"
                            tmax[k] = max(tmax.get(k, -1e9), t)
                        except (TypeError, ValueError):
                            pass
                        if float(th_.get("damage", 0) or 0) > 0:
                            tlinks.add(f"{obj}@{link}")
                    me = ev.get("mechanical") or {}
                    for fk in ("impact_forces", "filtered_qs_forces"):
                        v = me.get(fk)
                        if v is not None:
                            try:
                                mech_max[fk] = max(mech_max.get(fk, 0.0), float(v))
                            except (TypeError, ValueError):
                                pass
        if tmax:
            top = sorted(tmax.items(), key=lambda kv: -kv[1])[:6]
            out["max_temperature_top"] = {k: round(v, 2) for k, v in top}
        out["thermal_damage_links"] = sorted(tlinks)
        out["max_contact_force"] = {k: round(v, 2) for k, v in mech_max.items()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--playback", nargs="+", required=True)
    p.add_argument("--metrics", nargs="*", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = {"task": a.task, "files": {}}
    for path in a.playback:
        with h5py.File(path, "r") as f:
            res["files"][path] = {k: summarise_demo(f["data"][k], a.task) for k in sorted(f["data"].keys()) if k.startswith("demo_")}
    for m in a.metrics:
        with open(m) as f:
            res.setdefault("timing", {})[m] = json.load(f)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
