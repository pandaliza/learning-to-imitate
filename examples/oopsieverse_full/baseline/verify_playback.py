"""Verify official-playback HDF5 parts against replay/replay_labels.csv (force replay, teleop gripper).

Per episode: final task_completion vs official_success, per-entity final health vs replay labels,
the three predicates, action == teleop actions[1:], lengths (obs T+1, action T), seg class vocabulary,
file size. Writes outputs/oopsieverse_full_study/baseline/playback_verification.json.
"""
import csv
import glob
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_data import PARTS, STUDY, sha256_file  # noqa: E402

TELEOP = "/edrive1/lizad/oopsieverse_full_study/data/oopsieverse-demos/paper_experiments/teleop_data"


def main():
    labels = {}
    for r in csv.DictReader(open(os.path.join(STUDY, "replay/replay_labels.csv"))):
        labels[(r["task"], r["cond"], r["demo"])] = r
    rows = []
    for (task, cond, demo), lab in sorted(labels.items()):
        p = os.path.join(PARTS, f"{task}__{cond}__{demo}.hdf5")
        if not os.path.exists(p):
            rows.append({"task": task, "cond": cond, "demo": demo, "missing": True})
            continue
        with h5py.File(p, "r") as f, h5py.File(os.path.join(TELEOP, task, f"{cond}_live_feedback.hdf5"), "r") as t:
            g = f["data"][demo]
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in g.attrs["health_list_link_names"]]
            H = g["obs/health"][:].astype(np.float64)
            ents = {}
            for k, n in enumerate(names):
                ents.setdefault(n.split("@")[0], []).append(k)
            final = {o: float(H[-1, ks].min()) for o, ks in ents.items()}
            tc = g["task_completion"][:]
            act = g["action"][:]
            tel = t["data"][demo]["actions"][1:]
            T = act.shape[0]
            lens_ok = all(g["obs"][k].shape[0] == T + 1 for k in ["proprio", "health", "cam/robot0_eye_in_hand/seg"]) \
                and g["info/obs_info"].shape[0] == T + 1 and tc.shape[0] == T + 1
            vocab = sorted(set(json.loads(g["info/obs_info"][0])["robot0"]["robot0_eye_in_hand"]["seg_instance"].values()))
            succ = bool(tc[-1])
            paper = succ and all(v > 95.0 for v in final.values())
            strict = succ and bool(np.all(H[-1] >= 100 - 1e-3))
            compat = succ and float(np.mean(list(final.values()))) >= 95.0
            lab_ent = {o: float(lab[f"final_health_{o}"]) for o in final if lab.get(f"final_health_{o}") not in (None, "")}
            rows.append({
                "task": task, "cond": cond, "demo": demo, "T": T, "T_label": int(lab["T"]),
                "lens_ok": lens_ok, "action_maxabs_diff_vs_teleop": float(np.abs(act - tel).max()),
                "final_success": succ, "label_success": lab["official_success"] == "True",
                "first_success_idx": int(np.argmax(tc)) if tc.any() else None,
                "final_health": final, "label_final_health": lab_ent,
                "max_health_absdiff": max(abs(final[o] - lab_ent[o]) for o in lab_ent) if lab_ent else None,
                "paper": paper, "strict": strict, "compat": compat,
                "label_paper": lab["primary_safe_success"] == "True", "label_strict": lab["secondary_safe_success"] == "True",
                "label_compat": lab["compat_safe_success"] == "True",
                "size_bytes": os.path.getsize(p), "seg_shape": list(g["obs/cam/robot0_eye_in_hand/seg"].shape[1:]),
                "seg_vocab": vocab,
            })
    ok = [r for r in rows if not r.get("missing")]
    summ = {}
    for task in sorted({r["task"] for r in ok}):
        R = [r for r in ok if r["task"] == task]
        summ[task] = {
            "n": len(R),
            "success_match": sum(r["final_success"] == r["label_success"] for r in R),
            "paper_match": sum(r["paper"] == r["label_paper"] for r in R),
            "strict_match": sum(r["strict"] == r["label_strict"] for r in R),
            "compat_match": sum(r["compat"] == r["label_compat"] for r in R),
            "max_final_health_absdiff": max(r["max_health_absdiff"] or 0 for r in R),
            "lens_ok": sum(r["lens_ok"] for r in R),
            "actions_equal_teleop_1plus": sum(r["action_maxabs_diff_vs_teleop"] == 0 for r in R),
            "T_match": sum(r["T"] + 1 == r["T_label"] or r["T"] == r["T_label"] for r in R),
            "n_steps_total": int(sum(r["T"] for r in R)),
            "size_GB": round(sum(r["size_bytes"] for r in R) / 1e9, 2),
            "bytes_per_state": round(sum(r["size_bytes"] for r in R) / sum(r["T"] + 1 for r in R)),
            "paper_safe_successes": sum(r["paper"] for r in R),
            "seg_shape": R[0]["seg_shape"],
            "mismatches": [(r["cond"], r["demo"]) for r in R if r["final_success"] != r["label_success"] or
                           r["paper"] != r["label_paper"] or r["strict"] != r["label_strict"] or r["compat"] != r["label_compat"]],
        }
    out = {"generated_by": "examples/oopsieverse_full/baseline/verify_playback.py", "parts_dir": PARTS,
           "missing": [(r["task"], r["cond"], r["demo"]) for r in rows if r.get("missing")],
           "summary": summ, "episodes": rows}
    json.dump(out, open(os.path.join(STUDY, "baseline/playback_verification.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
