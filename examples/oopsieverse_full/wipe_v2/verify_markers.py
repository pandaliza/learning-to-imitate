"""Verification of the sim-derived wiped-marker identities (W3 target input).
For every demo with a markers npz: (1) sim per-step counts == playback info/newly_wiped (exact), (2) EEF site world
position == proprio[t+1,14:17], (3) at each wipe event (info index i, state i+1): distances of the wiped markers to
the EEF and to the sponge, and whether the wiped markers are the |S| nearest to the EEF among the markers not yet
wiped before that event (rank statistics). Writes reports/wipe_v2/verification/marker_verification.json."""
import glob
import json
import os
import sys

import numpy as np

OUT = "/edrive1/lizad/oopsieverse_full_study/reports/wipe_v2/verification/marker_verification.json"


def q(x):
    x = np.asarray(x)
    return {"n": int(len(x)), "min": float(x.min()), "p5": float(np.percentile(x, 5)), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "max": float(x.max()), "mean": float(x.mean())}


def main():
    files = sorted(glob.glob("/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/markers/*.npz"))
    per, d_eef, d_eef_xy, d_sp_xy, rank_ok, max_rank, nn_gap = {}, [], [], [], [], [], []
    for p in files:
        z = np.load(p)
        name = os.path.basename(p)[:-4]
        mp, ws, ee, sp = z["marker_pos_world"], z["wiped_step"], z["eef_pos_world"], z["sponge_pos"]
        ev = sorted(set(ws[ws >= 0].tolist()))
        n_ok, ranks = 0, []
        for i in ev:
            S = np.nonzero(ws == i)[0]
            remaining = np.nonzero((ws == -1) | (ws >= i))[0]          # not wiped before event i
            dist = np.linalg.norm(mp[remaining, :2] - ee[i, :2], axis=1)
            order = remaining[np.argsort(dist)]
            rk = np.array([int(np.nonzero(order == m)[0][0]) for m in S])  # 0-based rank by EEF xy distance
            ranks.append(int(rk.max()))
            n_ok += int(rk.max() < len(S))
            for m in S:
                d_eef.append(float(np.linalg.norm(mp[m] - ee[i])))
                d_eef_xy.append(float(np.linalg.norm(mp[m, :2] - ee[i, :2])))
                d_sp_xy.append(float(np.linalg.norm(mp[m, :2] - sp[i, :2])))
            others = np.setdiff1d(remaining, S)
            if len(others):
                nn_gap.append(float(np.linalg.norm(mp[others, :2] - ee[i, :2], axis=1).min() -
                                    np.linalg.norm(mp[S, :2] - ee[i, :2], axis=1).max()))
        rank_ok.append(n_ok)
        max_rank += ranks
        per[name] = {"events": len(ev), "markers_wiped": int((ws >= 0).sum()), "counts_equal": bool(z["counts_equal"]),
                     "eef_vs_proprio_maxabs_m": float(z["eef_vs_proprio_maxabs"]),
                     "events_wiped_are_nearest_to_eef": n_ok, "max_rank_excess": int(max(ranks) if ranks else 0)}
    n_ev = sum(v["events"] for v in per.values())
    res = {"n_demos": len(per), "n_events": n_ev, "n_markers_wiped": len(d_eef),
           "all_counts_equal": all(v["counts_equal"] for v in per.values()),
           "eef_vs_proprio_maxabs_m": max(v["eef_vs_proprio_maxabs_m"] for v in per.values()),
           "events_where_wiped_set_is_the_nearest_to_eef": sum(rank_ok),
           "frac_events_nearest": sum(rank_ok) / max(n_ev, 1),
           "max_rank_of_wiped_marker_by_eef_xy_dist (0-based)": q(max_rank),
           "dist_wiped_marker_to_eef_3d_m": q(d_eef), "dist_wiped_marker_to_eef_xy_m": q(d_eef_xy),
           "dist_wiped_marker_to_sponge_xy_m": q(d_sp_xy),
           "gap_nearest_unwiped_minus_farthest_wiped_xy_m": q(nn_gap) if nn_gap else None,
           "per_demo": per}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_demo"}, indent=1))


if __name__ == "__main__":
    main()
