"""Stage 0B: comparable safe/damaging pairs, intent-representation check, figures, optional probe.

Rules are imported from frozen_rules.py (written before this analysis ran).
Usage: .venv/bin/python examples/oopsieverse_audit/pair_intent_analysis.py
"""

import hashlib
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

import frozen_rules as FR  # noqa: E402
from common import FIG, OUT, OBS_KEYS, ROBOCASA_TASKS, load_task, normalize, repo_intent_at, repo_intent_normalizer, repo_obs  # noqa: E402

H = FR.H
SAFE_C, DMG_C, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#8a8a8a"
METRICS = [m for m, _ in FR.METRICS]
rng_global = np.random.default_rng(0)


# ----------------------------------------------------------------------------- stage / windows
def phase_progress(ep, t):
    if t < ep["grasp_onset"]:
        return "approach", float(ep["eef_obj_dist"][t])
    if ep["task"] == "wipe_countertop":
        return "manipulate", float(ep["proportion_wiped"][t])
    return "manipulate", float(ep["obj_lift"][t])


def window_ok(ep, t):
    if t is None or t < 1 or t + H >= ep["T"]:
        return False
    if ep["any_damage"] and t + H >= ep["first_damage_step"]:
        return False  # future window must end strictly before own first damage
    return True


def match_time(ep, phase, prog):
    if phase == "approach":
        cand = [t for t in range(1, min(ep["grasp_onset"], ep["T"])) if ep["eef_obj_dist"][t] <= prog]
        t = cand[0] if cand else ep["grasp_onset"] - 1
    else:
        arr = ep["proportion_wiped"] if ep["task"] == "wipe_countertop" else ep["obj_lift"]
        cand = [t for t in range(ep["grasp_onset"], ep["T"]) if arr[t] >= prog - 1e-9]
        t = cand[0] if cand else None
    return t if window_ok(ep, t) else None


def anchor_time(U, lead):
    t = U["first_damage_step"] - 1 - H - lead
    return t if window_ok(U, t) else None


def feats(ep, t, norm):
    P = ep["eef"][t + 1 : t + 1 + H]
    q = ep["quat"][t + 1 : t + 1 + H].mean(0)
    return dict(
        t=t,
        intent=repo_intent_at(ep, t, norm),
        P=P,
        qmean=q / np.linalg.norm(q),
        v=ep["speed"][t + 1 : t + 1 + H],
        a6=ep["grip_cmd"][t : t + H],
        w=ep["grip_width"][t + 1 : t + 1 + H],
        F=float(ep["obj_force"][t + 1 : t + 1 + H].max()),
    )


def dist(a, b):
    Pa, Pb = a["P"] - a["P"].mean(0), b["P"] - b["P"].mean(0)
    return dict(
        intent_norm=float(np.linalg.norm(a["intent"] - b["intent"])),
        intent_pos_m=float(np.linalg.norm(a["P"].mean(0) - b["P"].mean(0))),
        intent_rot_deg=float(np.degrees(2 * np.arccos(min(1.0, abs(float(a["qmean"] @ b["qmean"])))))),
        path_shape_mm=float(np.linalg.norm(Pa - Pb, axis=1).mean() * 1000),
        path_abs_mm=float(np.linalg.norm(a["P"] - b["P"], axis=1).mean() * 1000),
        speed_mm=float(np.abs(a["v"] - b["v"]).mean() * 1000),
        grip_cmd=float(np.abs(a["a6"] - b["a6"]).mean()),
        grip_width_mm=float(np.abs(a["w"] - b["w"]).mean() * 1000),
        obj_force_N=float(abs(a["F"] - b["F"])),
    )


# ----------------------------------------------------------------------------- similarity
def geo(a, b):
    return float(np.linalg.norm(a["obj0_world"][:2] - b["obj0_world"][:2])), float(np.linalg.norm(a["eef0_world"] - b["eef0_world"]))


def comparable(a, b):
    if a["split"] != b["split"]:
        return False
    r = FR.SIMILARITY[a["task"]]
    if r.get("same_reset_family") and a.get("reset_family") != b.get("reset_family"):
        return False
    do, de = geo(a, b)
    return do <= r["max_obj_xy_m"] and de <= r["max_eef0_m"]


def success(e):
    return bool(e["proxy_task_success"])


# ----------------------------------------------------------------------------- analysis
def auc(pos, neg):
    """P(pos > neg), ties 0.5."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((pos > neg) + 0.5 * (pos == neg)).mean())


def boot_ci(vals, n=2000, seed=0):
    vals = np.asarray(vals)
    if len(vals) == 0:
        return [float("nan")] * 2
    r = np.random.default_rng(seed)
    m = [vals[r.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def two_way_ci(records, n=2000, seed=1):
    """Bootstrap resampling damaging anchors AND safe comparator episodes (safe episodes are reused across
    anchors, so an anchor-only bootstrap is too narrow)."""
    if not records:
        return {}
    safe_ids = sorted({x["_id"] for ds, _ in records for x in ds})
    r = np.random.default_rng(seed)
    out = defaultdict(list)
    for _ in range(n):
        A = r.integers(0, len(records), len(records))
        wS = np.bincount(r.integers(0, len(safe_ids), len(safe_ids)), minlength=len(safe_ids))
        wmap = dict(zip(safe_ids, wS))
        for m in METRICS:
            vals = []
            for a in A:
                ds, dd = records[a]
                w = np.array([wmap[x["_id"]] for x in ds], dtype=float)
                if w.sum() == 0:
                    continue
                pos = np.array([x[m] for x in ds])[:, None]
                neg = np.array([x[m] for x in dd])[None, :]
                cmp_ = ((pos > neg) + 0.5 * (pos == neg)).mean(1)
                vals.append((w * cmp_).sum() / w.sum())
            if vals:
                out[m].append(np.mean(vals))
    return {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in out.items()}


def analyse_task(task, eps, norm):
    safe = [e for e in eps if success(e) and not e["any_damage"]]
    dmg = [e for e in eps if success(e) and e["any_damage"]]
    res = {"n_safe": len(safe), "n_damaging": len(dmg)}

    # --- eligible edges and one-to-one matching
    edges = []
    for i, S in enumerate(safe):
        for j, U in enumerate(dmg):
            if comparable(S, U):
                do, de = geo(S, U)
                edges.append((i, j, do / FR.SIMILARITY[task]["max_obj_xy_m"] + de / FR.SIMILARITY[task]["max_eef0_m"]))
    res["eligible_edges"] = len(edges)
    res["eligible_edges_by_split"] = dict(
        sorted(defaultdict(int, {}).items())
    )
    by_split = defaultdict(int)
    for i, j, c in edges:
        by_split[safe[i]["split"]] += 1
    res["eligible_edges_by_split"] = dict(by_split)
    pairs = []
    if edges:
        C = np.full((len(safe), len(dmg)), 1e6)
        for i, j, c in edges:
            C[i, j] = c
        ri, cj = linear_sum_assignment(C)
        for i, j in zip(ri, cj):
            if C[i, j] < 1e6:
                pairs.append((safe[i], dmg[j], float(C[i, j])))
    pairs.sort(key=lambda p: p[2])
    res["matched_pairs"] = len(pairs)

    # --- per-pair window metrics (lead 0/8/16)
    pair_rows = []
    for S, U, cost in pairs:
        for lead in FR.STAGE["lead_windows"]:
            tU = anchor_time(U, lead)
            row = dict(safe=S["episode_id"], damaging=U["episode_id"], geo_cost=cost, lead=lead, split=S["split"])
            if tU is None:
                row["status"] = "no_valid_anchor"
                pair_rows.append(row)
                continue
            ph, pr = phase_progress(U, tU)
            tS = match_time(S, ph, pr)
            row.update(t_U=tU, d_U=U["first_damage_step"], phase=ph, progress=pr, t_S=tS)
            if tS is None:
                row["status"] = "no_stage_match"
            else:
                row["status"] = "ok"
                row.update(dist(feats(S, tS, norm), feats(U, tU, norm)))
            pair_rows.append(row)
    res["pair_rows"] = pair_rows

    # --- anchor-discrimination AUC (null = damaging comparators under the same rule)
    auc_out = {}
    dd_dists = defaultdict(list)  # damaging-vs-damaging distances at lead 0 (primary rule) for counterexamples
    for rule in ("geometric", "any_geometry"):
        for lead in FR.STAGE["lead_windows"]:
            per_metric = defaultdict(list)
            records = []
            n_anchor, excluded = 0, 0
            phases = defaultdict(int)
            for U in dmg:
                tU = anchor_time(U, lead)
                if tU is None:
                    continue
                ph, pr = phase_progress(U, tU)
                fU = feats(U, tU, norm)
                ds, dd = [], []
                for Cc in eps:
                    if Cc is U or not success(Cc) or Cc["split"] != U["split"]:
                        continue
                    if rule == "geometric" and not comparable(Cc, U):
                        continue
                    tC = match_time(Cc, ph, pr)
                    if tC is None:
                        excluded += 1
                        continue
                    x = dist(feats(Cc, tC, norm), fU)
                    x["_id"] = Cc["episode_id"]
                    (dd if Cc["any_damage"] else ds).append(x)
                if rule == "geometric" and lead == 0:
                    for x in dd:
                        for m in METRICS:
                            dd_dists[m].append(x[m])
                if not ds or not dd:
                    continue
                n_anchor += 1
                phases[ph] += 1
                records.append((ds, dd))
                for m in METRICS:
                    per_metric[m].append(auc([x[m] for x in ds], [x[m] for x in dd]))
            auc_out[f"{rule}/lead{lead}"] = {
                "n_anchors": n_anchor,
                "n_distinct_safe_comparators": len({x["_id"] for ds, _ in records for x in ds}),
                "ci95_two_way": two_way_ci(records),
                "anchor_phases": dict(phases),
                "excluded_comparator_windows": excluded,
                **{m: {"auc": float(np.mean(v)), "ci95": boot_ci(v)} for m, v in per_metric.items()},
            }
    res["auc"] = auc_out

    # --- counterexamples among matched pairs (lead 0)
    ce = []
    if dd_dists["intent_norm"]:
        med_int = float(np.median(dd_dists["intent_norm"]))
        q75 = {m: float(np.percentile(dd_dists[m], 75)) for m in METRICS}
        res["dd_reference"] = {"median_intent_norm": med_int, "q75": q75, "n": len(dd_dists["intent_norm"]),
                               "median": {m: float(np.median(dd_dists[m])) for m in METRICS}}
        for r in pair_rows:
            if r.get("status") != "ok" or r["lead"] != 0:
                continue
            big = [m for m in ("speed_mm", "grip_cmd", "path_shape_mm", "obj_force_N") if r[m] >= q75[m]]
            if r["intent_norm"] <= med_int and big:
                ce.append(dict(safe=r["safe"], damaging=r["damaging"], intent_norm=r["intent_norm"], large=big))
    ce.sort(key=lambda x: x["intent_norm"])
    res["counterexamples"] = ce
    if dd_dists["intent_norm"]:
        ok = [r for r in pair_rows if r.get("status") == "ok" and r["lead"] == 0]
        res["pairs_intent_similar"] = sum(int(r["intent_norm"] <= med_int) for r in ok)
    ok0 = [r for r in pair_rows if r.get("status") == "ok" and r["lead"] == 0]
    res["matched_pairs_with_valid_lead0_window"] = len(ok0)
    if ok0:
        res["pair_lead0_median"] = {m: float(np.median([r[m] for r in ok0])) for m in METRICS}
    return res, pairs


# ----------------------------------------------------------------------------- figures
def plot_pair(S, U, tag, norm, path, note=""):
    d = U["first_damage_step"]
    tU = anchor_time(U, 0)
    ph, pr = phase_progress(U, tU)
    tS = match_time(S, ph, pr)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    for ep, col, lab, ta in ((S, SAFE_C, "no damage", tS), (U, DMG_C, "damaging", tU)):
        end = min(ep["T"], (ep["first_damage_step"] if ep["any_damage"] else ta) + 25)
        P = ep["eef"][:end]
        for k, (i, j, nm) in enumerate(((0, 1, "x-y (top view)"), (0, 2, "x-z (side view)"))):
            a = ax[0, k]
            a.plot(P[:, i], P[:, j], color=col, lw=1.6, label=f"{lab}: {ep['episode_id']}")
            if ta is not None:
                W = ep["eef"][ta + 1 : ta + 1 + H]
                a.plot(W[:, i], W[:, j], color=col, lw=4, alpha=0.45)
                m = W.mean(0)
                a.scatter([m[i]], [m[j]], marker="X", s=90, color=col, edgecolor="white", zorder=5)
            a.scatter([ep["obj0_world"][i]], [ep["obj0_world"][j]], marker="o", s=60, facecolor="none", edgecolor=col)
            if ep["any_damage"]:
                q = ep["eef"][ep["first_damage_step"]]
                a.scatter([q[i]], [q[j]], marker="*", s=160, color=col, edgecolor=INK, zorder=6)
            a.set_title(nm, fontsize=10)
            a.set_xlabel(["x", "y", "z"][i] + " (m)")
            a.set_ylabel(["x", "y", "z"][j] + " (m)")
        rel = np.arange(ep["T"]) - (ta if ta is not None else 0)
        sl = slice(max(0, (ta or 0) - 40), min(ep["T"], (ta or 0) + 40))
        ax[0, 2].plot(rel[sl], ep["speed"][sl] * 1000, color=col, lw=1.6)
        ax[1, 0].plot(rel[sl], ep["grip_cmd"][sl], color=col, lw=1.6)
        ax[1, 0].plot(rel[sl], (ep["grip_width"][sl] - 0.08) / 0.03, color=col, lw=1.0, ls="--")
        ax[1, 1].plot(rel[sl], ep["obj_force"][sl], color=col, lw=1.4)
        hh = ep["health"].min(1)
        ax[1, 2].plot(rel[sl], hh[sl], color=col, lw=1.6)
        if ep["any_damage"]:
            for a in (ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]):
                a.axvline(ep["first_damage_step"] - ta, color=col, ls=":", lw=1.2)
    for a in (ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]):
        a.axvspan(1, H, color="#dddddd", alpha=0.6, lw=0)
        a.set_xlabel("steps from matched decision time t (grey = intent window t+1..t+8)", fontsize=8.5)
    ax[0, 2].set_ylabel("eef speed (mm/step)")
    ax[1, 0].set_ylabel("gripper cmd (solid, -1=open); width (dashed, scaled)")
    ax[1, 1].set_ylabel("stored contact force on target object (N)")
    ax[1, 2].set_ylabel("min entity health")
    ax[0, 0].legend(fontsize=8, loc="best")
    dd = dist(feats(S, tS, norm), feats(U, tU, norm))
    fig.suptitle(
        f"{tag}  |  phase={ph}  first damage U=step {d} ({', '.join(e.split('@')[1] for e in U['damaged_entities'])}, "
        f"max loss {U['max_health_loss']:.1f})\n"
        f"intent_norm={dd['intent_norm']:.3f}  mean-pos diff={dd['intent_pos_m']*1000:.1f} mm  rot={dd['intent_rot_deg']:.1f} deg  |  "
        f"path-shape={dd['path_shape_mm']:.1f} mm  speed={dd['speed_mm']:.2f} mm/step  grip={dd['grip_cmd']:.2f}  "
        f"force={dd['obj_force_N']:.1f} N  {note}\nX = window mean (= intent label), * = first-damage pose, o = initial object",
        fontsize=9.5,
    )
    for a in ax.ravel():
        a.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_auc(all_res, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for a, task in zip(axes, all_res):
        A = all_res[task]["auc"]
        for k, lead in enumerate(FR.STAGE["lead_windows"]):
            key = f"geometric/lead{lead}"
            if key not in A or "intent_norm" not in A[key]:
                continue
            y = np.arange(len(METRICS)) + (k - 1) * 0.25
            x = [A[key][m]["auc"] for m in METRICS]
            ci = A[key]["ci95_two_way"]
            lo = [A[key][m]["auc"] - ci[m][0] for m in METRICS]
            hi = [ci[m][1] - A[key][m]["auc"] for m in METRICS]
            a.errorbar(x, y, xerr=[lo, hi], fmt="o", ms=5, lw=1.2, color=["#2a78d6", "#1baf7a", "#4a3aa7"][k],
                       label=f"lead {lead} (n anchors={A[key]['n_anchors']})")
        a.axvline(0.5, color=MUTED, lw=1)
        a.set_yticks(np.arange(len(METRICS)))
        a.set_yticklabels(METRICS)
        a.set_xlim(0, 1)
        a.set_title(f"{task}: anchor-discrimination AUC (geometric comparators)", fontsize=10)
        a.set_xlabel("P(dist to safe comparator > dist to damaging comparator); 0.5 = no separation")
        a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle("95% CI: two-way bootstrap over damaging anchors and safe comparator episodes", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- probe
def probe(task, eps, norm, group_of):
    X = {k: [] for k in list(FR.PROBE["features"]) + ["obs+intent+speed", "obs+intent+grip", "obs+intent+pathres"]}
    y, g, y_lead, eid = [], [], [], []
    for e in eps:
        O = normalize(repo_obs(e), norm)
        last = e["first_damage_step"] if e["any_damage"] else e["T"]
        for t in range(1, min(last, e["T"] - H)):
            base = np.r_[O[t - 1], O[t]]
            it = repo_intent_at(e, t, norm)
            P = e["eef"][t + 1 : t + 1 + H]
            prof = np.r_[e["speed"][t + 1 : t + 1 + H] * 100, e["grip_cmd"][t : t + H], ((P - P.mean(0)) * 100).ravel()]
            X["obs"].append(base)
            X["obs+intent"].append(np.r_[base, it])
            X["obs+intent+profile"].append(np.r_[base, it, prof])
            X["obs+intent+speed"].append(np.r_[base, it, prof[:H]])
            X["obs+intent+grip"].append(np.r_[base, it, prof[H:2 * H]])
            X["obs+intent+pathres"].append(np.r_[base, it, prof[2 * H:]])
            eid.append(e["episode_id"])
            y.append(int(e["any_damage"] and t + 1 <= e["first_damage_step"] <= t + H))
            # lead label: damage in the horizon AFTER the intent window (window itself fully pre-damage)
            y_lead.append(int(e["any_damage"] and t + H + 1 <= e["first_damage_step"] <= t + 2 * H)
                          if (not e["any_damage"] or t + H < e["first_damage_step"]) else -1)
            g.append(group_of[e["episode_id"]])
    y_all, g_all, y_lead, eid = np.array(y), np.array(g), np.array(y_lead), np.array(eid)
    out = {}
    for label, yv in (("damage_in_window", y_all), ("damage_in_next_window_after_clean_window", y_lead)):
        keep = yv >= 0
        out[label] = _fit_probe({k: np.array(v, dtype=np.float64)[keep] for k, v in X.items()}, yv[keep], g_all[keep], eid[keep])
    return out


def _auroc(scores, y):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def _fit_probe(X, y, g, eid):
    out = {"n_windows": int(len(y)), "n_pos": int(y.sum()), "n_groups": int(len(set(g))),
           "n_episodes": int(len(set(eid))), "n_pos_episodes": int(len(set(eid[y == 1])))}
    all_scores = {}
    for name, Xa in X.items():
        scores = np.zeros(len(y))
        for grp in sorted(set(g)):
            te = g == grp
            tr = ~te
            if y[tr].sum() == 0:
                scores[te] = 0.5
                continue
            mu, sd = Xa[tr].mean(0), Xa[tr].std(0) + 1e-6
            Xt = torch.tensor((Xa[tr] - mu) / sd)
            yt = torch.tensor(y[tr], dtype=torch.float64)
            wpos = 0.5 / yt.mean()
            wneg = 0.5 / (1 - yt.mean())
            sw = torch.where(yt > 0, wpos, wneg)
            w = torch.zeros(Xt.shape[1], dtype=torch.float64, requires_grad=True)
            b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
            opt = torch.optim.LBFGS([w, b], max_iter=300, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                z = Xt @ w + b
                loss = (sw * torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="none")).mean() + 1e-2 * (w ** 2).sum()
                loss.backward()
                return loss

            opt.step(closure)
            with torch.no_grad():
                scores[te] = torch.sigmoid(torch.tensor((Xa[te] - mu) / sd) @ w + b).numpy()
        pred = scores >= 0.5
        tpr = (pred & (y == 1)).sum() / max(1, (y == 1).sum())
        tnr = (~pred & (y == 0)).sum() / max(1, (y == 0).sum())
        aucv = _auroc(scores, y)
        all_scores[name] = scores
        out[name] = {"auroc": float(aucv), "balanced_acc": float((tpr + tnr) / 2)}
    # episode-bootstrap CI of AUROC differences vs obs and vs obs+intent (scores fixed)
    ue = np.array(sorted(set(eid)))
    idx_of = {e: np.where(eid == e)[0] for e in ue}
    r = np.random.default_rng(0)
    diffs = defaultdict(list)
    for _ in range(500):
        samp = np.concatenate([idx_of[e] for e in r.choice(ue, len(ue))])
        yb = y[samp]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        a = {k: _auroc(v[samp], yb) for k, v in all_scores.items()}
        diffs["intent_minus_obs"].append(a["obs+intent"] - a["obs"])
        for k in ("obs+intent+profile", "obs+intent+speed", "obs+intent+grip", "obs+intent+pathres"):
            diffs[f"{k.split('+')[-1]}_minus_intent"].append(a[k] - a["obs+intent"])
    out["auroc_diff_ci95_episode_bootstrap"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in diffs.items()}
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rules_sha = hashlib.sha256(open(FR.__file__, "rb").read()).hexdigest()
    frozen = {k: getattr(FR, k) for k in ("VERSION", "ERRATA", "H", "SIMILARITY", "STAGE", "METRICS", "NULL", "COUNTEREXAMPLE", "FIGURES", "PROBE")}
    frozen["sha256_of_frozen_rules_py"] = rules_sha
    with open(OUT / "frozen_rules.json", "w") as fh:
        json.dump(frozen, fh, indent=1)

    splits = json.load(open(OUT / "splits.json"))
    all_res, summary = {}, {}
    for task in ROBOCASA_TASKS:
        eps = load_task(task)
        group_of = {eid: grp for grp, ids in splits[task]["groups"].items() for eid in ids}
        # reset family (wipe): identical dirt layout within condition
        if task == "wipe_countertop":
            fam = {}
            for e in eps:
                key = (e["split"], tuple(np.round(e["dirt"].ravel(), 6)))
                e["reset_family"] = fam.setdefault(key, f"fam{len(fam)}")
        norm = repo_intent_normalizer(eps)
        res, pairs = analyse_task(task, eps, norm)
        res["intent_normalizer"] = {"slice": norm["slice"], "obs_keys": OBS_KEYS,
                                    "eef_min": norm["min"][norm["slice"][0]:norm["slice"][1]].tolist(),
                                    "eef_range": norm["range"][norm["slice"][0]:norm["slice"][1]].tolist()}
        # intent scale reference: per-step change of the intent label within episodes
        steps = []
        for e in eps:
            for t in range(1, e["T"] - H - 1):
                steps.append(np.linalg.norm(repo_intent_at(e, t + 1, norm) - repo_intent_at(e, t, norm)))
        res["intent_step_change_median"] = float(np.median(steps))
        # figures: predefined set
        byid = {e["episode_id"]: e for e in eps}
        figs = []
        for k, (S, U, c) in enumerate(pairs[:3]):
            if anchor_time(U, 0) is None or match_time(S, *phase_progress(U, anchor_time(U, 0))) is None:
                continue
            p = FIG / f"{task}_pair{k}.png"
            plot_pair(S, U, f"{task} matched pair #{k} (geo cost {c:.2f})", norm, p)
            figs.append(p.name)
        for k, ce in enumerate(res["counterexamples"][:2]):
            p = FIG / f"{task}_counterexample{k}.png"
            plot_pair(byid[ce["safe"]], byid[ce["damaging"]], f"{task} counterexample #{k}", norm, p,
                      note=f"[large: {','.join(ce['large'])}]")
            figs.append(p.name)
        res["figures"] = figs
        res["probe"] = probe(task, eps, norm, group_of)
        all_res[task] = res
        summary[task] = {k: v for k, v in res.items() if k != "pair_rows"}
    plot_auc(all_res, FIG / "auc_summary.png")

    with open(OUT / "pair_intent_results.json", "w") as fh:
        json.dump(all_res, fh, indent=1, default=float)
    import csv

    with open(OUT / "comparable_pairs.csv", "w", newline="") as fh:
        cols = ["task", "safe", "damaging", "split", "geo_cost", "lead", "status", "phase", "progress", "t_U", "d_U", "t_S"] + METRICS
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for task, r in all_res.items():
            for row in r["pair_rows"]:
                w.writerow({"task": task, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}})
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
