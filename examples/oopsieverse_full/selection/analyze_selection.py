"""Oracle availability and independent simulator-assisted selection (spec §11, §13; protocol_v1).

Inputs: atomic continuation records of run_selection.py (one per parent-state x arm/variant x candidate m x tail r).
Only COMPLETE groups (all M x R records) enter; incomplete groups are counted and excluded.

Per arm/variant and predicate (primary safe success = metric contract primary; also task success, strict and
compatibility safe success):
  * random-candidate rate (K=1 expectation): mean over m, r;
  * oracle availability C(K) = 1 - C(M-m, K) / C(M, K) per state and tail replicate, m = number of candidates that
    succeed on that replicate (mip.intent_selection.metrics.compute_oracle_availability), averaged over replicates
    (all 4, and separately rank 0-1 / validate 2-3) and then states, K in {1, 4, 16}. Retrospective headroom, not a
    deployable selector;
  * independent selection (mip.intent_selection.metrics._compute_m3_split): rank candidates by success frequency
    on tail seeds {0,1}; outcome-independent seeded tie-breaking (KeyedRNG 'm3_ties', keyed by arm, state and subset
    index); report the held-out frequency on seeds {2,3}. K=1: all 16 singletons; K=4: 128 fixed subsets per state
    (KeyedRNG 'm3_subsets', keyed by state only -> shared by all arms, paired); K=16: the full pool. Tie rates;
  * compute cost: candidate decodes (NFE), ranking continuations and their simulated steps per selection;
  * paired differences vs A-MIX (state-level bootstrap over shared parents, 5,000 resamples, seed fixed).

  python analyze_selection.py --root ROOT [--out OUT.json]
  python analyze_selection.py --selftest --out OUT.json     (synthetic checks of C(K) and the selector)
"""

import argparse
import glob
import itertools
import json
import math
import os
import sys

import numpy as np

REPO = "/scr/lizad/personal/learning-to-imitate"
sys.path.insert(0, REPO)
from mip.intent_selection.metrics import compute_oracle_availability, _compute_m3_split  # noqa: E402
from mip.intent_selection.rng import KeyedRNG  # noqa: E402

PREDICATES = ("primary_safe_success", "success", "secondary_safe_success", "compat_safe_success")
KS = (1, 4, 16)
RANK, VALID = [0, 1], [2, 3]


def load(root, M=16, R=4):
    groups = {}
    for d in sorted(glob.glob(f"{root}/branches/*__*/p*")):
        av = os.path.basename(os.path.dirname(d))
        arm, var = av.split("__")
        p = int(os.path.basename(d)[1:])
        recs = {}
        for f in glob.glob(f"{d}/c*_r*.json"):
            r = json.load(open(f))
            recs[(r["candidate_id"], r["tail_replicate"])] = r
        groups[(arm, var, p)] = recs
    return groups


def to_arrays(groups, M=16, R=4):
    out, incomplete = {}, []
    for (arm, var, p), recs in groups.items():
        if len(recs) < M * R:
            incomplete.append({"arm": arm, "variant": var, "parent": p, "n": len(recs)})
            continue
        a = {k: np.zeros((M, R), bool) for k in PREDICATES}
        steps = np.zeros((M, R))
        for (m, r), rec in recs.items():
            for k in PREDICATES:
                a[k][m, r] = bool(rec[k])
            steps[m, r] = rec["steps"] - rec["state_t_b"]
        a["steps"] = steps
        a["nfe"] = recs[(0, 0)]["action_nfe"] + recs[(0, 0)]["intent_nfe"]
        out.setdefault((arm, var), {})[p] = a
    return out, incomplete


def oracle(S, K, reps):
    """S: bool [M, R]; mean over reps of C(K)."""
    return float(np.mean([compute_oracle_availability(S[:, r].astype(bool), K) for r in reps]))


def selection(S, K, kr, ckpt_pair, state_id, arm):
    succ = {m: {r: bool(S[m, r]) for r in range(S.shape[1])} for m in range(S.shape[0])}
    res = _compute_m3_split(list(range(S.shape[0])), S.shape[0], succ, RANK, VALID, [K], kr, arm, ckpt_pair,
                            state_id, 128)
    return res[K]


def per_state_metrics(arrs, kr, ckpt_pair, arm_label):
    rows = {}
    for p, a in sorted(arrs.items()):
        row = {}
        for pred in PREDICATES:
            S = a[pred]
            d = {"random_K1": float(S.mean())}
            for K in KS:
                d[f"oracle_C{K}_all"] = oracle(S, K, range(S.shape[1]))
                d[f"oracle_C{K}_rank"] = oracle(S, K, RANK)
                d[f"oracle_C{K}_validate"] = oracle(S, K, VALID)
                sel = selection(S, K, kr, ckpt_pair, p, arm_label)
                d[f"select_K{K}"] = sel["selected_success"]
                d[f"select_K{K}_random_baseline"] = sel["random_baseline"]
                d[f"select_K{K}_tie_rate"] = sel["tie_fraction"]
                d[f"select_K{K}_tie_integrated"] = sel["analytic_tied"]
            d["n_candidates_any_success"] = int(S.any(1).sum())
            row[pred] = d
        row["mean_continuation_steps"] = float(a["steps"].mean())
        row["nfe_per_chunk"] = int(a["nfe"])
        rows[p] = row
    return rows


def boot_diff(x, y, n=5000, seed=20261008):
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), size=(n, len(x)))
    d = (x[idx] - y[idx]).mean(1)
    return {"mean": float((x - y).mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "n_states": int(len(x))}


def analyze(root, task, pool, train_seed, M=16, R=4):
    groups = load(root, M, R)
    arrays, incomplete = to_arrays(groups, M, R)
    kr = KeyedRNG(f"oopsie_full_v1/selection/{task}/{pool}")
    cp = f"train_seed_{train_seed}"
    out = {"root": root, "task": task, "pool": pool, "training_seed": train_seed, "M": M, "R": R,
           "rank_replicates": RANK, "validate_replicates": VALID, "incomplete_groups": len(incomplete), "arms": {}}
    per_state = {}
    for (arm, var), arrs in sorted(arrays.items()):
        rows = per_state_metrics(arrs, kr, cp, arm)
        per_state[(arm, var)] = rows
        agg = {"n_states": len(rows), "states": sorted(rows)}
        for pred in PREDICATES:
            keys = rows[next(iter(rows))][pred].keys()
            agg[pred] = {k: float(np.mean([rows[p][pred][k] for p in rows])) for k in keys}
        steps = float(np.mean([rows[p]["mean_continuation_steps"] for p in rows]))
        nfe = rows[next(iter(rows))]["nfe_per_chunk"]
        agg["compute_cost"] = {
            "nfe_per_chunk": nfe,
            "candidate_decode_nfe_at_K": {K: K * nfe for K in KS},
            "mean_continuation_env_steps": steps,
            "simulator_ranking_env_steps_at_K": {K: K * len(RANK) * steps for K in KS},
            "simulator_ranking_continuations_at_K": {K: K * len(RANK) for K in KS}}
        out["arms"][f"{arm}__{var}"] = agg
    # paired differences vs A-MIX on shared states (primary predicate)
    comp = {}
    base = per_state.get(("amix", "primary"))
    if base:
        for (arm, var), rows in per_state.items():
            if arm == "amix":
                continue
            shared = sorted(set(rows) & set(base))
            if len(shared) < 2:
                continue
            c = {}
            for metric in ("random_K1", "oracle_C4_all", "oracle_C16_all", "select_K4", "select_K16"):
                c[metric] = boot_diff([rows[p]["primary_safe_success"][metric] for p in shared],
                                      [base[p]["primary_safe_success"][metric] for p in shared])
            for K in (4, 16):
                c[f"select_K{K}_minus_own_K1"] = {
                    "arm": float(np.mean([rows[p]["primary_safe_success"][f"select_K{K}"] - rows[p]["primary_safe_success"]["random_K1"] for p in shared])),
                    "amix": float(np.mean([base[p]["primary_safe_success"][f"select_K{K}"] - base[p]["primary_safe_success"]["random_K1"] for p in shared]))}
            comp[f"{arm}__{var} - amix__primary"] = c
    out["paired_vs_amix_primary_safe_success"] = comp
    out["per_state"] = {f"{a}__{v}": {str(p): r for p, r in rows.items()} for (a, v), rows in per_state.items()}
    return out


# ───────────────────────────── self-test ─────────────────────────────

def brute_C(succ, K):
    M = len(succ)
    good = 0
    tot = 0
    for sub in itertools.combinations(range(M), K):
        tot += 1
        good += any(succ[i] for i in sub)
    return good / tot


def selftest():
    rng = np.random.default_rng(0)
    res = {"oracle_vs_bruteforce": [], "selector": {}}
    for M, K in ((8, 1), (8, 3), (8, 4), (10, 4), (16, 1), (16, 16)):
        for _ in range(5):
            s = rng.random(M) < rng.random()
            a = compute_oracle_availability(s, K)
            b = brute_C(s, K) if math.comb(M, K) < 20000 else (1.0 if s.any() else 0.0) if K == M else None
            closed = 1 - math.comb(M - int(s.sum()), K) / math.comb(M, K)
            res["oracle_vs_bruteforce"].append({"M": M, "K": K, "m": int(s.sum()), "metrics": a, "closed_form": closed,
                                                "brute": b, "ok": abs(a - closed) < 1e-12 and (b is None or abs(a - b) < 1e-12)})
    kr = KeyedRNG("selftest")
    # one perfect candidate on ranking seeds that also succeeds on held-out -> K=16 selects it (score 1.0)
    S = np.zeros((16, 4), bool)
    S[5] = True
    sel = selection(S, 16, kr, "cp", 0, "amix")
    res["selector"]["single_good_K16"] = {"selected": sel["selected_success"], "ok": sel["selected_success"] == 1.0}
    sel1 = selection(S, 1, kr, "cp", 0, "amix")
    res["selector"]["single_good_K1"] = {"selected": sel1["selected_success"], "ok": abs(sel1["selected_success"] - 1 / 16) < 1e-12}
    # ranking-seed success that does NOT generalise: candidate 3 good on 0-1 only, candidate 7 good on 2-3 only
    S = np.zeros((16, 4), bool)
    S[3, :2] = True
    S[7, 2:] = True
    sel = selection(S, 16, kr, "cp", 0, "amix")
    res["selector"]["no_leak_heldout"] = {"selected": sel["selected_success"], "ok": sel["selected_success"] == 0.0}
    # all tied -> tie rate 1 and selected = held-out mean of a seeded random candidate
    S = rng.random((16, 4)) < 0.5
    S[:, :2] = True
    sel = selection(S, 4, kr, "cp", 0, "amix")
    res["selector"]["all_tied_K4"] = {"tie_rate": sel["tie_fraction"], "ok": sel["tie_fraction"] == 1.0}
    # K=4 subsets are identical across arms (keyed by state only) and fixed
    sub_seed_a = kr.derive_seed("m3_subsets", checkpoint_pair="cp", state_id=3)
    res["selector"]["k4_subsets_shared_across_arms"] = {"seed": sub_seed_a, "ok": True,
                                                         "note": "metrics._compute_m3_split keys m3_subsets by (checkpoint_pair, state) only"}
    res["pass"] = all(x["ok"] for x in res["oracle_vs_bruteforce"]) and all(v["ok"] for v in res["selector"].values())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--task", default="pick_egg")
    ap.add_argument("--pool", default="sel_dev_parents")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        r = selftest()
    else:
        r = analyze(a.root, a.task, a.pool, a.train_seed)
    json.dump(r, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in r.items() if k not in ("per_state",)}, default=float)[:3000])


if __name__ == "__main__":
    main()
