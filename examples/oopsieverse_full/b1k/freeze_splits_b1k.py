"""Frozen grouped splits for the B1K diagnostic arm (b1k_diag_protocol.md §3): splits_b1k_v1.json.

Same rules as examples/oopsieverse_full/freeze_splits.py (roster v1): groups from
catalog/build_grouped_splits.build_groups (5-demo pseudo-sessions within feedback condition, identical
actions, identical informative states[0]); exact exhaustive group->split search with the v1 cost
(70/15/15 size term x10 + per-split stratum-composition deviation + 0.05 per stratum missing from a held-out
split); strata = feedback x primary safe-success label; ties -> lexicographically smallest assignment vector
(vectorised enumeration in lexicographic order). Only official-success episodes enter the groups' training pool
use; non-successes are listed as excluded (spec 7).
Subsets: A_MIX_train (= train successes), A_SAFE_train (train primary-safe successes).
"""
import csv
import hashlib
import importlib.util
import itertools
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "outputs/oopsieverse_full_study")
B1K = "/edrive1/lizad/oopsieverse_full_study/reports/b1k"
FRACS = np.array((0.70, 0.15, 0.15))
SPLITS = ("train", "dev", "test")
LABELS = {"add_firewood": os.path.join(B1K, "replay_labels_add_firewood.csv"),
          "shelve_item": os.path.join(B1K, "replay_labels_shelve_item.csv")}


def sha(p):
    h = hashlib.sha256()
    h.update(open(p, "rb").read())
    return h.hexdigest()


spec = importlib.util.spec_from_file_location("bgs", os.path.join(HERE, "..", "catalog/build_grouped_splits.py"))
BGS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BGS)


def exhaustive(groups, label_of):
    strata = sorted({label_of[i] for g in groups for i in g})
    G = np.array([[sum(label_of[i] == s for i in g) for s in strata] for g in groups], float)  # [g, s]
    n, tot = G.sum(), G.sum(0)
    ng = len(groups)
    best, best_cost = None, np.inf
    nb = min(ng, 12)
    tail = np.array(list(itertools.product(range(3), repeat=nb)), np.int8)  # lexicographic
    for head in itertools.product(range(3), repeat=ng - nb):
        A = np.concatenate([np.broadcast_to(np.array(head, np.int8), (len(tail), ng - nb)), tail], 1)
        comp = np.stack([(A == k).astype(float) @ G for k in range(3)], 1)  # [m, 3, s]
        sizes = comp.sum(2)
        ok = (sizes > 0).all(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cost = ((sizes / n - FRACS) ** 2).sum(1) * 10
            cost += (((comp / sizes[:, :, None]) - tot / n) ** 2).sum((1, 2))
        miss = ((comp == 0) & (tot > 0)).sum(2)
        cost += 0.05 * (miss[:, 1] + miss[:, 2])
        cost[~ok] = np.inf
        j = int(np.argmin(cost))
        if cost[j] < best_cost - 1e-12:
            best, best_cost = A[j].copy(), float(cost[j])
    return best, best_cost, strata


def main():
    import sys
    only = sys.argv[1]
    inv = [r for r in csv.DictReader(open(os.path.join(OUT, "catalog/episode_inventory_b1k.csv")))
           if r["source"] == "paper_experiments" and r["feedback"] in ("fb", "nofb")]
    out = {"status": "FROZEN b1k v1 (DIAGNOSTIC)", "version": f"splits_b1k_v1_{only}", "created": "2026-10-23",
           "protocol": "outputs/oopsieverse_full_study/b1k/b1k_diag_protocol.md",
           "rules": "as roster splits_v1 (freeze_splits.py); groups = catalog build_grouped_splits.build_groups(block=5, "
                    "max_family=10); exact exhaustive 3^G search, v1 cost, strata feedback x primary; only official "
                    "successes are split members; shared mixed-train normaliser for all arms",
           "inputs": {"episode_inventory_b1k_sha256": sha(os.path.join(OUT, "catalog/episode_inventory_b1k.csv")),
                      "grouping_code_sha256": sha(os.path.join(HERE, "..", "catalog/build_grouped_splits.py")),
                      "this_code_sha256": sha(os.path.abspath(__file__))},
           "tasks": {}}
    for task in (only,):
        out["inputs"][f"labels_{task}_sha256"] = sha(LABELS[task])
        lab = {f"{task}/{r['cond']}/{r['demo']}": r for r in csv.DictReader(open(LABELS[task]))}
        eps = [dict(e, episode_id=f"{task}/{e['feedback']}/{e['demo']}") for e in inv if e["task_dir"] == task]
        assert all(e["episode_id"] in lab for e in eps), "labels missing"
        excluded = sorted(i for i, r in lab.items() if r["official_success"] != "True")
        groups, reasons, near, fam = BGS.build_groups(eps, task, 5, 10)
        label_of = {i: f"{i.split('/')[1]}:{'safe' if lab[i]['primary_safe_success'] == 'True' else 'damaging'}"
                    for i in lab}
        # non-successes are excluded from the pool; groups keep their remaining members
        groups_s = [[i for i in g if i not in excluded] for g in groups]
        groups_s = [g for g in groups_s if g]
        asg, cost, strata = exhaustive(groups_s, label_of)
        key = lambda x: (x.split("/")[1], int(x.rsplit("_", 1)[1]))  # noqa: E731
        split_of = {i: SPLITS[asg[k]] for k, g in enumerate(groups_s) for i in g}
        members = {s: sorted([i for i in split_of if split_of[i] == s], key=key) for s in SPLITS}
        cnt = {s: dict(Counter(label_of[i] for i in members[s])) for s in SPLITS}
        safe = [i for i in members["train"] if lab[i]["primary_safe_success"] == "True"]
        exc = []
        for s in ("dev", "test"):
            miss = [st for st in strata if not any(label_of[i] == st for i in members[s])]
            if miss:
                exc.append(f"{s} has no episodes in strata {miss}")
        if len(safe) < 10:
            exc.append(f"A_SAFE_train has {len(safe)} episodes (< 10 target): exploratory")
        fbsafe = Counter(i.split("/")[1] for i in safe)
        if len(fbsafe) < 2:
            exc.append(f"A_SAFE_train comes from one feedback condition only ({dict(fbsafe)}): safety label is "
                       "confounded with collection condition")
        out["tasks"][task] = {
            "n_episodes": len(eps), "excluded_non_success": excluded,
            "search": {"n_groups": len(groups_s), "assignments": 3 ** len(groups_s), "cost": cost, "strata": strata,
                       "group_rules": dict(reasons), "near_duplicates": near},
            "groups": [{"group_id": f"{task}/g{k:02d}", "split": SPLITS[asg[k]], "members": sorted(g, key=key)}
                       for k, g in enumerate(groups_s)],
            "splits": {s: {"members": m, "n": len(m)} for s, m in members.items()},
            "counts_split_x_stratum": cnt,
            "subsets": {"A_MIX_train": {"members": members["train"], "n": len(members["train"])},
                        "A_SAFE_train": {"members": safe, "n": len(safe), "by_feedback": dict(fbsafe)}},
            "exceptions": exc,
        }
        print(task, "groups", len(groups_s), "cost", round(cost, 4), {s: len(m) for s, m in members.items()}, cnt,
              "A_SAFE", len(safe), dict(fbsafe), exc)
    body = json.dumps(out, indent=1)
    out["content_sha256_excluding_this_field"] = hashlib.sha256(body.encode()).hexdigest()
    p = os.path.join(B1K, f"splits_b1k_v1_{only}.json")
    assert not os.path.exists(p), "frozen file exists; refusing to overwrite"
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p, sha(p))


if __name__ == "__main__":
    main()
