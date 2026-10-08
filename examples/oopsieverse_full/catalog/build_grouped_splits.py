"""DRAFT grouped 70/15/15 splits for OopsieVerse full-study candidate tasks (spec section 7).

Reads ``episode_inventory.csv`` / ``episode_inventory_b1k.csv`` written by
``inventory_tasks_and_data.py``. No simulator. Labels used for stratification are the
PRELIMINARY stored-health classes (RoboCasa) and are pending replay verification; B1K
episodes have no stored health, so they are stratified by feedback condition only.

Grouping (union-find; any shared key merges groups):
  * reset family: identical init_robot_base_pos (+ identical dirt_marker_positions for wipe)
    when the family is smaller than --max-family; larger families (a whole feedback
    condition sharing one base pose) carry no per-episode reset information, so they fall
    back to collection pseudo-sessions instead.
    EXCEPTION wipe_countertop: spec section 7 requires identical layouts/base poses to stay
    grouped, so wipe families are never broken up regardless of size.
  * pseudo-session: blocks of --block consecutive demo indices within a feedback condition
    (no session/timestamp metadata exists in the files).
  * duplicate trajectory: identical action sequence, or identical states[0] (RoboCasa only;
    B1K only when states[0] varies per episode; add_firewood's states[0] is the shared init-state
    snapshot taken before reset noise, so identical states[0] there is not a duplicate).
  * near-duplicate initial state (RoboCasa): object and eef start within 5 mm, compared only
    among episodes with the same dirt layout (wipe starts are all within ~mm of each other).
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FRACS = (0.70, 0.15, 0.15)
SPLITS = ("train", "dev", "test")


class UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def build_groups(eps, task, block, max_family, near_mm=5.0):
    ids = [e["episode_id"] for e in eps]
    uf = UF(ids)
    reasons = Counter()
    keyed = defaultdict(list)
    # B1K: states[0] is informative (per-episode reset) only when most episodes differ
    # (shelve_item 88/90 unique); add_firewood has 2 unique of 60 = shared init snapshot.
    s0_informative = len({e["state0_hash"] for e in eps}) > 0.5 * len(eps)
    for e in eps:
        keyed[("act", e["actions_hash"])].append(e["episode_id"])
        # B1K states[0] is the shared init-state pickle snapshot (taken before the per-episode
        # reset noise), so identical states[0] there is a reset family, not a duplicate.
        if e["backend"] == "robocasa" or s0_informative:
            keyed[("s0", e["state0_hash"])].append(e["episode_id"])
    fam = defaultdict(list)
    for e in eps:
        k = (e["feedback"], e.get("init_base_pos", ""), e.get("dirt_hash", ""))
        if e.get("init_base_pos"):
            fam[k].append(e["episode_id"])
    family_of = {}
    for k, members in fam.items():
        for m in members:
            family_of[m] = f"{k[0]}|pose:{hashlib.sha256(k[1].encode()).hexdigest()[:8]}|dirt:{k[2] or '-'}"
        if task == "wipe_countertop" or len(members) <= max_family:
            for m in members[1:]:
                uf.union(members[0], m)
            if len(members) > 1:
                reasons["reset_family"] += 1
    by_cond = defaultdict(list)
    for e in eps:
        by_cond[e["feedback"]].append(e)
    for cond, lst in by_cond.items():
        lst = sorted(lst, key=lambda e: int(e["demo"].split("_")[1]))
        for i in range(0, len(lst), block):
            blk = lst[i:i + block]
            for e in blk[1:]:
                uf.union(blk[0]["episode_id"], e["episode_id"])
            reasons["pseudo_session_block"] += 1
    for (kind, _), members in keyed.items():
        if len(members) > 1:
            for m in members[1:]:
                uf.union(members[0], m)
            reasons[f"duplicate_{kind}"] += 1
    near_pairs = []
    if eps and eps[0].get("init_obj_pos"):
        P = []
        for e in eps:
            o = json.loads(e["init_obj_pos"])
            P.append(np.array(sum((o[k] for k in sorted(o)), []) + json.loads(e["init_eef_pos"])))
        for i in range(len(eps)):
            for j in range(i + 1, len(eps)):
                # wipe: object/eef starts are within a few mm in ALL episodes (tight reset
                # distribution); the reset is defined by the dirt layout, so only compare
                # episodes sharing a dirt layout (empty for non-wipe tasks).
                if eps[i].get("dirt_hash", "") != eps[j].get("dirt_hash", ""):
                    continue
                if np.abs(P[i] - P[j]).max() * 1000 <= near_mm:
                    uf.union(ids[i], ids[j])
                    near_pairs.append((ids[i], ids[j]))
    groups = defaultdict(list)
    for i in ids:
        groups[uf.find(i)].append(i)
    gl = sorted(groups.values(), key=lambda g: sorted(g)[0])
    return gl, dict(reasons), near_pairs, family_of


def assign(groups, label_of, seed, iters):
    """Random search over group->split assignments minimising size + stratum deviation."""
    rng = np.random.default_rng(seed)
    n = sum(len(g) for g in groups)
    strata = sorted({label_of[i] for g in groups for i in g})
    gcount = np.array([[sum(label_of[i] == s for i in g) for s in strata] for g in groups], float)
    tot = gcount.sum(0)
    target = np.array(FRACS)
    best, best_cost = None, np.inf
    for _ in range(iters):
        a = rng.choice(3, size=len(groups), p=FRACS)
        sizes = np.array([gcount[a == k].sum() for k in range(3)])
        if (sizes == 0).any():
            continue
        cost = ((sizes / n - target) ** 2).sum() * 10
        for k in range(3):
            comp = gcount[a == k].sum(0)
            cost += (((comp / sizes[k]) - tot / n) ** 2).sum()
            # penalise empty strata in held-out splits when the stratum exists
            cost += 0.05 * ((comp == 0) & (tot > 0)).sum() * (k > 0)
        if cost < best_cost:
            best, best_cost = a.copy(), cost
    return best, best_cost, strata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, type=Path)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--max-family", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200000)
    a = ap.parse_args()

    rc = [r for r in load(a.catalog / "episode_inventory.csv")
          if r["source"] == "paper_experiments" and r["feedback"] in ("nofb", "fb")]
    b1k = [r for r in load(a.catalog / "episode_inventory_b1k.csv")
           if r["source"] == "paper_experiments" and r["feedback"] in ("nofb", "fb")]
    out, meta = {}, {}
    for task_dir in ("pick_egg", "wipe_countertop", "shelve_item", "add_firewood"):
        eps = [r for r in rc + b1k if r["task_dir"] == task_dir]
        for e in eps:
            e["episode_id"] = f"{task_dir}/{e['feedback']}/{e['demo']}"
        groups, reasons, near, family_of = build_groups(eps, task_dir, a.block, a.max_family)
        if eps[0]["backend"] == "robocasa":
            label_of = {e["episode_id"]: f"{e['feedback']}:{e['health_class']}" for e in eps}
            strat = "feedback x stored-health class (zero / mild<=5 / substantial>5), PRELIMINARY"
        else:
            label_of = {e["episode_id"]: e["feedback"] for e in eps}
            strat = "feedback only (no stored health/success in B1K files)"
        asg, cost, strata = assign(groups, label_of, a.seed, a.iters)
        split_of = {}
        for g, k in zip(groups, asg):
            for i in g:
                split_of[i] = SPLITS[k]
        ep_by_id = {e["episode_id"]: e for e in eps}
        counts = {}
        for s in SPLITS:
            mem = [i for i in split_of if split_of[i] == s]
            c = dict(n=len(mem), n_groups=sum(1 for g, k in zip(groups, asg) if SPLITS[k] == s),
                     by_stratum=dict(sorted(Counter(label_of[i] for i in mem).items())))
            if eps[0]["backend"] == "robocasa":
                c["paper_all_gt95_safe"] = sum(ep_by_id[i]["paper_all_gt95"] == "True" for i in mem)
                c["code_mean_ge95_safe"] = sum(ep_by_id[i]["code_mean_ge95"] == "True" for i in mem)
                c["strict_zero_loss"] = sum(ep_by_id[i]["health_class"] == "zero" for i in mem)
            counts[s] = c
        exceptions = []
        for s in ("dev", "test"):
            missing = [st for st in strata if counts[s]["by_stratum"].get(st, 0) == 0]
            if missing:
                exceptions.append(f"{s} has no episodes from strata {missing}")
        if task_dir == "wipe_countertop":
            exceptions.append("wipe nofb is a single reset family (one base pose + one dirt layout, 30 demos) "
                              "and must sit wholly in one split; nofb cannot be represented in all splits")
        out[task_dir] = dict(
            status="DRAFT - pending replay-verified success/safety labels",
            stratification=strat, seed=a.seed, search_iters=a.iters, cost=float(cost),
            group_rules=reasons, near_duplicate_pairs=near,
            groups=[dict(group_id=f"g{gi:02d}", split=SPLITS[k], members=sorted(g, key=lambda x: (x.split('/')[1], int(x.rsplit('_', 1)[1]))))
                    for gi, (g, k) in enumerate(zip(groups, asg))],
            counts=counts, exceptions=exceptions,
            reset_families=dict(Counter(family_of.values())) if family_of else {},
        )
        meta[task_dir] = dict(n_groups=len(groups), group_sizes=sorted(len(g) for g in groups))
    (a.catalog / "splits_draft.json").write_text(json.dumps(out, indent=1))
    for t, v in out.items():
        print(t, meta[t], v["group_rules"], "near:", len(v["near_duplicate_pairs"]))
        for s in SPLITS:
            print("   ", s, v["counts"][s])
        print("    exceptions:", v["exceptions"])


if __name__ == "__main__":
    main()
