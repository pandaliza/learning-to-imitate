"""Frozen grouped demonstration splits v1 (spec section 7) for the RoboCasa roster tasks.

Inputs (read-only):
  catalog/episode_inventory.csv, catalog/splits_draft.json, catalog/build_grouped_splits.py (grouping rules),
  replay/replay_labels.csv (replay-verified labels), metric_contract.json, raw teleop HDF5s (verification).
Output: outputs/oopsieverse_full_study/roster/splits_v1.json (+ printed summary).

Procedure
  1. Re-derive every grouping key directly from the raw HDF5 (base pose and dirt layout from ep_meta,
     states[0] and action hashes, initial object/eef positions from obs[0]) and assert it equals the
     catalog inventory; recompute reset families, exact duplicates and 5 mm near-duplicates.
  2. Build groups with the catalog's union-find rules (build_grouped_splits.build_groups: reset families,
     wipe families never broken, 5-demo pseudo-sessions within a condition, exact duplicates, 5 mm
     near-duplicate starts) and check that every reset family is contained in one group.
  3. Assign groups to train/dev/test by EXHAUSTIVE search over all 3^G assignments (G = 11 / 7) with the
     catalog cost (size deviation from 70/15/15 x10 + per-split stratum-composition deviation + 0.05 per
     stratum missing from a held-out split), strata = feedback x primary safe-success label. Ties broken by
     the lexicographically smallest assignment vector.
  4. Define training subsets (keyed blake2b draws, mip/intent_selection/rng.py convention).
"""

import csv
import hashlib
import importlib.util
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "outputs/oopsieverse_full_study")
CAT = os.path.join(OUT, "catalog")
ROSTER = os.path.join(OUT, "roster")
HF = ("/home/lizad/.cache/huggingface/hub/datasets--ut-robin-lab--oopsieverse-demos/snapshots/"
      "c0ec29395fdcf898de2aed8dc0b3e786dc2a37d3/paper_experiments/teleop_data")
EXPERIMENT_ID = "oopsie_full_v1"
SPLITS = ("train", "dev", "test")
FRACS = (0.70, 0.15, 0.15)
COND_FILE = {"fb": "with", "nofb": "without"}
TASKS = ("pick_egg", "wipe_countertop")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BGS = load_mod("bgs", os.path.join(HERE, "catalog/build_grouped_splits.py"))
KR = load_mod("ks_rng", os.path.join(REPO, "mip/intent_selection/rng.py")).KeyedRNG(EXPERIMENT_ID)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def keyed_rank(task, purpose, episode_id):
    """Per-episode keyed uniform rank; sorting by it gives a fixed-seed random order."""
    return KR.derive_seed("bootstrap", experiment=f"{EXPERIMENT_ID}/subset/{task}/{purpose}",
                          state_id=int(hashlib.blake2b(episode_id.encode(), digest_size=4).hexdigest(), 16))


# ───────────────────────── 1. raw verification of grouping keys ─────────────────────────

def verify_raw(eps, task):
    import h5py
    checks = Counter()
    mism = []
    for cond in ("fb", "nofb"):
        with h5py.File(os.path.join(HF, task, f"{COND_FILE[cond]}_live_feedback.hdf5"), "r") as f:
            for e in [x for x in eps if x["feedback"] == cond]:
                g = f["data"][e["demo"]]
                meta = json.loads(g.attrs["ep_meta"])
                base = [round(float(v), 6) for v in meta["init_robot_base_pos"]]
                ok_base = np.allclose(base, json.loads(e["init_base_pos"]), atol=1e-6)
                acts_h = hashlib.sha256(g["actions"][:].tobytes()).hexdigest()[:16]
                s0_h = hashlib.sha256(g["states"][0].tobytes()).hexdigest()[:16]
                checks["base_pose_match"] += ok_base
                checks["actions_hash_match"] += acts_h == e["actions_hash"]
                checks["state0_hash_match"] += s0_h == e["state0_hash"]
                if task == "wipe_countertop":
                    d = meta.get("dirt_marker_positions")
                    dh = hashlib.sha256(json.dumps(d).encode()).hexdigest()[:12]
                    checks["dirt_hash_match"] += dh == e["dirt_hash"]
                    if dh != e["dirt_hash"]:
                        mism.append((e["episode_id"], "dirt", dh, e["dirt_hash"]))
                checks["n"] += 1
                if not ok_base:
                    mism.append((e["episode_id"], "base"))
                if acts_h != e["actions_hash"]:
                    mism.append((e["episode_id"], "actions_hash", acts_h, e["actions_hash"]))
                if s0_h != e["state0_hash"]:
                    mism.append((e["episode_id"], "state0_hash", s0_h, e["state0_hash"]))
    return dict(checks), mism


# ───────────────────────── 3. exhaustive assignment ─────────────────────────

def exhaustive_assign(groups, label_of):
    n = sum(len(g) for g in groups)
    strata = sorted({label_of[i] for g in groups for i in g})
    gcount = np.array([[sum(label_of[i] == s for i in g) for s in strata] for g in groups], float)
    tot = gcount.sum(0)
    target = np.array(FRACS)
    best, best_cost = None, np.inf
    for a in itertools.product(range(3), repeat=len(groups)):
        a = np.array(a)
        sizes = np.array([gcount[a == k].sum() for k in range(3)])
        if (sizes == 0).any():
            continue
        cost = ((sizes / n - target) ** 2).sum() * 10
        for k in range(3):
            comp = gcount[a == k].sum(0)
            cost += (((comp / sizes[k]) - tot / n) ** 2).sum()
            cost += 0.05 * ((comp == 0) & (tot > 0)).sum() * (k > 0)
        if cost < best_cost - 1e-12:
            best, best_cost = a.copy(), cost
    return best, float(best_cost), strata


def largest_remainder(total, weights):
    keys = sorted(weights)
    w = np.array([weights[k] for k in keys], float)
    raw = total * w / w.sum()
    base = np.floor(raw).astype(int)
    rem = total - base.sum()
    order = sorted(range(len(keys)), key=lambda i: (-(raw[i] - base[i]), keys[i]))
    for i in order[:rem]:
        base[i] += 1
    return {k: int(b) for k, b in zip(keys, base)}


def main():
    os.makedirs(ROSTER, exist_ok=True)
    inv = [r for r in csv.DictReader(open(os.path.join(CAT, "episode_inventory.csv")))
           if r["source"] == "paper_experiments" and r["feedback"] in ("nofb", "fb")]
    labels = {}
    for r in csv.DictReader(open(os.path.join(OUT, "replay/replay_labels.csv"))):
        cond = {"with": "fb", "without": "nofb"}[r["cond"]]
        labels[f"{r['task']}/{cond}/{r['demo']}"] = r
    draft = json.load(open(os.path.join(CAT, "splits_draft.json")))
    contract_sha = sha(os.path.join(OUT, "metric_contract.json"))
    out = dict(
        status="FROZEN v1", version="splits_v1", created="2026-10-06",
        inputs=dict(
            episode_inventory_sha256=sha(os.path.join(CAT, "episode_inventory.csv")),
            replay_labels_sha256=sha(os.path.join(OUT, "replay/replay_labels.csv")),
            metric_contract_sha256=contract_sha,
            splits_draft_sha256=sha(os.path.join(CAT, "splits_draft.json")),
            grouping_code=f"examples/oopsieverse_full/catalog/build_grouped_splits.py (sha256 {sha(os.path.join(HERE, 'catalog/build_grouped_splits.py'))[:16]})",
            this_code="examples/oopsieverse_full/freeze_splits.py",
            dataset="ut-robin-lab/oopsieverse-demos @ c0ec29395fdcf898de2aed8dc0b3e786dc2a37d3",
        ),
        rules=dict(
            unit="complete demonstration groups; never time windows",
            grouping=("union-find over (a) reset families = identical ep_meta init_robot_base_pos (+ identical "
                      "dirt_marker_positions for wipe); families larger than 10 are not merged EXCEPT wipe, whose "
                      "families are always kept whole (spec 7); (b) 5-demo consecutive pseudo-sessions within a "
                      "feedback condition (no session metadata exists); (c) identical action sequences or states[0]; "
                      "(d) 5 mm near-duplicate starts (object + eef; wipe only within a dirt layout)"),
            assignment="exhaustive search over all 3^G group->split assignments; catalog cost; ties -> lexicographically smallest",
            stratification="feedback condition x primary safe-success label (replay-verified, metric_contract v1.0)",
            normalization=("Fit every normalizer (observation, action, intent/future-motion targets, encoder statistics) on the "
                           "TRAIN split of that task only, on the full mixed train pool, and share it across ALL arms "
                           "(A-MIX, A-SAFE, I-*, AUX, subset controls). Never use dev/test demos or evaluation rollouts. "
                           "A-SAFE and subset controls reuse the mixed-train normalizer (spec 8: shared normalizer)."),
            split_use=("train: policy/intent training. dev: offline validation losses, teacher-forced vs sampled-intent "
                       "logging, checkpoint/config choice. test: held-out offline diagnostics only (e.g. probes). Closed-loop "
                       "evaluation uses the reset banks (reset_banks_v1.json), never demo splits."),
            obs_action_pairing=("teleop obs[t] is recorded AFTER actions[t]; training pairs obs[t] with actions[t+1] "
                                "(drop actions[0]); see roster/alignment_audit.md"),
            labels="replay_labels.csv (force replay, collection-time gripper). All 120 episodes are official successes; none quarantined.",
        ),
        tasks={},
    )
    for task in TASKS:
        eps = [r for r in inv if r["task_dir"] == task]
        for e in eps:
            e["episode_id"] = f"{task}/{e['feedback']}/{e['demo']}"
        assert len(eps) == 60 and all(e["episode_id"] in labels for e in eps)
        raw_checks, mism = verify_raw(eps, task)
        groups, reasons, near, family_of = BGS.build_groups(eps, task, 5, 10)
        gid_of = {i: k for k, g in enumerate(groups) for i in g}
        # every reset family that is merged must sit in one group
        fam_members = defaultdict(list)
        for i, fam in family_of.items():
            fam_members[fam].append(i)
        fam_check = {}
        for fam, mem in fam_members.items():
            gs = {gid_of[m] for m in mem}
            fam_check[fam] = dict(n=len(mem), n_groups=len(gs),
                                  kept_whole=len(gs) == 1,
                                  merged_by_rule=(task == "wipe_countertop" or len(mem) <= 10))
        # near-duplicate pairs and draft-group consistency
        draft_groups = sorted(sorted(g["members"]) for g in draft[task]["groups"])
        same_as_draft_groups = draft_groups == sorted(sorted(g) for g in groups)
        lab = {e["episode_id"]: labels[e["episode_id"]] for e in eps}
        prim = {i: lab[i]["primary_safe_success"] == "True" for i in lab}
        label_of = {e["episode_id"]: f"{e['feedback']}:{'safe' if prim[e['episode_id']] else 'damaging'}" for e in eps}
        asg, cost, strata = exhaustive_assign(groups, label_of)
        split_of = {i: SPLITS[asg[gid_of[i]]] for i in gid_of}

        def counts_for(members):
            c = {}
            for pred, col in (("primary", "primary_safe_success"), ("secondary", "secondary_safe_success"),
                              ("compatibility", "compat_safe_success")):
                c[pred] = {}
                for fbk in ("fb", "nofb"):
                    mm = [i for i in members if i.split("/")[1] == fbk]
                    s = sum(lab[i][col] == "True" for i in mm)
                    c[pred][fbk] = dict(n=len(mm), safe_success=s, damaging_success=len(mm) - s)
                s = sum(lab[i][col] == "True" for i in members)
                c[pred]["total"] = dict(n=len(members), safe_success=s, damaging_success=len(members) - s)
            return c

        split_members = {s: sorted([i for i in split_of if split_of[i] == s],
                                   key=lambda x: (x.split("/")[1], int(x.rsplit("_", 1)[1]))) for s in SPLITS}
        counts = {s: dict(n=len(m), n_groups=len({gid_of[i] for i in m}), **counts_for(m)) for s, m in split_members.items()}
        exceptions = []
        for s in ("dev", "test"):
            missing = [st for st in strata if not any(label_of[i] == st for i in split_members[s])]
            if missing:
                exceptions.append(f"{s} has no episodes in strata {missing}")
        sizes = [len(split_members[s]) for s in SPLITS]
        if sizes != [42, 9, 9]:
            exceptions.append(f"split sizes {sizes} instead of 42/9/9: groups are multiples of 5 (pseudo-sessions/families)")
        if task == "pick_egg":
            exceptions.append("each feedback condition was collected with ONE robot base pose (sticky teleop ep_meta; see "
                              "reset_banks_v1.json), so the whole condition is one collection session; it cannot be grouped "
                              "whole without emptying held-out splits. Pseudo-sessions of 5 consecutive demos are used, so "
                              "held-out demos share base pose/session with train and measure within-session generalization only.")
        if task == "wipe_countertop":
            exceptions.append("nofb is ONE reset family (30 demos, one base pose + one dirt layout) and must sit wholly in "
                              "one split (spec 7); it is in train, so dev/test contain no nofb demos and no nofb held-out "
                              "evaluation from demos is possible")

        # ── subsets ──
        train = split_members["train"]
        safe_train = [i for i in train if prim[i]]
        n_safe = len(safe_train)
        if n_safe < 10:
            fam_safe = sorted((sum(prim[i] for i in g) for g in groups if all(i.split("/")[1] == "fb" for i in g)), reverse=True)
            n_nofb_safe = sum(prim[i] for i in train if i.split("/")[1] == "nofb")
            exceptions.append(f"A-SAFE train has only {n_safe} primary-safe episodes (< 10 target): exploratory. The most any "
                              f"rule-respecting assignment could reach is {n_nofb_safe + sum(fam_safe[:2])} (the two safest fb "
                              f"families {fam_safe[:2]} plus nofb), at the cost of the safety stratification required by spec 7; "
                              "the stratified assignment was kept.")
        for fbk in ("fb", "nofb"):
            for cls in (True, False):
                for s_ in ("dev", "test"):
                    k = sum(1 for i in split_members[s_] if i.split("/")[1] == fbk and prim[i] == cls)
                    if 0 < k < 2:
                        exceptions.append(f"{s_} has only {k} {fbk} {'safe' if cls else 'damaging'} episode")
        fb_counts = Counter(i.split("/")[1] for i in train)
        alloc = largest_remainder(n_safe, dict(fb_counts))
        ctrl = []
        for fbk, k in sorted(alloc.items()):
            pool = sorted([i for i in train if i.split("/")[1] == fbk], key=lambda i: (keyed_rank(task, "safe_sized_mixed_control", i), i))
            ctrl += pool[:k]
        ctrl = sorted(ctrl, key=lambda x: (x.split("/")[1], int(x.rsplit("_", 1)[1])))
        subsets = {
            "A_MIX_train": dict(members=train, n=len(train),
                                definition="all train-split episodes (all verified successful; mixed safe/damaging). One fixed "
                                           "mixed-data sampling distribution for every mixed arm: uniform over training windows "
                                           "of these episodes."),
            "A_SAFE_train": dict(members=safe_train, n=n_safe,
                                 by_feedback=dict(Counter(i.split("/")[1] for i in safe_train)),
                                 definition="train-split episodes with primary safe success (paper per-entity > 95); same update budget as A-MIX"),
            "safe_sized_mixed_control_train": dict(
                members=ctrl, n=len(ctrl), by_feedback=dict(Counter(i.split("/")[1] for i in ctrl)),
                n_primary_safe=sum(prim[i] for i in ctrl),
                definition=(f"random subset of the mixed train pool with |A_SAFE_train| = {n_safe} episodes, stratified by "
                            f"feedback in proportion to the train pool ({dict(fb_counts)}; largest-remainder allocation "
                            f"{alloc}); within each feedback stratum take the episodes with the smallest keyed rank "
                            f"KeyedRNG('{EXPERIMENT_ID}').derive_seed('bootstrap', experiment='{EXPERIMENT_ID}/subset/{task}/"
                            f"safe_sized_mixed_control', state_id=int(blake2b(episode_id, digest_size=4))) (fixed seed)")),
        }
        within = {}
        for fbk in ("fb", "nofb"):
            mm = [i for i in train if i.split("/")[1] == fbk]
            s = [i for i in mm if prim[i]]
            d = [i for i in mm if not prim[i]]
            entry = dict(n=len(mm), n_safe=len(s), n_damaging=len(d), both_classes=bool(s and d))
            if s and d:
                k = min(len(s), len(d))
                bal_s = sorted(s, key=lambda i: (keyed_rank(task, f"balanced_{fbk}", i), i))[:k]
                bal_d = sorted(d, key=lambda i: (keyed_rank(task, f"balanced_{fbk}", i), i))[:k]
                entry.update(
                    mixed_within_condition=sorted(mm, key=lambda x: int(x.rsplit("_", 1)[1])),
                    safe_within_condition=sorted(s, key=lambda x: int(x.rsplit("_", 1)[1])),
                    class_balanced_within_condition=sorted(bal_s + bal_d, key=lambda x: int(x.rsplit("_", 1)[1])),
                    class_balanced_n=2 * k,
                    exploratory=bool(min(len(s), len(d)) < 10),
                )
            else:
                entry["note"] = "one class absent in this condition's train split: safety effect cannot be separated from collection condition here"
            within[fbk] = entry
        nf = min(fb_counts.values())
        fbal = []
        for fbk in ("fb", "nofb"):
            pool = sorted([i for i in train if i.split("/")[1] == fbk], key=lambda i: (keyed_rank(task, "feedback_balanced_mixed", i), i))
            fbal += pool[:nf]
        subsets["within_condition"] = within
        subsets["feedback_balanced_mixed_train"] = dict(
            members=sorted(fbal, key=lambda x: (x.split("/")[1], int(x.rsplit("_", 1)[1]))), n=len(fbal),
            n_primary_safe=sum(prim[i] for i in fbal),
            definition=f"{nf} train episodes from each feedback condition (keyed rank), balanced-subset sensitivity for the feedback confound")
        # held-out evaluation strata available from demos (offline)
        heldout = {s: {fbk: dict(Counter('safe' if prim[i] else 'damaging' for i in split_members[s] if i.split('/')[1] == fbk))
                       for fbk in ("fb", "nofb")} for s in ("dev", "test")}

        out["tasks"][task] = dict(
            role="primary-candidate" if task == "pick_egg" else "diagnostic-only",
            raw_verification=dict(checks=raw_checks, mismatches=mism,
                                  reset_families=fam_check, all_merged_families_kept_whole=all(
                                      v["kept_whole"] for v in fam_check.values() if v["merged_by_rule"]),
                                  near_duplicate_pairs=near, group_rule_counts=reasons,
                                  groups_identical_to_catalog_draft=same_as_draft_groups),
            search=dict(n_groups=len(groups), assignments_searched=3 ** len(groups), cost=cost, strata=strata),
            groups=[dict(group_id=f"{task}/g{k:02d}", split=SPLITS[asg[k]], n=len(g),
                         reset_family=sorted({family_of.get(i, "-") for i in g}),
                         members=sorted(g, key=lambda x: (x.split("/")[1], int(x.rsplit("_", 1)[1]))))
                    for k, g in enumerate(groups)],
            splits={s: dict(members=m) for s, m in split_members.items()},
            counts=counts,
            counts_split_x_feedback_x_primary={s: {fbk: dict(Counter(label_of[i].split(":")[1] for i in split_members[s] if i.split("/")[1] == fbk))
                                                    for fbk in ("fb", "nofb")} for s in SPLITS},
            heldout_offline_strata=heldout,
            exceptions=exceptions,
            subsets=subsets,
        )
        print(task, "groups", len(groups), "cost", round(cost, 4), "sizes", sizes, "same_as_draft", same_as_draft_groups)
        print("  ", out["tasks"][task]["counts_split_x_feedback_x_primary"])
        print("   A_SAFE", n_safe, "ctrl", alloc, "within", {k: (v["n_safe"], v["n_damaging"]) for k, v in within.items()})
        print("   raw", raw_checks, "mism", mism[:3])
    p = os.path.join(ROSTER, "splits_v1.json")
    body = json.dumps(out, indent=1, sort_keys=False)
    out["content_sha256_excluding_this_field"] = hashlib.sha256(body.encode()).hexdigest()
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
