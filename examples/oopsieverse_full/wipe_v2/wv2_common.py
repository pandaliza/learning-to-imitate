"""Shared constants and helpers for the wipe_countertop v2 screening evaluation (Amendment 8; spec
/edrive1/lizad/oopsieverse_full_study/reports/wipe_v2/wipe_v2_spec.md). New code only: the frozen evaluator
(baseline/rollout_eval.py, matrix/rollout_eval_matrix.py) is imported, never edited.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.dirname(HERE)
for _p in (os.path.join(FULL, "matrix"), os.path.join(FULL, "baseline"), FULL, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPO = "/scr/lizad/personal/learning-to-imitate"
OVS = "/edrive1/lizad/oopsieverse_full_study"
STUDY = os.path.join(REPO, "outputs/oopsieverse_full_study")
V2_RUNS = os.path.join(OVS, "runs/wipe_v2")
V2_REPORTS = os.path.join(OVS, "reports/wipe_v2")
POOLS_FILE = os.environ.get("WV2_POOLS_FILE_SMOKE_ONLY") or os.path.join(V2_RUNS, "pools/pools_v1.json")
BANK_FILE = os.path.join(STUDY, "roster/reset_banks_v1.json")
SPLITS_FILE = os.path.join(STUDY, "roster/splits_v1.json")
TELEOP = os.path.join(OVS, "data/oopsieverse-demos/paper_experiments/teleop_data/wipe_countertop/{cond}_live_feedback.hdf5")
TASK = "wipe_countertop"
COND = {"fb": "with", "nofb": "without"}

# Layout groups (splits_v1 wipe_countertop groups; one teleop session = one ep_meta = one dirt layout + base pose).
# Test groups g04 (fb/demo_25-29) and g05 (fb/demo_5-9) are deliberately absent: not touched during screening.
TRAIN_GROUPS = ["g01", "g02", "g06"]
DEV_GROUPS = ["g00", "g03"]
FORBIDDEN_GROUPS = ["g04", "g05"]
POOL_SPEC = {"P-train": (TRAIN_GROUPS, 20), "P-devlayout": (DEV_GROUPS, 20)}
P_RANDOM_N = 50  # spec: "the existing frozen eval_dev bank (50 fully random resets)" -> eval_dev entries 0000-0049
SEED_EXPERIMENT = "oopsie_full_v1/wipe_v2/pools_v1/{pool}/{group}"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_json(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dirt_hash12(dirt):
    """Catalog / splits_v1 convention (catalog/inventory_tasks_and_data.py): sha256(json.dumps(positions))[:12]."""
    return hashlib.sha256(json.dumps(dirt).encode()).hexdigest()[:12]


def split_groups():
    sp = json.load(open(SPLITS_FILE))["tasks"][TASK]
    return {g["group_id"].split("/")[-1]: g for g in sp["groups"]}


def group_demos(gid):
    """[(cond, demo)] of a splits_v1 group, in listed order."""
    g = split_groups()[gid]
    out = []
    for m in g["members"]:
        _, fb, demo = m.split("/")
        out.append((COND[fb], demo))
    return out


def alias_seg_lut(class_to_id, seg_id_to_class):
    """Counter-label alias (Amendment 7.2): identical to wipe_diag/diag_rollout.AliasPolicy.seg_lut. A name that maps
    to 'unknown' is mapped to 'Damageable'+name when that is a vocabulary class (plain env.reset() names the fixture
    'Counter'; playback / training data name it 'DamageableCounter')."""
    m = max(int(k) for k in seg_id_to_class)
    lut = np.zeros(m + 1, dtype=np.int64)
    for k, name in seg_id_to_class.items():
        cid = class_to_id.get(name, 0)
        if cid == 0 and ("Damageable" + name) in class_to_id:
            cid = class_to_id["Damageable" + name]
        lut[int(k)] = cid
    return lut


def frozen_seg_lut(class_to_id, seg_id_to_class):
    """The frozen evaluator's LUT (rollout_eval.Policy.seg_lut), for comparison only."""
    m = max(int(k) for k in seg_id_to_class)
    lut = np.zeros(m + 1, dtype=np.int64)
    for k, name in seg_id_to_class.items():
        lut[int(k)] = class_to_id.get(name, 0)
    return lut


def load_pools(check=True):
    d = json.load(open(POOLS_FILE))
    if check:
        body = {k: v for k, v in d.items() if k != "content_sha256"}
        assert sha256_json(body) == d["content_sha256"], "pools_v1.json content hash mismatch (frozen file modified?)"
        assert d["status"].startswith("FROZEN"), d["status"]
    return d


def pool_resets(pool, pools=None):
    """Reset dicts for rollout_eval.apply_reset, in frozen order. P-train / P-devlayout: sticky-ep_meta resets
    {seed, ep_meta}; P-random: canonical bank resets {seed} (eval_dev[0:50], frozen reset_banks_v1)."""
    pools = pools or load_pools()
    if pool == "P-random":
        bank = json.load(open(BANK_FILE))
        assert bank["content_sha256"] == pools["P-random"]["bank_content_sha256"]
        ents = bank["tasks"][TASK]["pools"]["eval_dev"]["entries"][: pools["P-random"]["n"]]
        assert [e["reset_id"] for e in ents] == pools["P-random"]["reset_ids"]
        return [{"id": e["reset_id"].replace("/", "__"), "seed": e["seed"], "family": e["reset_id"],
                 "obj_pos": e["obj_pos"], "base_xy": e["base_xy"], "state_sha": e["state_sha"],
                 "dirt_sha": e.get("dirt_sha"), "in_support": e.get("obj_rel_base_inside_demo_box"),
                 "pool": pool, "layout_group": None} for e in ents]
    P = pools["pools"][pool]
    lay = pools["layouts"]
    out = []
    for e in P["entries"]:
        L = lay[e["layout_group"]]
        em = json.loads(L["ep_meta_json"])
        out.append({"id": e["reset_id"].replace("/", "__"), "seed": e["seed"], "family": e["reset_id"],
                    "ep_meta": em, "obj_pos": e["obj_pos"], "state_sha": e["state_sha"], "dirt_sha": e["dirt_sha"],
                    "pool": pool, "layout_group": e["layout_group"], "in_support": None})
    return out
