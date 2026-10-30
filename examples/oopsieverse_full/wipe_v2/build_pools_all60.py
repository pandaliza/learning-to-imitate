"""Pools for the matched-condition baseline check (spec addendum 2026-10-27, "paper condition"), frozen before any
all60 checkpoint is evaluated. pools_v1.json and pools_demo_v1.json are not touched.

  P-demo60   exact starts of ALL 60 wipe demos (both teleop files; splits_v1 groups g00..g06 incl. test groups g04/g05,
             which this separate check is allowed to use): build_pool_demo.restore_and_check (apply_reset with ep_meta +
             model_file + states[0], then observable refresh; per-entry deviations vs the recorded obs[0]); a second
             shuffled restore pass in the same env.            -> pools/pools_demo60_v1.json
  P-layout60 sticky-ep_meta resets, 20 per layout over the 7 layouts (140), built and verified exactly like P-train /
             P-devlayout (build_pools.snapshot in the evaluator env; dirt == source session exactly, one base pose per
             layout, placement differs; verify = a NEW process re-resetting all 140 in shuffled order).
                                                               -> pools/pools_layout60_v1.json

  python build_pools_all60.py demo60-build | demo60-freeze | layout60-build | layout60-verify | layout60-freeze
"""
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wv2_common as C  # noqa: E402
import rollout_eval as RE  # noqa: E402
import build_pools as BP  # noqa: E402
import build_pool_demo as BPD  # noqa: E402

OUT = os.path.join(C.V2_RUNS, "pools")
ALL_GROUPS = ["g00", "g01", "g02", "g03", "g04", "g05", "g06"]
DEMO60_FILE = os.path.join(OUT, "pools_demo60_v1.json")
LAYOUT60_FILE = os.path.join(OUT, "pools_layout60_v1.json")
SEED_DEMO60 = "oopsie_full_v1/wipe_v2/pools_demo60_v1/P-demo60/{group}"
SEED_LAYOUT60 = "oopsie_full_v1/wipe_v2/pools_layout60_v1/P-layout60/{group}"
PER_LAYOUT = 20


def used_seeds():
    s = BP.bank_wipe_seeds()
    s |= {e["seed"] for p in C.load_pools()["pools"].values() for e in p["entries"]}
    s |= {e["seed"] for e in json.load(open(BPD.POOL_DEMO_FILE))["entries"]}
    return s


def _freeze(path, doc):
    if os.path.exists(path):
        sys.exit(f"REFUSE: {path} exists (frozen)")
    doc["content_sha256"] = C.sha256_json(doc)
    json.dump(doc, open(path + ".tmp", "w"), indent=1, default=float)
    os.replace(path + ".tmp", path)
    fsha = C.sha256_file(path)
    open(path + ".sha256", "w").write(f"{fsha}  {os.path.basename(path)}\ncontent_sha256 {doc['content_sha256']}\n")
    os.chmod(path, 0o444)
    print("FROZEN", path, "file_sha256", fsha, "content_sha256", doc["content_sha256"], flush=True)


# ─────────────────────────────── P-demo60 ───────────────────────────────

def demo60_entries():
    KR = BP.keyed_rng()
    used = used_seeds()
    ents = []
    for gid in ALL_GROUPS:
        for k, (cond, demo) in enumerate(C.group_demos(gid)):
            i = k
            while True:
                s = int(KR.derive_seed("env", experiment=SEED_DEMO60.format(group=gid), state_id=i))
                if s not in used:
                    break
                i += 1000
            used.add(s)
            ents.append({"reset_id": f"wipe_v2/P-demo60/{gid}/{cond}__{demo}", "layout_group": gid, "cond": cond, "demo": demo,
                         "seed": s, "teleop_file": C.TELEOP.format(cond=cond),
                         "split": C.split_groups()[gid]["split"]})
    return ents


def cmd_demo60_build():
    ents = demo60_entries()
    os.chdir(RE.OOPSIE)
    env = BP.make_env()
    part = os.path.join(OUT, "demo60_build_partial.jsonl")
    done = {}
    if os.path.exists(part):
        for line in open(part):
            try:
                x = json.loads(line)
                done[x["reset_id"]] = x
            except Exception:
                pass
    pf = open(part, "a")
    t0 = time.time()
    for e in ents:
        if e["reset_id"] in done and done[e["reset_id"]]["seed"] == e["seed"]:
            e.update(done[e["reset_id"]])
            continue
        e.update(BPD.restore_and_check(env, e))
        pf.write(json.dumps(e, default=float) + "\n")
        pf.flush()
        print(e["reset_id"], json.dumps({k: v for k, v in e["dev"].items() if not k.startswith("stale")}), f"{time.time() - t0:.0f}s", flush=True)
    order = list(range(len(ents)))
    random.Random(20261027).shuffle(order)
    for i in order:
        x = BPD.restore_and_check(env, ents[i])
        ents[i]["second_pass_state_equal"] = x["fp"]["state_sha"] == ents[i]["fp"]["state_sha"]
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "egl": os.environ.get("MUJOCO_EGL_DEVICE_ID"), "entries": ents,
               "wall_s": time.time() - t0}, open(os.path.join(OUT, "demo60_build.json"), "w"), indent=1, default=float)
    print("BUILD_DONE", len(ents), "second pass equal", sum(e["second_pass_state_equal"] for e in ents), flush=True)
    os._exit(0)


def cmd_demo60_freeze():
    b = json.load(open(os.path.join(OUT, "demo60_build.json")))
    ents = b["entries"]
    keys = sorted({k for e in ents for k in e["dev"]})
    maxdev = {k: max(e["dev"][k] for e in ents if isinstance(e["dev"].get(k), float)) for k in keys}
    want = {g: len(C.group_demos(g)) for g in ALL_GROUPS}
    checks = {"n": len(ents), "per_group": {g: sum(e["layout_group"] == g for e in ents) for g in ALL_GROUPS},
              "max_deviation_vs_recorded_obs0": maxdev,
              "second_pass_state_reproduced": f"{sum(e['second_pass_state_equal'] for e in ents)}/{len(ents)}",
              "counter_visible_alias_eye_in_hand": f"{sum(e['counter_px_at_reset']['robot0_eye_in_hand/alias'] > 0 for e in ents)}/{len(ents)}",
              "alias_lut_equals_frozen_lut": f"{sum(e['alias_lut_equals_frozen_lut'] for e in ents)}/{len(ents)}",
              "markers_wiped_at_restore": max(e["n_wiped_at_restore"] for e in ents)}
    ok = (len(ents) == 60 and checks["per_group"] == want and all(e["second_pass_state_equal"] for e in ents)
          and maxdev["sim_state_vs_states0_excl_time"] == 0.0 and maxdev["dirt_marker_xy_vs_ep_meta"] < 1e-9
          and checks["markers_wiped_at_restore"] == 0 and maxdev["robot0_eef_pos"] < 1e-3 and maxdev["sponge_pos"] < 1e-3
          and maxdev["robot0_joint_pos"] < 1e-3 and all(e["counter_px_at_reset"]["robot0_eye_in_hand/alias"] > 0 for e in ents))
    print(json.dumps(checks, indent=1))
    if not ok:
        sys.exit("FREEZE REFUSED")
    keep = ("reset_id", "layout_group", "split", "cond", "demo", "seed", "teleop_file")
    doc = {"status": "FROZEN v1", "version": "wipe_v2_pools_demo60_v1", "pool": "P-demo60", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "declared_before_any_all60_evaluation": True,
           "spec": "wipe_v2_spec.md, Addendum (2026-10-27): matched-condition baseline check",
           "spec_sha256": C.sha256_file(os.path.join(C.V2_REPORTS, "wipe_v2_spec.md")),
           "procedure": {"reset": "rollout_eval.apply_reset(env, {seed, ep_meta, model_xml, state=states[0]}) + observable refresh "
                                  "(build_pool_demo.apply_reset_refresh); then normal physics + policy actions",
                         "seed": "KeyedRNG('oopsie_full_v1').derive_seed('env', experiment='" + SEED_DEMO60 + "', state_id=k)",
                         "code_sha256": {f: C.sha256_file(os.path.join(HERE, f)) for f in ("build_pools_all60.py", "build_pool_demo.py", "wv2_common.py")}},
           "entries": [dict({k: e[k] for k in keep}, **e["shas"], state_sha=e["fp"]["state_sha"], dirt_sha=e["fp"]["dirt_sha"],
                            obj_pos=e["fp"]["obj_pos"], base_pos=e["fp"]["base_pos"], deviation_vs_recorded_obs0=e["dev"])
                       for e in ents],
           "checks": checks}
    _freeze(DEMO60_FILE, doc)


# ─────────────────────────────── P-layout60 ───────────────────────────────

def layouts7():
    from path_remap import remap_asset_paths
    import h5py
    out = {}
    for gid in ALL_GROUPS:
        demos = C.group_demos(gid)
        raws = []
        for cond, demo in demos:
            with h5py.File(C.TELEOP.format(cond=cond), "r") as f:
                r = f["data"][demo].attrs["ep_meta"]
                raws.append(r.decode() if isinstance(r, bytes) else r)
        em_json = remap_asset_paths(raws[0])
        em = json.loads(em_json)
        g = C.split_groups()[gid]
        out[gid] = {"group": gid, "split": g["split"], "reset_family": g["reset_family"], "source_demo": f"{demos[0][0]}/{demos[0][1]}",
                    "members": [f"{c}/{d}" for c, d in demos],
                    "all_members_identical_ep_meta": len({C.sha256_json(json.loads(r)) for r in raws}) == 1,
                    "raw_ep_meta_sha256": C.sha256_json(json.loads(raws[0])), "ep_meta_sha256": C.sha256_json(em), "ep_meta_json": em_json,
                    "dirt_hash12": C.dirt_hash12(em["dirt_marker_positions"]), "dirt_marker_positions": em["dirt_marker_positions"],
                    "init_robot_base_pos": em["init_robot_base_pos"], "init_robot_base_ori": em["init_robot_base_ori"]}
        assert g["reset_family"][0].endswith("dirt:" + out[gid]["dirt_hash12"]), gid
    return out


def layout60_entries():
    KR = BP.keyed_rng()
    used = used_seeds()
    ents = []
    for gid in ALL_GROUPS:
        i = k = 0
        while k < PER_LAYOUT:
            s = int(KR.derive_seed("env", experiment=SEED_LAYOUT60.format(group=gid), state_id=i))
            if s not in used:
                used.add(s)
                ents.append({"pool": "P-layout60", "layout_group": gid, "key_index": i, "seed": s,
                             "reset_id": f"wipe_v2/P-layout60/{gid}/{k:04d}"})
                k += 1
            i += 1
    return ents


def cmd_layout60_build():
    lay = layouts7()
    ents = layout60_entries()
    env = BP.make_env()
    t0 = time.time()
    for e in ents:
        L = lay[e["layout_group"]]
        obs, _ = RE.apply_reset(env, {"seed": e["seed"], "ep_meta": json.loads(L["ep_meta_json"])})
        s = BP.snapshot(env, obs)
        e.update(s)
        dirt = L["dirt_marker_positions"]
        e["check"] = {"dirt_env_equals_source_exact": s["marker_xy_env"] == [[float(a), float(b)] for a, b in dirt],
                      "dirt_sim_xy_equals_source_maxabs": float(np.max(np.abs(np.asarray(s["marker_world_pos"])[:, :2] - np.asarray(dirt)))),
                      "base_equals_source_exact": [float(x) for x in env.init_robot_base_pos] == [float(x) for x in L["init_robot_base_pos"]],
                      "ep_meta_dirt_after_equals_source": env.get_ep_meta()["dirt_marker_positions"] == dirt}
        print(e["reset_id"], e["seed"], s["fp"]["state_sha"], e["check"], f"{time.time() - t0:.0f}s", flush=True)
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "egl": os.environ.get("MUJOCO_EGL_DEVICE_ID"), "layouts": lay, "entries": ents,
               "wall_s": time.time() - t0}, open(os.path.join(OUT, "layout60_build.json"), "w"))
    print("BUILD_DONE", len(ents), flush=True)
    os._exit(0)


def cmd_layout60_verify():
    b = json.load(open(os.path.join(OUT, "layout60_build.json")))
    jobs = list(b["entries"])
    random.Random(20261027).shuffle(jobs)
    env = BP.make_env()
    rows = []
    for e in jobs:
        L = b["layouts"][e["layout_group"]]
        obs, _ = RE.apply_reset(env, {"seed": e["seed"], "ep_meta": json.loads(L["ep_meta_json"])})
        fp = BP.snapshot(env, obs)["fp"]
        rows.append({"reset_id": e["reset_id"], "state_equal_build": fp["state_sha"] == e["fp"]["state_sha"],
                     "dirt_equal_build": fp["dirt_sha"] == e["fp"]["dirt_sha"]})
        print(json.dumps(rows[-1]), flush=True)
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "egl": os.environ.get("MUJOCO_EGL_DEVICE_ID"), "rows": rows},
              open(os.path.join(OUT, "layout60_verify.json"), "w"), indent=1)
    print("VERIFY_DONE", len(rows), flush=True)
    os._exit(0)


def cmd_layout60_freeze():
    b = json.load(open(os.path.join(OUT, "layout60_build.json")))
    v = json.load(open(os.path.join(OUT, "layout60_verify.json")))["rows"]
    ents = b["entries"]
    per_group = {}
    for g in ALL_GROUPS:
        es = [e for e in ents if e["layout_group"] == g]
        P = np.asarray([e["fp"]["obj_pos"] for e in es])
        dd = np.linalg.norm(P[:, None, :2] - P[None, :, :2], axis=-1)[np.triu_indices(len(es), 1)]
        per_group[g] = {"n": len(es), "dirt_sha_unique": sorted({e["fp"]["dirt_sha"] for e in es}),
                        "base_unique": len({tuple(e["fp"]["init_robot_base_pos"]) for e in es}),
                        "state_sha_unique": len({e["fp"]["state_sha"] for e in es}),
                        "sponge_xy_pairwise_min_mm": float(dd.min() * 1000), "sponge_xy_span_cm": (np.ptp(P[:, :2], axis=0) * 100).round(2).tolist()}
    checks = {"n": len(ents), "layout_members_identical_ep_meta": all(L["all_members_identical_ep_meta"] for L in b["layouts"].values()),
              "dirt_matches_source_exact_all": all(e["check"]["dirt_env_equals_source_exact"] and e["check"]["ep_meta_dirt_after_equals_source"]
                                                   and e["check"]["dirt_sim_xy_equals_source_maxabs"] < 1e-9 for e in ents),
              "base_matches_source_exact_all": all(e["check"]["base_equals_source_exact"] for e in ents),
              "one_dirt_layout_per_group": all(len(x["dirt_sha_unique"]) == 1 for x in per_group.values()),
              "one_base_pose_per_group": all(x["base_unique"] == 1 for x in per_group.values()),
              "object_placement_differs_within_group": all(x["sponge_xy_pairwise_min_mm"] > 0 and x["state_sha_unique"] == x["n"] for x in per_group.values()),
              "counter_visible_alias_all": all(e["counter_px_at_reset"]["robot0_eye_in_hand/alias"] > 0 for e in ents),
              "verify_state_reproduced": f"{sum(r['state_equal_build'] for r in v)}/{len(v)}",
              "verify_dirt_reproduced": f"{sum(r['dirt_equal_build'] for r in v)}/{len(v)}"}
    ok = (len(ents) == 140 and len(v) == 140 and all(r["state_equal_build"] and r["dirt_equal_build"] for r in v)
          and all(checks[k] for k in ("layout_members_identical_ep_meta", "dirt_matches_source_exact_all", "base_matches_source_exact_all",
                                      "one_dirt_layout_per_group", "one_base_pose_per_group", "object_placement_differs_within_group",
                                      "counter_visible_alias_all")))
    print(json.dumps(checks, indent=1))
    print(json.dumps(per_group, indent=1))
    if not ok:
        sys.exit("FREEZE REFUSED")
    lay = {g: dict(L, marker_world_pos=next(e["marker_world_pos"] for e in ents if e["layout_group"] == g)) for g, L in b["layouts"].items()}
    doc = {"status": "FROZEN v1", "version": "wipe_v2_pools_layout60_v1", "pool": "P-layout60", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "declared_before_any_all60_evaluation": True,
           "spec": "wipe_v2_spec.md, Addendum (2026-10-27): matched-condition baseline check",
           "spec_sha256": C.sha256_file(os.path.join(C.V2_REPORTS, "wipe_v2_spec.md")),
           "procedure": {"reset": "rollout_eval.apply_reset(env, {'seed': s, 'ep_meta': E}) (sticky ep_meta, as pools_v1 P-train/P-devlayout)",
                         "seed_derivation": "KeyedRNG('oopsie_full_v1').derive_seed('env', experiment='" + SEED_LAYOUT60 + "', state_id=i); "
                                            "skips seeds used by reset_banks_v1 wipe, pools_v1, pools_demo_v1",
                         "code_sha256": {f: C.sha256_file(os.path.join(HERE, f)) for f in ("build_pools_all60.py", "build_pools.py", "wv2_common.py")}},
           "layouts": lay,
           "entries": [{"reset_id": e["reset_id"], "pool": "P-layout60", "layout_group": e["layout_group"], "key_index": e["key_index"],
                        "seed": e["seed"], "state_sha": e["fp"]["state_sha"], "dirt_sha": e["fp"]["dirt_sha"],
                        "model_sha_norm": e["fp"]["model_sha_norm"], "base_pos": e["fp"]["base_pos"], "obj_pos": e["fp"]["obj_pos"],
                        "obj_yaw": e["fp"]["obj_yaw"], "objects": e["objects"], "eef_pos": e["fp"]["eef_pos"], "joint_pos": e["fp"]["joint_pos"]}
                       for e in ents],
           "checks": checks, "per_group": per_group}
    _freeze(LAYOUT60_FILE, doc)


# ─────────────────────────────── loaders (evaluator) ───────────────────────────────

def load_pool_doc(path):
    d = json.load(open(path))
    body = {k: v for k, v in d.items() if k != "content_sha256"}
    assert C.sha256_json(body) == d["content_sha256"], f"{path}: content hash mismatch"
    return d


def demo60_resets():
    d = load_pool_doc(DEMO60_FILE)
    out = []
    for e in d["entries"]:
        r, _, _ = BPD.load_demo(e, check=e)
        r.update(pool="P-demo60", state_sha=e["state_sha"], dirt_sha=e["dirt_sha"], obj_pos=e["obj_pos"])
        out.append(r)
    return out


def layout60_resets():
    d = load_pool_doc(LAYOUT60_FILE)
    out = []
    for e in d["entries"]:
        em = json.loads(d["layouts"][e["layout_group"]]["ep_meta_json"])
        out.append({"id": e["reset_id"].replace("/", "__"), "seed": e["seed"], "family": e["reset_id"], "ep_meta": em,
                    "obj_pos": e["obj_pos"], "state_sha": e["state_sha"], "dirt_sha": e["dirt_sha"], "pool": "P-layout60",
                    "layout_group": e["layout_group"], "in_support": None})
    return out


if __name__ == "__main__":
    {"demo60-build": cmd_demo60_build, "demo60-freeze": cmd_demo60_freeze, "layout60-build": cmd_layout60_build,
     "layout60-verify": cmd_layout60_verify, "layout60-freeze": cmd_layout60_freeze}[sys.argv[1]]()
