"""Build, verify and freeze the wipe v2 screening pools (spec "Evaluation pools"), BEFORE any v2 policy is evaluated.

  P-train      60 resets: 20 per training layout group (splits_v1 g01 fb/demo_10-14, g02 fb/demo_15-19, g06 nofb/demo_0-29)
  P-devlayout  40 resets: 20 per dev layout group (g00 fb/demo_0-4, g03 fb/demo_20-24)
  P-random     pointer to the frozen reset_banks_v1 eval_dev bank, entries 0000-0049 (canonical resets)
Test groups g04/g05 and the eval_test bank are never read.

Sticky-ep_meta mechanism (roster_v1 T4): reset = rollout_eval.apply_reset(env, {"seed": s, "ep_meta": E}) i.e.
env.set_ep_meta(E); env.rng = default_rng(s); env.reset() (+ the evaluator's get_observations()). E = the session's
stored ep_meta (asset paths remapped by path_remap, nothing else) fixes layout/style, fixtures, base pose and dirt
layout; robot joint noise and the sponge placement are drawn from env.rng(s). Seeds:
KeyedRNG('oopsie_full_v1').derive_seed('env', experiment='oopsie_full_v1/wipe_v2/pools_v1/<pool>/<group>', state_id=i),
checked disjoint from every reset_banks_v1 wipe seed (incl. spares) and from each other.

  build   reset every entry in order in ONE evaluator env (rollout_eval.make_env, use_camera_obs=False, ctxfix), record
          fingerprints (reset_probe.fingerprint) + ep_meta / dirt / base / object placements -> pools/build.json
  verify  in a NEW process, re-reset all pool entries AND P-random in a shuffled, interleaved order (exercises the
          used-env / sticky-ep_meta history) and compare fingerprints -> pools/verify.json
  freeze  check every predeclared condition and write pools/pools_v1.json (+ content_sha256, file sha256); refuses
          to overwrite an existing frozen file.
Render: EGL on one parity-passed GPU (MUJOCO_EGL_DEVICE_ID), CUDA_VISIBLE_DEVICES="".
"""
import json
import os
import random
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wv2_common as C  # noqa: E402
import rollout_eval as RE  # noqa: E402

OUT = os.path.join(C.V2_RUNS, "pools")
CAMS = ["robot0_eye_in_hand", "robot0_agentview_right", "robot0_agentview_left"]
CLASS_TO_ID = {"unknown": 0, "DamageableCounter": 1, "dirt": 2, "robot": 3, "sponge": 4}


def keyed_rng():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ks_rng", os.path.join(C.REPO, "mip/intent_selection/rng.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.KeyedRNG("oopsie_full_v1")


def bank_wipe_seeds():
    bank = json.load(open(C.BANK_FILE))
    s = set()
    for pool, p in bank["tasks"][C.TASK]["pools"].items():
        for e in p["entries"]:
            s.add(int(e["seed"]))
    try:
        sp = json.load(open(os.path.join(C.STUDY, "roster/evidence/bank_seeds_with_spares.json")))
        for k, v in sp["bank"][C.TASK].items():
            for e in v:
                s.add(int(e["seed"]))
    except Exception as ex:  # spares file layout differs: entries alone are still checked
        print("spares not read:", ex)
    return s


def layouts():
    """group -> layout record (ep_meta of the first member; every member's ep_meta must be identical)."""
    from path_remap import remap_asset_paths
    out = {}
    for gid in C.TRAIN_GROUPS + C.DEV_GROUPS:
        assert gid not in C.FORBIDDEN_GROUPS
        demos = C.group_demos(gid)
        raws = []
        for cond, demo in demos:
            with h5py.File(C.TELEOP.format(cond=cond), "r") as f:
                raws.append(f["data"][demo].attrs["ep_meta"])
        raws = [r.decode() if isinstance(r, bytes) else r for r in raws]
        canon = [C.sha256_json(json.loads(r)) for r in raws]
        em_json = remap_asset_paths(raws[0])
        em = json.loads(em_json)
        out[gid] = {"group": gid, "split": C.split_groups()[gid]["split"],
                    "reset_family": C.split_groups()[gid]["reset_family"],
                    "source_demo": f"{demos[0][0]}/{demos[0][1]}", "members": [f"{c}/{d}" for c, d in demos],
                    "all_members_identical_ep_meta": len(set(canon)) == 1,
                    "raw_ep_meta_sha256": C.sha256_json(json.loads(raws[0])),
                    "ep_meta_sha256": C.sha256_json(em), "ep_meta_json": em_json,
                    "dirt_hash12": C.dirt_hash12(em["dirt_marker_positions"]),
                    "dirt_marker_positions": em["dirt_marker_positions"],
                    "init_robot_base_pos": em["init_robot_base_pos"], "init_robot_base_ori": em["init_robot_base_ori"],
                    "layout_id": em["layout_id"], "style_id": em["style_id"]}
        fam = out[gid]["reset_family"][0]
        assert fam.endswith("dirt:" + out[gid]["dirt_hash12"]), (gid, fam, out[gid]["dirt_hash12"])
    return out


def derive_entries():
    KR = keyed_rng()
    used = bank_wipe_seeds()
    n_bank = len(used)
    ents = []
    for pool, (groups, n) in C.POOL_SPEC.items():
        for gid in groups:
            exp = C.SEED_EXPERIMENT.format(pool=pool, group=gid)
            i = k = 0
            while k < n:
                s = int(KR.derive_seed("env", experiment=exp, state_id=i))
                if s not in used:
                    used.add(s)
                    ents.append({"pool": pool, "layout_group": gid, "key_index": i, "seed": s,
                                 "reset_id": f"wipe_v2/{pool}/{gid}/{k:04d}"})
                    k += 1
                i += 1
    return ents, n_bank


def make_env():
    os.chdir(RE.OOPSIE)
    from ctxfix import install_ctx_free
    env = RE.make_env(C.TASK, use_camera_obs=False)
    install_ctx_free(env)
    return env


def snapshot(env, obs):
    """Fingerprint + layout/object detail of the current post-reset state."""
    from reset_probe import fingerprint
    fp = fingerprint(env, C.TASK, obs)
    sim = env.sim
    mk = []
    for m in env.markers:
        bid = sim.model.body_name2id(m.root_body)
        mk.append([float(x) for x in sim.data.body_xpos[bid]])
    objs = {}
    for name, bid in env.obj_body_id.items():
        objs[name] = {"pos": [float(x) for x in sim.data.body_xpos[bid]], "quat_wxyz": [float(x) for x in sim.data.body_xquat[bid]]}
    names = sorted(set(env.seg_id_to_class.values()))
    px = {}
    for cam in CAMS:
        raw = np.asarray(obs[f"{cam}_segmentation_class"]).astype(np.int64)
        for kind, fn in (("alias", C.alias_seg_lut), ("frozen", C.frozen_seg_lut)):
            lut = fn(CLASS_TO_ID, env.seg_id_to_class)
            px[f"{cam}/{kind}"] = int((lut[np.clip(raw, 0, len(lut) - 1)] == 1).sum())
    return {"fp": fp, "marker_world_pos": mk, "marker_xy_env": [[float(p[0]), float(p[1])] for p in env._marker_positions],
            "objects": objs, "ep_meta_after_sha256": C.sha256_json(env.get_ep_meta()),
            "seg_class_names": names, "counter_px_at_reset": px}


def cmd_build():
    lay = layouts()
    ents, n_bank = derive_entries()
    env = make_env()
    t0 = time.time()
    part = os.path.join(OUT, "build_partial.jsonl")  # resumable: a GPU-headroom kill loses at most one reset
    done = {}
    if os.path.exists(part):
        for line in open(part):
            try:
                x = json.loads(line)
                done[x["reset_id"]] = x
            except Exception:
                pass
    pf = open(part, "a")
    for k, e in enumerate(ents):
        if e["reset_id"] in done:
            assert done[e["reset_id"]]["seed"] == e["seed"]
            e.update(done[e["reset_id"]])
            continue
        L = lay[e["layout_group"]]
        obs, _ = RE.apply_reset(env, {"seed": e["seed"], "ep_meta": json.loads(L["ep_meta_json"])})
        s = snapshot(env, obs)
        e.update(s)
        dirt = L["dirt_marker_positions"]
        e["check"] = {
            "dirt_env_equals_source_exact": s["marker_xy_env"] == [[float(a), float(b)] for a, b in dirt],
            "dirt_sim_xy_equals_source_maxabs": float(np.max(np.abs(np.asarray(s["marker_world_pos"])[:, :2] - np.asarray(dirt)))),
            "base_equals_source_exact": [float(x) for x in env.init_robot_base_pos] == [float(x) for x in L["init_robot_base_pos"]],
            "base_obs_minus_source_maxabs": float(np.max(np.abs(np.asarray(s["fp"]["base_pos"][:2]) - np.asarray(L["init_robot_base_pos"][:2])))),
            "ep_meta_dirt_after_equals_source": env.get_ep_meta()["dirt_marker_positions"] == dirt,
        }
        print(k, e["reset_id"], e["seed"], s["fp"]["state_sha"], s["fp"]["obj_pos"], e["check"], f"{time.time() - t0:.0f}s", flush=True)
        e["build_process"] = os.getpid()
        pf.write(json.dumps(e) + "\n")
        pf.flush()
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "egl": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
               "n_bank_seeds_checked": n_bank, "layouts": lay, "entries": ents, "wall_s": time.time() - t0},
              open(os.path.join(OUT, "build.json.tmp"), "w"))
    os.replace(os.path.join(OUT, "build.json.tmp"), os.path.join(OUT, "build.json"))
    print("BUILD_DONE", len(ents))


def cmd_verify():
    b = json.load(open(os.path.join(OUT, "build.json")))
    bank = json.load(open(C.BANK_FILE))["tasks"][C.TASK]["pools"]["eval_dev"]["entries"][: C.P_RANDOM_N]
    jobs = [("pool", e) for e in b["entries"]] + [("bank", e) for e in bank]
    random.Random(20261025).shuffle(jobs)
    env = make_env()
    rows = []
    for kind, e in jobs:
        if kind == "pool":
            L = b["layouts"][e["layout_group"]]
            obs, _ = RE.apply_reset(env, {"seed": e["seed"], "ep_meta": json.loads(L["ep_meta_json"])})
        else:
            obs, _ = RE.apply_reset(env, {"seed": e["seed"]})
        s = snapshot(env, obs)
        fp = s["fp"]
        r = {"kind": kind, "reset_id": e["reset_id"], "seed": e["seed"], "state_sha": fp["state_sha"],
             "dirt_sha": fp["dirt_sha"], "obj_pos": fp["obj_pos"], "seg_class_names": s["seg_class_names"],
             "counter_px_at_reset": s["counter_px_at_reset"]}
        if kind == "pool":
            r["state_equal_build"] = fp["state_sha"] == e["fp"]["state_sha"]
            r["model_equal_build"] = fp["model_sha_norm"] == e["fp"]["model_sha_norm"]
            r["dirt_equal_build"] = fp["dirt_sha"] == e["fp"]["dirt_sha"]
        else:
            r["state_equal_bank"] = fp["state_sha"] == e["state_sha"]
            r["model_equal_bank"] = fp["model_sha_norm"] == e["model_sha_norm"]
            r["dirt_equal_bank"] = fp["dirt_sha"] == e.get("dirt_sha")
            r["obj_pos_vs_bank_maxabs"] = float(np.max(np.abs(np.asarray(fp["obj_pos"]) - np.asarray(e["obj_pos"]))))
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("seg_class_names", "counter_px_at_reset")}), flush=True)
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "egl": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
               "order": "shuffled Random(20261025), pool + P-random interleaved", "rows": rows},
              open(os.path.join(OUT, "verify.json"), "w"), indent=1)
    print("VERIFY_DONE", len(rows))


def cmd_freeze():
    if os.path.exists(C.POOLS_FILE):
        sys.exit(f"REFUSE: {C.POOLS_FILE} exists (frozen)")
    b = json.load(open(os.path.join(OUT, "build.json")))
    v = json.load(open(os.path.join(OUT, "verify.json")))["rows"]
    bank_all = json.load(open(C.BANK_FILE))
    ents = b["entries"]
    checks = {}
    checks["layout_members_identical_ep_meta"] = all(L["all_members_identical_ep_meta"] for L in b["layouts"].values())
    checks["no_test_group"] = not any(e["layout_group"] in C.FORBIDDEN_GROUPS for e in ents)
    checks["dirt_matches_source_exact_all"] = all(e["check"]["dirt_env_equals_source_exact"] and
                                                  e["check"]["ep_meta_dirt_after_equals_source"] and
                                                  e["check"]["dirt_sim_xy_equals_source_maxabs"] < 1e-9 for e in ents)
    checks["base_matches_source_exact_all"] = all(e["check"]["base_equals_source_exact"] for e in ents)
    per_group = {}
    for g in C.TRAIN_GROUPS + C.DEV_GROUPS:
        es = [e for e in ents if e["layout_group"] == g]
        P = np.asarray([e["fp"]["obj_pos"] for e in es])
        d = np.linalg.norm(P[:, None, :2] - P[None, :, :2], axis=-1)[np.triu_indices(len(es), 1)]
        J = np.asarray([e["fp"]["joint_pos"] for e in es])
        dj = np.abs(J[:, None] - J[None]).max(-1)[np.triu_indices(len(es), 1)]
        per_group[g] = {"n": len(es), "dirt_sha_unique": sorted({e["fp"]["dirt_sha"] for e in es}),
                        "base_unique": len({tuple(e["fp"]["init_robot_base_pos"]) for e in es}),
                        "state_sha_unique": len({e["fp"]["state_sha"] for e in es}),
                        "sponge_xy_pairwise_min_mm": float(d.min() * 1000), "sponge_xy_pairwise_median_mm": float(np.median(d) * 1000),
                        "sponge_xy_span_cm": (np.ptp(P[:, :2], axis=0) * 100).round(2).tolist(),
                        "sponge_yaw_span_rad": float(np.ptp([e["fp"]["obj_yaw"] for e in es])),
                        "joint_pos_pairwise_min_maxabs": float(dj.min())}
    checks["one_dirt_layout_per_group"] = all(len(x["dirt_sha_unique"]) == 1 for x in per_group.values())
    checks["one_base_pose_per_group"] = all(x["base_unique"] == 1 for x in per_group.values())
    checks["object_placement_differs_within_group"] = all(x["sponge_xy_pairwise_min_mm"] > 0 and x["state_sha_unique"] == x["n"]
                                                          for x in per_group.values())
    checks["counter_visible_alias_all"] = all(all(e["counter_px_at_reset"][f"{c}/alias"] > 0 for c in CAMS[:1]) for e in ents)
    vp = [r for r in v if r["kind"] == "pool"]
    vb = [r for r in v if r["kind"] == "bank"]
    checks["verify_pool_state_reproduced"] = f"{sum(r['state_equal_build'] for r in vp)}/{len(vp)}"
    checks["verify_pool_dirt_reproduced"] = f"{sum(r['dirt_equal_build'] for r in vp)}/{len(vp)}"
    checks["verify_bank_state_equals_bank"] = f"{sum(r['state_equal_bank'] for r in vb)}/{len(vb)}"
    checks["verify_bank_dirt_equals_bank"] = f"{sum(r['dirt_equal_bank'] for r in vb)}/{len(vb)}"
    checks["verify_bank_obj_pos_maxabs"] = max(r["obj_pos_vs_bank_maxabs"] for r in vb)
    ok = (checks["layout_members_identical_ep_meta"] and checks["no_test_group"] and checks["dirt_matches_source_exact_all"]
          and checks["base_matches_source_exact_all"] and checks["one_dirt_layout_per_group"]
          and checks["one_base_pose_per_group"] and checks["object_placement_differs_within_group"]
          and all(r["state_equal_build"] and r["dirt_equal_build"] for r in vp) and len(vp) == len(ents)
          and all(r["dirt_equal_bank"] for r in vb) and checks["verify_bank_obj_pos_maxabs"] < 1e-3)
    print(json.dumps(checks, indent=1))
    print(json.dumps(per_group, indent=1))
    if not ok:
        sys.exit("FREEZE REFUSED: a predeclared check failed")
    keep = ("reset_id", "pool", "layout_group", "key_index", "seed")
    pools = {}
    for pool, (groups, n) in C.POOL_SPEC.items():
        es = [e for e in ents if e["pool"] == pool]
        pools[pool] = {"n": len(es), "groups": groups, "per_group": n,
                       "entries": [dict({k: e[k] for k in keep}, state_sha=e["fp"]["state_sha"], dirt_sha=e["fp"]["dirt_sha"],
                                        model_sha_norm=e["fp"]["model_sha_norm"],
                                        ep_meta_sha256=b["layouts"][e["layout_group"]]["ep_meta_sha256"],
                                        dirt_hash12=b["layouts"][e["layout_group"]]["dirt_hash12"],
                                        base_pos=e["fp"]["base_pos"], init_robot_base_pos=e["fp"]["init_robot_base_pos"],
                                        obj_pos=e["fp"]["obj_pos"], obj_yaw=e["fp"]["obj_yaw"], obj_quat=e["fp"]["obj_quat"],
                                        objects=e["objects"], eef_pos=e["fp"]["eef_pos"], joint_pos=e["fp"]["joint_pos"],
                                        health_min=e["fp"]["health_min"], success_at_reset=e["fp"]["success_at_reset"])
                                   for e in es]}
    be = bank_all["tasks"][C.TASK]["pools"]["eval_dev"]["entries"][: C.P_RANDOM_N]
    lay = {g: dict(L, marker_world_pos=next(e["marker_world_pos"] for e in ents if e["layout_group"] == g))
           for g, L in b["layouts"].items()}
    doc = {
        "status": "FROZEN v1", "version": "wipe_v2_pools_v1", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spec": os.path.join(C.V2_REPORTS, "wipe_v2_spec.md"), "spec_sha256": C.sha256_file(os.path.join(C.V2_REPORTS, "wipe_v2_spec.md")),
        "declared_before_any_v2_policy_evaluation": True,
        "procedure": {
            "reset": "rollout_eval.apply_reset(env, {'seed': s, 'ep_meta': E}): env.set_ep_meta(E); env.rng = np.random.default_rng(s); "
                     "env.reset(); hide_markers; env.get_observations(). E = layouts[g].ep_meta_json (stored session ep_meta, asset paths "
                     "remapped by path_remap.remap_asset_paths only). Sticky ep_meta fixes layout/style/fixtures, base pose and dirt; "
                     "joint noise and sponge placement follow env.rng(s) (roster_v1 T4).",
            "env": "rollout_eval.make_env('wipe_countertop', use_camera_obs=False) + matrix/ctxfix.install_ctx_free (the evaluator env)",
            "seed_derivation": "KeyedRNG('oopsie_full_v1').derive_seed('env', experiment='" + C.SEED_EXPERIMENT + "', state_id=i); "
                               "skips seeds already used by reset_banks_v1 wipe (any pool) or earlier pool entries",
            "fingerprint": "examples/oopsieverse_full/reset_probe.fingerprint (state_sha = sha256(sim state[1:])[:16], dirt_sha over env._marker_positions)",
            "build_order": "P-train g01,g02,g06 then P-devlayout g00,g03, one env; verify: separate process, shuffled with P-random",
            "code_sha256": {f: C.sha256_file(os.path.join(C.HERE, f)) for f in ("build_pools.py", "wv2_common.py")},
        },
        "layouts": lay,
        "pools": pools,
        "P-random": {"source": C.BANK_FILE, "pool": "eval_dev", "n": C.P_RANDOM_N, "selection": "entries 0000-0049 in listed order",
                     "bank_content_sha256": bank_all["content_sha256"], "bank_file_sha256": C.sha256_file(C.BANK_FILE),
                     "reset_ids": [e["reset_id"] for e in be], "reset": "canonical (unset_ep_meta; rng=default_rng(seed); reset)",
                     "note": "the bank's eval_dev holds 100 entries; the spec fixes P-random at 50 -> the first 50"},
        "checks": checks, "per_group": per_group,
        "test_bank_and_test_layouts_touched": False,
    }
    doc["content_sha256"] = C.sha256_json(doc)
    tmp = C.POOLS_FILE + ".tmp"
    json.dump(doc, open(tmp, "w"), indent=1)
    os.replace(tmp, C.POOLS_FILE)
    fsha = C.sha256_file(C.POOLS_FILE)
    open(C.POOLS_FILE + ".sha256", "w").write(f"{fsha}  pools_v1.json\ncontent_sha256 {doc['content_sha256']}\n")
    os.chmod(C.POOLS_FILE, 0o444)
    print("FROZEN", C.POOLS_FILE, "file_sha256", fsha, "content_sha256", doc["content_sha256"])


if __name__ == "__main__":
    {"build": cmd_build, "verify": cmd_verify, "freeze": cmd_freeze}[sys.argv[1]]()
