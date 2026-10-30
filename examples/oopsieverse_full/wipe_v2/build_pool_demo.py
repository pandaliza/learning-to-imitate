"""P-demo pool (spec addendum 2026-10-26): the FULL initial state of each of the 40 splits_v1 TRAIN demonstrations.

Reset = rollout_eval.apply_reset(env, {seed, ep_meta, model_xml, state}) exactly as wipe_diag/openloop_replay.py and
official playback restore a demo: set_ep_meta(ep_meta) -> reset -> reset_from_xml_string(model_file) -> sim.reset ->
hide markers -> set_state(states[0]) -> forward -> sync damage velocities (asset paths remapped by path_remap only),
FOLLOWED BY an observable refresh (env._update_observables(force=True); env.get_observations()): after set_state,
robosuite returns CACHED observables from the preceding reset (the known stale playback obs[0], roster
alignment_audit.md), so without the refresh the policy's first frame would not show the restored state. The refresh
recomputes sensors / renders only (sim state unchanged, checked). After the restore: normal physics, policy actions.

  build   one evaluator env (rollout_eval.make_env, use_camera_obs=False, ctxfix): restore every demo, compare with the
          demo's recorded first observation (teleop obs[0] <-> states[0]: robot0_eef_pos / eef_quat / joint_pos /
          gripper_qpos / base_pos / sponge_pos / sponge_quat / robot0_proprio / health), the sim state vs states[0], the
          dirt marker positions vs ep_meta, and the counter-label alias (counter pixels with alias and frozen LUTs);
          then a second, shuffled restore pass of all 40 (determinism across env history) -> pools/demo_build.json
  freeze  checks -> pools/pools_demo_v1.json (+ .sha256); refuses to overwrite. pools_v1.json is not touched.
The pool stores pointers (teleop file, demo) + sha256 of the raw ep_meta, model_file and states[0]; load_demo_resets()
re-reads and re-verifies them.
"""
import hashlib
import json
import os
import random
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wv2_common as C  # noqa: E402
import rollout_eval as RE  # noqa: E402

OUT = os.path.join(C.V2_RUNS, "pools")
POOL_DEMO_FILE = os.path.join(OUT, "pools_demo_v1.json")
CAMS = ["robot0_eye_in_hand", "robot0_agentview_right", "robot0_agentview_left"]
CLASS_TO_ID = {"unknown": 0, "DamageableCounter": 1, "dirt": 2, "robot": 3, "sponge": 4}
OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_joint_pos", "robot0_gripper_qpos", "robot0_base_pos",
            "sponge_pos", "sponge_quat", "robot0_proprio", "health"]
SEED_EXP = "oopsie_full_v1/wipe_v2/pools_demo_v1/P-demo/{group}"


def _sha(b):
    if isinstance(b, str):
        b = b.encode()
    return hashlib.sha256(b).hexdigest()


def demo_entries():
    from build_pools import keyed_rng, bank_wipe_seeds
    KR = keyed_rng()
    used = bank_wipe_seeds()
    used |= {e["seed"] for p in C.load_pools()["pools"].values() for e in p["entries"]}
    ents = []
    for gid in C.TRAIN_GROUPS:
        for k, (cond, demo) in enumerate(C.group_demos(gid)):
            i = k
            while True:
                s = int(KR.derive_seed("env", experiment=SEED_EXP.format(group=gid), state_id=i))
                if s not in used:
                    break
                i += 1000
            used.add(s)
            ents.append({"reset_id": f"wipe_v2/P-demo/{gid}/{cond}__{demo}", "layout_group": gid, "cond": cond,
                         "demo": demo, "seed": s, "teleop_file": C.TELEOP.format(cond=cond)})
    return ents


def load_demo(e, check=None):
    """Reset dict for apply_reset (+ the recorded obs[0]). check: dict of sha256 to verify."""
    from path_remap import remap_asset_paths
    with h5py.File(e["teleop_file"], "r") as f:
        g = f["data"][e["demo"]]
        em_raw = g.attrs["ep_meta"]
        xml_raw = g.attrs["model_file"]
        em_raw = em_raw.decode() if isinstance(em_raw, bytes) else em_raw
        xml_raw = xml_raw.decode() if isinstance(xml_raw, bytes) else xml_raw
        s0 = np.asarray(g["states"][0], np.float64)
        obs0 = {k: np.asarray(g["obs"][k][0], np.float64) for k in OBS_KEYS if k in g["obs"]}
        n = int(g["actions"].shape[0])
    shas = {"ep_meta_raw_sha256": _sha(em_raw), "model_file_raw_sha256": _sha(xml_raw), "state0_sha256": _sha(s0.tobytes())}
    if check:
        for k, v in shas.items():
            assert check[k] == v, (e["reset_id"], k)
    r = {"id": e["reset_id"].replace("/", "__"), "seed": e["seed"], "family": e["reset_id"],
         "ep_meta": json.loads(remap_asset_paths(em_raw)), "model_xml": remap_asset_paths(xml_raw), "state": s0.tolist(),
         "pool": "P-demo", "layout_group": e["layout_group"], "in_support": None, "demo_len": n}
    return r, obs0, shas


def load_demo_resets(pool_doc=None):
    d = pool_doc or json.load(open(POOL_DEMO_FILE))
    body = {k: v for k, v in d.items() if k != "content_sha256"}
    assert C.sha256_json(body) == d["content_sha256"], "pools_demo_v1.json content hash mismatch"
    out = []
    for e in d["entries"]:
        r, _, _ = load_demo(e, check=e)
        r.update(state_sha=e["state_sha"], dirt_sha=e["dirt_sha"], obj_pos=e["obj_pos"])
        out.append(r)
    return out


def apply_reset_refresh(env, reset):
    """rollout_eval.apply_reset + observable refresh for state restores (see header)."""
    out = RE.apply_reset(env, reset)
    if reset.get("state") is None:
        return out
    sb = env.sim.get_state().flatten().copy()
    env._update_observables(force=True)
    out = env.get_observations()
    assert np.array_equal(sb, env.sim.get_state().flatten()), "observable refresh changed the sim state"
    return out


def restore_and_check(env, e):
    from reset_probe import fingerprint
    r, obs0, shas = load_demo(e)
    stale, _ = RE.apply_reset(env, r)
    stale = {k: np.asarray(stale[k], np.float64).copy() for k in ("robot0_eef_pos", "robot0_proprio") if k in stale}
    obs, _ = apply_reset_refresh(env, r)
    fp = fingerprint(env, C.TASK, obs)
    dev = {}
    for k, v in obs0.items():
        if k in obs:
            a = np.asarray(obs[k], np.float64).reshape(-1)
            b = v.reshape(-1)
            if k.endswith("quat") and a.shape == b.shape:
                dev[k] = float(min(np.abs(a - b).max(), np.abs(a + b).max()))  # q and -q are the same rotation
            else:
                dev[k] = float(np.abs(a - b).max()) if a.shape == b.shape else f"shape {a.shape} vs {b.shape}"
    st = env.sim.get_state().flatten()
    dev["stale_without_refresh_eef_pos"] = float(np.abs(stale["robot0_eef_pos"] - obs0["robot0_eef_pos"]).max())
    dev["stale_without_refresh_proprio"] = float(np.abs(stale["robot0_proprio"] - obs0["robot0_proprio"]).max())
    dev["sim_state_vs_states0_excl_time"] = float(np.abs(st[1:] - np.asarray(r["state"])[1:]).max())
    m, d = env.sim.model, env.sim.data
    mk = np.array([d.body_xpos[m.body_name2id(x.root_body)] for x in env.markers])
    dev["dirt_marker_xy_vs_ep_meta"] = float(np.abs(mk[:, :2] - np.asarray(r["ep_meta"]["dirt_marker_positions"])).max())
    names = sorted(set(env.seg_id_to_class.values()))
    px = {}
    for cam in CAMS:
        raw = np.asarray(obs[f"{cam}_segmentation_class"]).astype(np.int64)
        for kind, fn in (("alias", C.alias_seg_lut), ("frozen", C.frozen_seg_lut)):
            lut = fn(CLASS_TO_ID, env.seg_id_to_class)
            px[f"{cam}/{kind}"] = int((lut[np.clip(raw, 0, len(lut) - 1)] == 1).sum())
    luts_equal = bool(np.array_equal(C.alias_seg_lut(CLASS_TO_ID, env.seg_id_to_class), C.frozen_seg_lut(CLASS_TO_ID, env.seg_id_to_class)))
    return {"fp": fp, "dev": dev, "shas": shas, "seg_class_names": names, "counter_px_at_reset": px,
            "alias_lut_equals_frozen_lut": luts_equal, "n_wiped_at_restore": len(env.wiped_markers)}


def cmd_build():
    ents = demo_entries()
    os.chdir(RE.OOPSIE)
    from ctxfix import install_ctx_free
    env = RE.make_env(C.TASK, use_camera_obs=False)
    install_ctx_free(env)
    t0 = time.time()
    for e in ents:
        e.update(restore_and_check(env, e))
        print(e["reset_id"], json.dumps(e["dev"]), e["counter_px_at_reset"]["robot0_eye_in_hand/alias"], f"{time.time() - t0:.0f}s", flush=True)
    order = list(range(len(ents)))
    random.Random(20261026).shuffle(order)
    for i in order:
        x = restore_and_check(env, ents[i])
        ents[i]["second_pass_state_equal"] = x["fp"]["state_sha"] == ents[i]["fp"]["state_sha"]
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "egl": os.environ.get("MUJOCO_EGL_DEVICE_ID"), "entries": ents,
               "wall_s": time.time() - t0}, open(os.path.join(OUT, "demo_build.json"), "w"), indent=1, default=float)
    print("BUILD_DONE", len(ents), "second pass equal", sum(e["second_pass_state_equal"] for e in ents), flush=True)
    os._exit(0)


def cmd_freeze():
    if os.path.exists(POOL_DEMO_FILE):
        sys.exit(f"REFUSE: {POOL_DEMO_FILE} exists (frozen)")
    b = json.load(open(os.path.join(OUT, "demo_build.json")))
    ents = b["entries"]
    keys = sorted({k for e in ents for k in e["dev"]})
    maxdev = {k: max(e["dev"][k] for e in ents if isinstance(e["dev"].get(k), float)) for k in keys}
    checks = {
        "n": len(ents), "per_group": {g: sum(e["layout_group"] == g for e in ents) for g in C.TRAIN_GROUPS},
        "members_equal_splits_v1_train": sorted(f"{e['cond']}/{e['demo']}" for e in ents) ==
                                         sorted(f"{c}/{d}" for g in C.TRAIN_GROUPS for c, d in C.group_demos(g)),
        "max_deviation_vs_recorded_obs0": maxdev,
        "second_pass_state_reproduced": f"{sum(e['second_pass_state_equal'] for e in ents)}/{len(ents)}",
        "counter_visible_alias_eye_in_hand": f"{sum(e['counter_px_at_reset']['robot0_eye_in_hand/alias'] > 0 for e in ents)}/{len(ents)}",
        "alias_lut_equals_frozen_lut": f"{sum(e['alias_lut_equals_frozen_lut'] for e in ents)}/{len(ents)}",
        "markers_wiped_at_restore": max(e["n_wiped_at_restore"] for e in ents),
    }
    ok = (checks["n"] == 40 and checks["per_group"] == {"g01": 5, "g02": 5, "g06": 30} and checks["members_equal_splits_v1_train"]
          and all(e["second_pass_state_equal"] for e in ents) and maxdev["sim_state_vs_states0_excl_time"] == 0.0
          and maxdev["dirt_marker_xy_vs_ep_meta"] < 1e-9 and checks["markers_wiped_at_restore"] == 0
          and maxdev["robot0_eef_pos"] < 1e-3 and maxdev["sponge_pos"] < 1e-3 and maxdev["robot0_joint_pos"] < 1e-3
          and all(e["counter_px_at_reset"]["robot0_eye_in_hand/alias"] > 0 for e in ents))
    print(json.dumps(checks, indent=1))
    if not ok:
        sys.exit("FREEZE REFUSED")
    keep = ("reset_id", "layout_group", "cond", "demo", "seed", "teleop_file")
    doc = {"status": "FROZEN v1", "version": "wipe_v2_pools_demo_v1", "pool": "P-demo", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "declared_before_any_policy_evaluation_on_it": True,
           "spec": "wipe_v2_spec.md, Addendum (2026-10-26): P-demo pool", "spec_sha256": C.sha256_file(os.path.join(C.V2_REPORTS, "wipe_v2_spec.md")),
           "procedure": {"reset": "rollout_eval.apply_reset(env, {seed, ep_meta, model_xml, state=states[0]}) (as wipe_diag/openloop_replay.py) "
                                  "+ observable refresh (env._update_observables(force=True); get_observations()); then normal physics + policy actions",
                         "seed": "KeyedRNG('oopsie_full_v1').derive_seed('env', experiment='" + SEED_EXP + "', state_id=k); feeds env.rng "
                                 "(overridden by the restored state) and the per-episode policy-noise generator",
                         "code_sha256": {f: C.sha256_file(os.path.join(HERE, f)) for f in ("build_pool_demo.py", "wv2_common.py")}},
           "entries": [dict({k: e[k] for k in keep}, **e["shas"], state_sha=e["fp"]["state_sha"], dirt_sha=e["fp"]["dirt_sha"],
                            obj_pos=e["fp"]["obj_pos"], base_pos=e["fp"]["base_pos"], deviation_vs_recorded_obs0=e["dev"])
                       for e in ents],
           "checks": checks}
    doc["content_sha256"] = C.sha256_json(doc)
    json.dump(doc, open(POOL_DEMO_FILE + ".tmp", "w"), indent=1, default=float)
    os.replace(POOL_DEMO_FILE + ".tmp", POOL_DEMO_FILE)
    fsha = C.sha256_file(POOL_DEMO_FILE)
    open(POOL_DEMO_FILE + ".sha256", "w").write(f"{fsha}  pools_demo_v1.json\ncontent_sha256 {doc['content_sha256']}\n")
    os.chmod(POOL_DEMO_FILE, 0o444)
    print("FROZEN", POOL_DEMO_FILE, "file_sha256", fsha, "content_sha256", doc["content_sha256"])


if __name__ == "__main__":
    {"build": cmd_build, "freeze": cmd_freeze}[sys.argv[1]]()
