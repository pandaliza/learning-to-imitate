"""Analyze the enumerated reset banks + determinism tests and write roster/reset_banks_v1.json.

Run in the oopsie env after ``build_reset_banks.py derive|enumerate|tests``.
"""

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_reset_banks import EXPERIMENT_ID, POOLS, RUN_DIR, TASKS  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "outputs/oopsieverse_full_study/roster")
HF = ("/home/lizad/.cache/huggingface/hub/datasets--ut-robin-lab--oopsieverse-demos/snapshots/"
      "c0ec29395fdcf898de2aed8dc0b3e786dc2a37d3/paper_experiments/teleop_data")
OBJ = {"pick_egg": "egg", "wipe_countertop": "sponge"}
FP_KEYS = ["state_sha", "model_sha_norm", "health_sha", "dirt_sha"]
COINCIDENCE_MM = 5.0
COINCIDENCE_RULE = {
    "pick_egg": "drop a candidate whose (robot base xy, egg xyz) is within 5 mm (Chebyshev) of any demo obs[0]; back-fill from the pool's spares in key order",
    "wipe_countertop": ("drop a candidate whose dirt layout is identical to a demo's AND whose (base xy, sponge xyz) is within 5 mm of "
                        "that demo (the wipe reset family is base pose + dirt layout); back-fill from spares"),
}


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


def yaw_wxyz(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def demo_initial_states(task):
    """obs[0] of each demo. NOTE obs[0] is recorded after actions[0] (teleop), i.e. one control step
    after the reset; actions[0] is a zero arm command in the inspected demos."""
    import h5py
    rows = []
    for cond, fb in (("with", "fb"), ("without", "nofb")):
        with h5py.File(os.path.join(HF, task, f"{cond}_live_feedback.hdf5"), "r") as f:
            for d in sorted(f["data"], key=lambda s: int(s.split("_")[1])):
                g = f["data"][d]
                o = OBJ[task]
                q = g["obs"][f"{o}_quat"][0]  # robosuite obs quats are xyzw
                meta = json.loads(g.attrs["ep_meta"])
                r = dict(episode_id=f"{task}/{fb}/{d}", base_pos=g["obs/robot0_base_pos"][0].tolist(),
                         eef_pos=g["obs/robot0_eef_pos"][0].tolist(), obj_pos=g["obs"][f"{o}_pos"][0].tolist(),
                         obj_yaw=yaw_wxyz([q[3], q[0], q[1], q[2]]), arm_action0_absmax=float(np.abs(g["actions"][0][:6]).max()))
                r["obj_minus_base"] = (np.array(r["obj_pos"]) - np.array(r["base_pos"])).tolist()
                if "dirt_marker_positions" in meta:
                    mp = np.array(meta["dirt_marker_positions"], dtype=np.float64)
                    r.update(dirt_sha=_h(mp), dirt_start=mp[0].tolist(), dirt_centroid=mp.mean(0).tolist())
                rows.append(r)
    return rows


def rng_summary(vals):
    v = np.asarray(vals, dtype=np.float64)
    return dict(min=np.round(v.min(0), 4).tolist(), max=np.round(v.max(0), 4).tolist(),
                span=np.round(v.max(0) - v.min(0), 4).tolist())


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def main():
    seeds = json.load(open(os.path.join(RUN_DIR, "bank_seeds.json")))
    bank = seeds["bank"]
    enum = {}
    errors = []
    for line in open(os.path.join(RUN_DIR, "enumeration.jsonl")):
        r = json.loads(line)
        if "error" in r:
            errors.append(r)
        else:
            enum[r["reset_id"]] = r
    tests = [json.loads(l) for l in open(os.path.join(RUN_DIR, "tests.jsonl"))]
    out = dict(
        status="FROZEN v1", version="reset_banks_v1", created="2026-10-06", experiment_id=EXPERIMENT_ID,
        procedure=dict(
            env=("oopsiebench EnvironmentRegistry damageable_class (pick_egg -> DamageablePickEgg, wipe_countertop -> "
                 "DamageableWipeCounter), robot PandaOmron, default composite controller, control_freq 20, ignore_done=True, "
                 "use_camera_obs=False, no renderer; collection-time continuous gripper "
                 "(damagesim.robosuite.io_utils.patch_gripper_for_position_control() before construction)"),
            canonical_reset=("env.unset_ep_meta(); env.rng = np.random.default_rng(seed); env.reset(); "
                             "obs = env.get_observations()[0]   (examples/oopsieverse_full/reset_probe.py: seeded_reset)"),
            equivalent_fresh_reset="construct the env with seed=seed and call env.reset() once (verified bitwise-equal sim state, T1/T2)",
            obs_refresh_required=("DamageableWipeCounter.reset returns the observation from BEFORE its IK placement and 5 "
                                  "zero-action steps (wipe_counter.py reset() discards reset_robot_above_sponge's return), so "
                                  "the returned proprio/health are stale (EEF ~36 cm off). Evaluators must use "
                                  "env.get_observations() after reset. Same call is harmless for pick_egg."),
            reset_randomness=("released env reset randomness only (metric_contract resolved_decisions.reset_perturbation): robot "
                              "base deviation x +-0.15 m, y +-0.05 m, yaw 0 (kitchen.py:405-407), robot joint noise N(0, 0.02 rad), "
                              "object placement (egg 0.2x0.2 m / sponge 0.1x0.1 m region, yaw +-0.1 rad, pick_egg.py/wipe_counter.py "
                              "_get_obj_cfgs), wipe dirt path start/direction; layout 2 / style 4 fixed"),
            seed_derivation=(f"KeyedRNG('{EXPERIMENT_ID}').derive_seed('env', experiment='{EXPERIMENT_ID}/reset_bank/<task>/<pool>', "
                             "state_id=<index>) (mip/intent_selection/rng.py): blake2b(digest_size=8) of "
                             "'<experiment>|env|state_id=<index>', first 4 bytes little-endian & 0x7FFFFFFF. Collisions within a task "
                             "are skipped (none occurred)."),
            reset_family=("each bank entry is one reset family = one seed = one deterministic initial state. Rollout sampling "
                          "seeds (3 per training seed, spec 10) vary only policy noise on the same initial state."),
        ),
        pools_definition={
            "eval_dev": "100 development resets (spec 7/10): tuning, checkpoint/config selection, competence screens",
            "eval_test": "200 frozen test reset families (spec 7/10): every primary arm x training seed x 3 rollout seeds; learned-selection closed-loop eval",
            "sel_dev_parents": "50 development parent resets (spec 11): proposal/oracle debugging and tuning",
            "sel_test_parents": "100 test parent resets (spec 11) for oracle / simulator-assisted selection",
            "sel_train_parents": "400 parents reserved for learned-selection scorer label generation (spec 11: 'new training parents only'); all descendants stay with their parent",
            "reserve": "300 resets for replacements or later amendments; usable only through a logged amendment, never ad hoc",
        },
        parent_sharing=("SHARED across training seeds: the same parent reset IDs are used for every training seed and arm "
                        "(paired design; enables the crossed reset x seed bootstrap of spec 13). '50 dev / 100 test parents per "
                        "task/training seed' is satisfied by reusing the same IDs for each training seed. What differs per training "
                        "seed is the intermediate pre-damage state drawn from each parent; derive it with KeyedRNG"
                        f"('{EXPERIMENT_ID}').derive_seed('state_collection', experiment='{EXPERIMENT_ID}/selection/<task>/<pool>', "
                        "checkpoint_pair='train_seed_<k>', state_id=<parent index>) for the source-policy choice and branch time, "
                        "and keep proposal ('candidate'/'decoder') and tail ('continuation') RNG in their own namespaces."),
        disjointness_rule="all pools are disjoint per task by seed AND by initial sim state; banks are disjoint from demo initial states (checked below)",
        tasks={},
    )

    # ── determinism tests ──
    fresh = defaultdict(list)
    for r in tests:
        if r["case"] == "fresh" and "error" not in r:
            fresh[(r["task"], r["seed"])].append(r["fp"])
    det = dict(test_errors=[r for r in tests if "error" in r], cases=Counter(r["case"] for r in tests))
    t1 = [all(v[0][k] == v[1][k] for k in FP_KEYS if k in v[0]) for v in fresh.values() if len(v) == 2]
    t1_raw_model = [v[0]["model_sha"] == v[1]["model_sha"] for v in fresh.values() if len(v) == 2]
    det["T1_fresh_vs_fresh_two_processes"] = dict(
        n=len(t1), identical_state_model_health_dirt=sum(t1),
        raw_model_xml_identical=sum(t1_raw_model),
        note="raw model XML differs only in a visual-only counter geom rgba drawn from global np.random and a serialization-only mesh content_type attribute; normalized XML identical")
    # T2 a: enumeration (used env, canonical seeded_reset) vs fresh
    t2 = []
    for (task, s), v in fresh.items():
        match = [e for e in enum.values() if e["task"] == task and e["seed"] == s]
        if match:
            t2.append(dict(task=task, seed=s, worker_reset_index=match[0]["worker_reset_index"],
                           identical=all(match[0][k] == v[0][k] for k in FP_KEYS if k in v[0])))
    det["T2a_fresh_vs_canonical_reset_in_used_worker_env"] = dict(n=len(t2), identical=sum(x["identical"] for x in t2), rows=t2)
    t2b = [r for r in tests if r["case"] == "damaged_then_seeded" and "error" not in r]
    det["T2b_fresh_vs_canonical_reset_after_driving_env_into_damage"] = dict(
        n=len(t2b),
        identical=sum(all(r["fp"][k] == fresh[(r["task"], r["seed"])][0][k] for k in FP_KEYS if k in r["fp"]) for r in t2b),
        health_min_before_reset=[round(r["before_reset"]["health_min"], 2) for r in t2b],
        health_min_after_reset=[round(r["after_reset"]["health_min"], 2) for r in t2b])
    t3 = [r for r in tests if r["case"] == "fresh_then_plain_reset" and "error" not in r]
    det["T3_second_plain_reset_on_same_env"] = dict(
        n=len(t3),
        first_equals_fresh=sum(r["fp_first"]["state_sha"] == fresh[(r["task"], r["seed"])][0]["state_sha"] for r in t3),
        second_equals_first=sum(r["fp_second"]["state_sha"] == r["fp_first"]["state_sha"] for r in t3),
        base_xy_shift_m=[round(float(np.linalg.norm(np.array(r["fp_second"]["base_pos"][:2]) - np.array(r["fp_first"]["base_pos"][:2]))), 4) for r in t3],
        conclusion=("a repeated env.reset() without reseeding is a NEW draw from the advanced env.rng (different base pose, "
                    "object placement, dirt layout), not a repeat of the seed's reset; it is a valid random reset but not the bank entry"))
    t4 = [r for r in tests if r["case"] == "sticky_ep_meta" and "error" not in r]
    det["T4_sticky_ep_meta"] = dict(
        n=len(t4),
        base_pose_kept_from_previous_episode=sum(r["fp_sticky"]["base_pos"] == r["fp_first"]["base_pos"] for r in t4),
        dirt_kept_from_previous_episode=sum(r["fp_sticky"].get("dirt_sha") == r["fp_first"].get("dirt_sha") for r in t4 if "dirt_sha" in r["fp_first"]),
        object_placement_follows_new_seed=sum(r["fp_sticky"]["obj_pos"] == r["fp_clean"]["obj_pos"] for r in t4),
        sticky_equals_clean_state=sum(r["fp_sticky"]["state_sha"] == r["fp_clean"]["state_sha"] for r in t4),
        conclusion=("after set_ep_meta(get_ep_meta()) -- what the teleop DamageDataCollectionWrapper.reset, playback and replay do -- "
                    "every later reset reuses the previous base pose (and wipe dirt layout) while re-drawing object placement and "
                    "joint noise. A used env that ever had ep_meta set is therefore NOT a fresh reset; the canonical reset calls "
                    "unset_ep_meta() first. This also explains the demo structure: one base pose per pick_egg condition and one "
                    "base pose + dirt layout per wipe collection session."))
    out["determinism"] = det

    all_ok = True
    for task in TASKS:
        demos = demo_initial_states(task)
        D_base = np.array([d["base_pos"][:2] for d in demos])
        D_obj = np.array([d["obj_pos"] for d in demos])
        D_rel = np.array([d["obj_minus_base"] for d in demos])
        D_eef = np.array([d["eef_pos"] for d in demos])
        D_key = np.concatenate([D_base, D_obj], 1)
        demo_dirt = {d.get("dirt_sha") for d in demos} - {None}
        uniq_base = sorted({tuple(np.round(b, 4)) for b in D_base})
        rel_lo, rel_hi = D_rel.min(0), D_rel.max(0)
        tdict = dict(n_resets_enumerated=0, pools={}, demo_initial_states=dict(
            n=len(demos), unique_base_xy=[list(b) for b in uniq_base],
            base_xy=rng_summary(D_base), obj_pos=rng_summary(D_obj), obj_minus_base=rng_summary(D_rel),
            eef_pos=rng_summary(D_eef), obj_yaw_minus_pi=rng_summary(wrap(np.array([d["obj_yaw"] for d in demos]) - np.pi)[:, None]),
            **({"dirt_start": rng_summary([d["dirt_start"] for d in demos])} if task == "wipe_countertop" else {}),
            n_dirt_layouts=len(demo_dirt) or None,
            note="obs[0] of each demo = one control step after the (unrecorded) reset; actions[0] arm command max |.| = "
                 f"{max(d['arm_action0_absmax'] for d in demos):.3g}"))
        seen_seed, seen_state = {}, {}
        dup_seed, dup_state = [], []
        dropped, pose_near_other_dirt = [], 0
        for pool in POOLS:
            cands = bank[task][pool] + bank[task][pool + "__spares"]
            entries = []
            B, O, R, E, Y, NN, NNrel, inside, dirtd = [], [], [], [], [], [], [], [], []
            init_dmg, succ0 = 0, 0
            for it in cands:
                if len(entries) == POOLS[pool]:
                    break
                fp = enum.get(it["reset_id"])
                if fp is None:
                    all_ok = False
                    print("NOT ENUMERATED", it)
                    continue
                b = np.array(fp["base_pos"][:2]); o = np.array(fp["obj_pos"]); rel = np.array(fp["obj_minus_base"])
                cheb = np.abs(D_key - np.concatenate([b, o])).max(1) * 1000
                j = int(np.argmin(cheb))
                same_dirt = fp.get("dirt_sha") in demo_dirt
                if task == "pick_egg":
                    coincident = cheb[j] <= COINCIDENCE_MM
                else:
                    coincident = bool(same_dirt and cheb[j] <= COINCIDENCE_MM)
                    pose_near_other_dirt += bool(cheb[j] <= COINCIDENCE_MM and not same_dirt)
                if coincident:
                    dropped.append(dict(candidate_id=it["reset_id"], seed=it["seed"], nearest_demo=demos[j]["episode_id"],
                                        chebyshev_mm=round(float(cheb[j]), 2), same_dirt_layout=same_dirt))
                    continue
                if it["seed"] in seen_seed:
                    dup_seed.append((it["reset_id"], seen_seed[it["seed"]]))
                seen_seed[it["seed"]] = it["reset_id"]
                if fp["state_sha"] in seen_state:
                    dup_state.append((it["reset_id"], seen_state[fp["state_sha"]]))
                seen_state[fp["state_sha"]] = it["reset_id"]
                tdict["n_resets_enumerated"] += 1
                nn_rel = float(np.linalg.norm(D_rel - rel, axis=1).min() * 100)
                ins = bool(np.all(rel >= rel_lo) and np.all(rel <= rel_hi))
                B.append(b); O.append(o); R.append(rel); E.append(fp["eef_pos"]); Y.append(fp["obj_yaw"])
                NN.append(float(cheb[j])); NNrel.append(nn_rel); inside.append(ins)
                init_dmg += fp["initial_damage"]; succ0 += fp["success_at_reset"]
                e = dict(reset_id=f"{task}/{pool}/{len(entries):04d}", seed=it["seed"], key_index=it["key_index"],
                         candidate_id=it["reset_id"],
                         state_sha=fp["state_sha"], model_sha_norm=fp["model_sha_norm"],
                         base_xy=[round(x, 4) for x in fp["base_pos"][:2]], obj_pos=[round(x, 4) for x in fp["obj_pos"]],
                         obj_yaw=round(fp["obj_yaw"], 4), obj_minus_base=[round(x, 4) for x in fp["obj_minus_base"]],
                         nearest_demo_chebyshev_mm=round(float(cheb[j]), 1), nearest_demo=demos[j]["episode_id"],
                         nearest_demo_obj_rel_base_cm=round(nn_rel, 2), obj_rel_base_inside_demo_box=ins,
                         initial_damage=fp["initial_damage"], health_min_at_reset=fp["health_min"],
                         success_at_reset=fp["success_at_reset"])
                if task == "wipe_countertop":
                    e.update(dirt_sha=fp["dirt_sha"], dirt_start=[round(x, 4) for x in fp["dirt_start"]])
                    dirtd.append(fp["dirt_start"])
                entries.append(e)
            if len(entries) < POOLS[pool]:
                all_ok = False
            seed_list = [e["seed"] for e in entries]
            pool_out = dict(
                n=len(entries), seeds=seed_list, seeds_sha256=hashlib.sha256(json.dumps(seed_list).encode()).hexdigest(),
                states_sha256=hashlib.sha256(json.dumps([e.get("state_sha") for e in entries]).encode()).hexdigest(),
                n_spares_used=sum(1 for e in entries if "spare" in e["candidate_id"]),
                variation=dict(base_xy=rng_summary(B), obj_pos=rng_summary(O), obj_minus_base=rng_summary(R),
                               eef_pos=rng_summary(E), obj_yaw_minus_pi=rng_summary(wrap(np.array(Y) - np.pi)[:, None]),
                               **({"dirt_start": rng_summary(dirtd)} if dirtd else {})),
                demo_support=dict(
                    frac_obj_rel_base_inside_demo_box=round(float(np.mean(inside)), 3),
                    nearest_demo_obj_rel_base_cm=dict(median=round(float(np.median(NNrel)), 2), p90=round(float(np.percentile(NNrel, 90)), 2), max=round(float(np.max(NNrel)), 2)),
                    nearest_demo_base_xy_cm=dict(median=round(float(np.median([np.linalg.norm(D_base - b, axis=1).min() * 100 for b in B])), 2),
                                                 max=round(float(np.max([np.linalg.norm(D_base - b, axis=1).min() * 100 for b in B])), 2))),
                nearest_demo_chebyshev_mm=dict(min=round(float(np.min(NN)), 1), median=round(float(np.median(NN)), 1)),
                n_initial_damage=init_dmg, n_success_at_reset=succ0,
                entries=entries,
            )
            tdict["pools"][pool] = pool_out
        tdict["checks"] = dict(
            seeds_unique_within_task=not dup_seed, initial_states_unique_within_task=not dup_state,
            pools_disjoint_by_seed_and_state=not dup_seed and not dup_state,
            duplicate_seeds=dup_seed, duplicate_states=dup_state,
            coincidence_rule=COINCIDENCE_RULE[task],
            dropped_as_coincident_with_demo=dropped,
            n_pose_within_5mm_of_demo_but_different_dirt_layout=(pose_near_other_dirt if task == "wipe_countertop" else None),
            all_pools_full=all(p["n"] == POOLS[k] for k, p in tdict["pools"].items()),
            draft_seed_overlap=sorted(set(seen_seed) & (set(range(0, 10000)) | set(range(900000, 901000)))),
        )
        print(task, "kept", tdict["n_resets_enumerated"], "dup seeds", len(dup_seed), "dup states", len(dup_state),
              "dropped", dropped, "pose-near-other-dirt", pose_near_other_dirt)
        for pool, p in tdict["pools"].items():
            print("  ", pool, p["n"], "spares", p["n_spares_used"], "inside", p["demo_support"]["frac_obj_rel_base_inside_demo_box"],
                  "nnrel", p["demo_support"]["nearest_demo_obj_rel_base_cm"], "base", p["demo_support"]["nearest_demo_base_xy_cm"],
                  "initdmg", p["n_initial_damage"], "succ0", p["n_success_at_reset"], "cheb", p["nearest_demo_chebyshev_mm"])
        out["tasks"][task] = tdict
    unresolved = [e for e in errors if e["reset_id"] not in enum]
    out["enumeration_errors"] = dict(
        unresolved=unresolved,
        retried_infrastructure_errors=[dict(reset_id=e["reset_id"], seed=e["seed"], error=e["error"]) for e in errors if e["reset_id"] in enum],
        note=("robosuite CustomMaterial writes textures to the hard-coded /tmp/robosuite_temp_tex/<name>.png "
              "(robosuite/utils/mjcf_utils.py:227) on every wipe model load; concurrent processes can read a half-written "
              "dirt.png ('empty PNG file'). Retried successfully with an identical fingerprint path; evaluators must treat it "
              "as a retryable infrastructure error, never as a policy failure."))
    out["all_checks_pass"] = bool(all_ok and not unresolved and all(
        t["checks"]["pools_disjoint_by_seed_and_state"] and t["checks"]["all_pools_full"] for t in out["tasks"].values()))
    os.makedirs(OUT, exist_ok=True)
    body = json.dumps(out, indent=1)
    out["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    json.dump(out, open(os.path.join(OUT, "reset_banks_v1.json"), "w"), indent=1)
    print("all_checks_pass", out["all_checks_pass"], "errors", len(errors))
    print(json.dumps({k: v for k, v in det.items() if k != "test_errors"}, default=str)[:3000])


if __name__ == "__main__":
    main()
