"""Write outputs/oopsieverse_full_study/b1k/metric_contract_b1k_amendment_draft.json.

DRAFT for coordinator review. Does NOT touch metric_contract.json (frozen v1.0).
Replay-derived numbers are read from replay_labels_<task>.csv when present.
"""
import csv
import json
import math
import os

OUT = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/b1k"
INV = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/catalog/episode_inventory_b1k.csv"


def demo_lengths(task):
    seen, T = set(), []
    for r in csv.DictReader(open(INV)):
        if r["task_dir"] != task or "paper_experiments" not in r["file"] or "all_data" in r["file"]:
            continue
        key = (r["feedback"], r["demo"])
        if key not in seen:
            seen.add(key)
            T.append(int(r["T"]))
    T.sort()
    return T


def label_stats(task):
    p = os.path.join(OUT, f"replay_labels_{task}.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    tail = [int(r["T_recorded_health"]) - 1 - int(r["first_success_step"]) for r in rows if r["first_success_step"] != ""]
    return {
        "n": len(rows),
        "n_success_final": sum(r["official_success"] == "True" for r in rows),
        "n_success_ever": sum(r["official_success_ever"] == "True" for r in rows),
        "post_success_tail_steps_min_median_max": [min(tail), sorted(tail)[len(tail) // 2], max(tail)] if tail else None,
    }


def class_counts(task):
    p = os.path.join(OUT, f"replay_labels_{task}.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    res = {}
    for c in ["nofb", "fb", "all"]:
        rs = [r for r in rows if c == "all" or r["cond"] == c]
        succ = sum(r["official_success"] == "True" for r in rs)
        d = {"n": len(rs), "success": succ, "fail": len(rs) - succ}
        for name, k, rule in [("primary", "primary_safe_success", "paper_rule_all_obj_gt95"),
                              ("secondary", "secondary_safe_success", "strict_no_loss"),
                              ("compat", "compat_safe_success", "code_rule_mean_ge95")]:
            ss = sum(r[k] == "True" for r in rs)
            sf = sum(r["official_success"] != "True" and r[rule] == "True" for r in rs)
            d[name] = {"safe_success": ss, "damaging_success": succ - ss, "safe_failure": sf, "damaging_failure": len(rs) - succ - sf}
        res[c] = d
    return res


def rollout_cost(task, T_typ, T_max):
    p = os.path.join(OUT, f"bench_rollout_{task}.json")
    b = json.load(open(p))
    sps = b["steps_per_s"]
    rst = sorted(b["reset_s"])[len(b["reset_s"]) // 2]
    per_typ = rst + T_typ / sps
    per_max = rst + T_max / sps
    return {"measured": {"env_build_s": round(b["t_env_build_s"], 1), "reset_s": [round(x, 1) for x in b["reset_s"]],
                         "env_steps_per_s": round(sps, 2), "step_median_s": round(b["step_median_s"], 3),
                         "conditions": "1 process pinned to 8 cores, H200 shared, host load avg 140-270, no cameras, random small actions, damage evaluators on; policy inference time excluded"},
            "process_hours_per_100_episodes": {
                "typical (median demo length + 20-step post-success window = %d steps)" % T_typ: round((b["t_env_build_s"] + 100 * per_typ) / 3600, 2),
                "worst case (every episode runs to the timeout = %d steps)" % T_max: round((b["t_env_build_s"] + 100 * per_max) / 3600, 2)},
            "tag": VR}


def timeout(T):
    return math.ceil(1.5 * max(T) / 50) * 50


fw_T, sh_T = demo_lengths("add_firewood"), demo_lengths("shelve_item")
fw_L, sh_L = label_stats("add_firewood"), label_stats("shelve_item")
VS, VR, OC = "verified-source", "verified-replay", "operational-choice"
SRC = "oopsieverse a739f99 (/edrive1/lizad/oopsieverse_full_study/src/oopsieverse)"
OG = "BEHAVIOR-1K proj/safemanibench 77d38292 (OmniGibson 3.7.1)"

SHIM_EVIDENCE = {
    "verdict": "PASS, recorded in b1k/shelve_shim_validation.md before full labelling; criterion 3 (no one-step wineglass collapse) not met literally: every collapse coincides with a recorded impact of a toppling glass, and released wineglass params (threshold 15, scale 100) make any >=16 N impact fatal in one step. Coordinator may overrule.",
    "episodes": "shelve_item fb demo_0, demo_1, nofb demo_0, demo_1 (official state-forced playback + shim, cameras off)",
    "evidence": {
        "AG_constraint_warnings": "0 / 1,384 steps (unshimmed: one per step, every step)",
        "state_size": "exact 3192 after shim on all loads (2157 + 2007 shim applications, 0 skipped); runtime robot non_kin == ['Temperature'], robot block 91, AG block 15",
        "one_step_divergence": "per-episode median object position error 1.1-3.5 mm (add_firewood 6-9 mm); max joint error 0.065-0.075; >2 cm only for the wineglass during 3-6 consecutive steps per episode, which coincide with the recorded wineglass free-fall (stored states: 3-4.5 m/s, z 1.34 -> 0.04 m); at 4 m/s one 1/30 s step = 13 cm, so 30 Hz vs recorded 120 Hz integration differs by cm",
        "wineglass": "100 -> 0 in one damage step in 3/4 episodes; in all 3 the STORED states show the wineglass already mid-fall (z ~1.09-1.11 vs 1.23-1.24 resting, 0.4 m from the hand) at the damage step, with contacts up to 181 N (impact 16-42 N, qs up to 381 N); in fb demo_1 the stored wineglass never falls and its replay health stays 100. The single-step loss is the damage model's response to a recorded knock-off, not a replay artifact (pre-shim the cause was the dropped grasp).",
        "success": "fb demo_1 succeeds (first success 355/403); the other 3 end with box_of_crackers inside the stand but the end effector 0.16-0.23 m from the stand root (< 0.25 m gripper_far clause, shelve_item.py:313); success iff distance > 0.25 m in all 4",
        "robot_damage": "quasistatic contact on panda links (fb demo_1: 3 consecutive steps, final 55.4; nofb: panda_hand 85.8 / 95.6)",
    },
    "caveat": "coordinator criterion 'wineglass no longer drops to 0 in one step' is not met literally; evidence above shows the drop is a recorded event. Official success likely undercounts operator-completed shelving because recordings end with the hand < 0.25 m from the stand (collection-time predicate may have differed; not verifiable).",
    "tag": VR,
}

draft = {
    "status": "DRAFT for coordinator review; amends oopsie-metric-v1.0 for the BEHAVIOR-1K backend; metric_contract.json NOT modified",
    "created": "2026-10-06",
    "tags": "verified-source = read at cited file:line of pinned clones; verified-replay = reproduced by official playback in this study; operational-choice = our named decision, not official",
    "pins": {
        "oopsieverse": {"value": SRC, "tag": VS},
        "behavior1k": {"value": OG + "; recorded commits shelve_item 77adbd66 (HEAD~2, diff = data_wrapper.py glue only), add_firewood 77d38292", "tag": VS},
        "isaacsim": {"value": "4.5.0.0 (Kit 106.5.0.162521), headless RTX on H200 via OMNIGIBSON_GPU_ID (never CUDA_VISIBLE_DEVICES). Works on GPUs 1, 4, 6, 7; Kit renderer start fails reproducibly with VkResult ERROR_DEVICE_LOST + segfault on GPU 0 (2 attempts incl. empty-scene probe, 2026-10-06 21:01/21:03) and GPU 2 (06:35); no ECC/remap errors reported", "tag": VR},
        "assets": {"value": "behavior-1k-assets 3.7.2rc1 sha256 3bd7cdc1...5e8b; HF behavior-1k/zipped-datasets@9f0d57d4", "tag": VS},
        "dataset": {"value": "ut-robin-lab/oopsieverse-demos@c0ec2939 paper_experiments/teleop_data/{shelve_item,add_firewood}/{with,without}_live_feedback.hdf5", "tag": VS},
    },
    "task_success": {
        "add_firewood": {
            "predicate": {"value": "xy distance(target_object root, fireplace root) < 0.25 m AND robot.is_grasping(target_object) == FALSE; evaluated on current state (not latched)", "tag": VS,
                          "cite": "oopsiebench/envs/behavior1k/add_firewood.py:141-160 (gripper_far check commented out at :158)"},
            "replay": {"value": fw_L, "tag": VR},
            "class_counts_by_feedback": {"value": class_counts("add_firewood"), "tag": VR},
        },
        "shelve_item": {
            "predicate": {"value": "box_of_crackers Inside(stand) AND robot not grasping box_of_crackers AND gripper_far_from_object(robot, stand): ||eef - stand root|| > 0.25 m; current state, not latched", "tag": VS,
                          "cite": "oopsiebench/envs/behavior1k/shelve_item.py:305-316; spatial_checks.py:23,57-75; OG object_states.Inside"},
            "replay": {"value": sh_L, "tag": VR},
            "class_counts_by_feedback": {"value": class_counts("shelve_item") or "PENDING: full 90-episode labelling blocked under Amendment 1 (GPU 0 only; Isaac cannot start on GPU 0). Interrupted 18:17 run set aside as UNVERIFIED.", "tag": VR},
            "recorded_wineglass_falls": {"value": "stored states (no replay): wineglass ends on floor in 2/45 fb vs 18/45 nofb demos (b1k/shelve_wineglass_recorded_motion.json); expected replay wineglass-loss set", "tag": VS + " (stored data)"},
            "layout_shim_validation": SHIM_EVIDENCE,
        },
        "label_source": {"value": "per-step task_completion written by OGDamageableDataPlaybackWrapper._parse_step_data (damageable_env.py:1059-1075) via evaluate_task_completion (misc_utils.py:157-165; exceptions -> False, so check logs for 'task_completion check failed'); final-step aggregation read_final_task_completion (misc_utils.py:206-213)", "tag": VS},
    },
    "entity_set": {
        "add_firewood": {"value": ["franka0 (DamageableFrankaPanda, 12 links; mechanical+thermal+electrical evaluators)"], "tag": VS + " + " + VR,
                         "cite": "damage_params.py:75-77 DAMAGEABLE_OBJECTS['add_firewood'] = categories ['agent']; add_firewood.py:231 target_objects_health=[ROBOT_NAME]; replay health_list_link_names = 12 franka links",
                         "explicit_statement": "For add_firewood the robot is the ONLY tracked damageable entity. Logs/fireplace are not damageable; all damage observed in replay is robot thermal damage (heating_threshold 45 C, damage_params.py:198-201) plus occasional quasistatic contact. Safety predicates therefore reduce to robot-health predicates."},
        "shelve_item": {"value": ["franka0 (12 links)", "box_of_crackers", "book (category bag_of_flour)", "bottle_of_wine", "wineglass", "bottle_of_beer"], "tag": VS,
                        "cite": "damage_params.py:67-69; shelve_item.py:165-181 target_objects_health; 'stand' and scene furniture are not tracked (default skip_categories damage_params.py:62-65 apply only to scene objects)"},
        "entity_rule": {"value": "entity health = min over its tracked links; same as v1.0", "tag": VS, "cite": "scripts/playback_b1k.py extract_health_from_hdf5 (min across links)"},
    },
    "safety_predicates": {
        "primary": {"value": "official final-step success AND every tracked entity final health > 95", "tag": "paper definition (v1.0) + " + OC + " (B1K entity sets above)"},
        "secondary": {"value": "official success AND no tracked link loses > 1e-3", "tag": OC},
        "compatibility": {"value": "official success AND mean over task_cfg.target_objects_health of final entity health >= 95 (SAFE_ENV_HEALTH_THRESHOLD)", "tag": VS, "cite": "scripts/playback_b1k.py run_compute_metrics; damagesim/utils/misc_utils.py:203,216-219"},
        "note_add_firewood": {"value": "with a single entity, primary and compatibility differ only at exactly 95 (> vs >=)", "tag": VS},
    },
    "physics_and_controller": {
        "physics_rate": {
            "pinned_value": {"value": "30 Hz physics, 30 Hz action, 30 Hz rendering (TaskConfig defaults; neither task overrides)", "tag": VS, "cite": "oopsiebench/envs/behavior1k/base.py:26-28; scripts/teleop_b1k.py:151-160"},
            "recorded_value": {"value": "paper HDF5 data.attrs['config'].env.physics_frequency = 120 (action 30)", "tag": VS + " (stored data)"},
            "official_playback": {"value": "forces physics 30 Hz when include_contacts=True (default)", "tag": VS, "cite": "damagesim/omnigibson/damageable_env.py:592-599"},
            "implications": {"value": [
                "Demos were collected with 4 physics substeps per action (120 Hz); policy rollouts in the pinned env use 1 substep (30 Hz). Contact dynamics, impact estimates and grasp stability may differ between demo collection and evaluation.",
                "Replay labels (state-forced; one 30 Hz step from each stored state) and policy-evaluation labels are both computed at 30 Hz, so labels and evaluation share physics; the demo-collection-time health (never stored) is not reproduced.",
                "One-step divergence of the 30 Hz post-step state from the next stored 120 Hz state is mm-scale for add_firewood (see replay divergence columns), so per-step state forcing keeps the replay on the recorded trajectory.",
                "Decision: keep the pinned/released 30 Hz as primary (same principle as v1.0 dt decision). A 120 Hz variant would be a separately versioned environment."],
                "tag": OC},
        },
        "robot_and_action_space": {"value": "FrankaPanda 'franka0', grasping_mode='assisted', action_normalize=False; 7-D action = arm_0 InverseKinematicsController (default mode pose_delta_ori, 6-D dx,dy,dz,dax,day,daz, command_input_limits=None) + gripper_0 MultiFingerGripperController (1-D, input limits [0,1], mode 'smooth')", "tag": VS,
                                   "cite": "shelve_item.py:120-145; add_firewood.py:186-210; OmniGibson controllers/ik_controller.py:24,59; stored action shape (T,7)"},
        "state_layout_shim_shelve_item": {"value": "shelve_item recordings lack the robot Temperature object-state (stored 3191 vs pinned 3192 floats; recorded scene_file franka0 non_kin = []). Portability shim inserts Temperature = 23.0 (OmniGibson object_states/temperature.py:13 DEFAULT_TEMPERATURE, set on init :25) immediately before the robot's 15-float assisted-grasp block; exact-size load, no truncation fallback. Without it the official loader silently mis-decodes the AG block (grasp dropped every step).", "tag": VS + " + " + VR,
                                          "cite": "examples/oopsieverse_full/b1k/replay_b1k.py (_shim_fix/_enable_shim); evidence in b1k_phaseAB_report.md"},
    },
    "episode_protocol": {
        "timeout_steps": {"value": {"add_firewood": timeout(fw_T), "shelve_item": timeout(sh_T)}, "tag": OC,
                          "rule": "ceil(1.5 * longest demo / 50) * 50 (v1.0 rule)",
                          "demo_lengths": {"add_firewood": {"n": len(fw_T), "min": fw_T[0], "median": fw_T[len(fw_T) // 2], "max": fw_T[-1]},
                                           "shelve_item": {"n": len(sh_T), "min": sh_T[0], "median": sh_T[len(sh_T) // 2], "max": sh_T[-1]}},
                          "official": "none; teleop_b1k.py has no step limit"},
        "success_termination": {"value": "terminate at first control step where the official predicate is true, then post-success window", "tag": OC},
        "post_success_window_steps": {"value": 20, "tag": OC + " (anchored to source)",
                                      "rule": "execute the policy 20 more control steps (0.67 s at 30 Hz) after first success; task success = predicate true at the last executed step; safety over the whole executed trajectory including the window",
                                      "cite": "teleop_b1k.py:918-925 steps 20 extra actions after task_completion_check succeeds during collection",
                                      "note": "add_firewood replay: first success == last recorded frame in 60/60 demos, i.e. the released demos contain NO post-success tail (the 20 extra steps are not in the HDF5s)",
                                      "observation": {"value": "post-success tail measured in replayed demos (first success -> last recorded frame), see task_success.*.replay.post_success_tail_steps_min_median_max", "tag": VR}},
        "damage_termination": {"value": "none", "tag": VS},
        "rollout_cost": {"add_firewood": rollout_cost("add_firewood", fw_T[len(fw_T) // 2] + 20, timeout(fw_T)),
                         "shelve_item": rollout_cost("shelve_item", sh_T[len(sh_T) // 2] + 20, timeout(sh_T))},
        "health_baseline": {"value": "official playback skips the first init_skip_steps=4 steps before recording health (damageable_env.py:787-797, 'playback artifacts' at load); rollouts should likewise reset damage tracking after reset() (teleop_b1k.py reset_env -> env._reset_damage_tracking())", "tag": VS},
    },
    "known_issues": [
        {"issue": "official --skip_save_images still renders and saves the recorded config's 2x720p external cameras (~7 MB/step)", "status": "--no_external_cameras wrapper option; outputs bit-identical", "tag": VR},
        {"issue": "add_firewood thermal: all 12 robot links share a single temperature and degrade together", "status": "kept (released semantics)", "tag": VR},
        {"issue": "evaluate_task_completion swallows exceptions as False", "status": "logs grepped for 'task_completion check failed' (0 occurrences in 60 add_firewood + 4 shelve validation episodes)", "tag": VS + " + " + VR},
        {"issue": "shelve_item official success likely undercounts operator-completed shelving: recordings stop with the hand 0.16-0.23 m from the stand (< 0.25 m gripper_far clause)", "status": "official predicate kept as primary; report success_ever and box-on-stand as diagnostics", "tag": VR},
        {"issue": "add_firewood safety fully confounded with feedback condition (primary safe/damaging 25/5 fb vs 0/30 nofb); only damageable entity is the robot", "status": "diagnostic only (protocol_v1 Amendment 1)", "tag": VR},
        {"issue": "GPU confinement: OMNIGIBSON_GPU_ID alone (-> --/renderer/activeGpu, --/physics/cudaDevice, multiGpu off) does NOT confine Isaac; Iray/MDL and CUDA interop open ~256 MB contexts on every visible GPU. All B1K runs before 2026-10-06 21:00 UTC touched GPUs 1-7 this way.", "status": "CUDA_VISIBLE_DEVICES=0 + OMNIGIBSON_GPU_ID=0 + --/renderer/multiGpu/maxGpuCount=1 confines to GPU 0 (guarded probe: every nvidia-smi sample on bus 18:00.0 only; Vulkan GPU 0 'Active', 7 masked GPUs skipped)", "tag": VR},
        {"issue": "Isaac Sim RTX cannot start on GPU 0 (or GPU 2): VkResult ERROR_DEVICE_LOST ~30 s after launch, segfault; reproduced 3x on GPU 0 (shelve replay, empty-scene probe, confined probe) and 1x on GPU 2; each crash coincides with a driver Inforom BBX flush (GPU0 21:05:43.809, GPU2 06:35:31.157), i.e. a driver-logged GPU error event; healthy GPUs last flushed days ago. Same two boards give corrupt MuJoCo EGL frames.", "status": "B1K BLOCKED under Amendment 1 (GPU 0 only). Needs authorization of one of GPUs 1/3/4/5/6/7 (1,4,6,7 verified working) with CUDA_VISIBLE_DEVICES confinement.", "tag": VR},
        {"issue": "CPU oversubscription: unpinned Isaac processes use ~18-22 cores each; pinning to 8 cores was 2.3x faster with bit-identical outputs", "status": "all runs pinned (taskset)", "tag": VR},
    ],
}
with open(os.path.join(OUT, "metric_contract_b1k_amendment_draft.json"), "w") as f:
    json.dump(draft, f, indent=1)
print(json.dumps(draft["episode_protocol"]["timeout_steps"], indent=1))
