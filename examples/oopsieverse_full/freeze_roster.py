"""Frozen task roster v1 (spec section 5): roster/roster_v1.json + roster_v1.md.

Inputs (read-only): catalog/roster_draft.json, catalog/episode_inventory.csv, replay/replay_labels.csv,
metric_contract.json (FROZEN v1.0), env/env_manifest.json, roster/splits_v1.json, roster/reset_banks_v1.json.
Pure JSON/CSV; no simulator.
"""

import csv
import hashlib
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "outputs/oopsieverse_full_study")
CAT = os.path.join(OUT, "catalog")
ROS = os.path.join(OUT, "roster")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def outcome_counts(rows, col, success_col):
    res = {}
    for fb in ("fb", "nofb", "total"):
        rr = rows if fb == "total" else [r for r in rows if r["fb"] == fb]
        succ = [r for r in rr if r[success_col]]
        safe = sum(r[col] for r in succ)
        res[fb] = dict(n=len(rr), safe_success=safe, damaging_success=len(succ) - safe,
                       failed_or_unknown=len(rr) - len(succ))
    return res


def main():
    contract = json.load(open(os.path.join(OUT, "metric_contract.json")))
    assert contract["status"] == "FROZEN", contract["status"]
    draft = json.load(open(os.path.join(CAT, "roster_draft.json")))
    splits = json.load(open(os.path.join(ROS, "splits_v1.json")))
    banks = json.load(open(os.path.join(ROS, "reset_banks_v1.json")))
    manifest = json.load(open(os.path.join(OUT, "env/env_manifest.json")))
    inv = {f"{r['task_dir']}/{r['feedback']}/{r['demo']}": r for r in csv.DictReader(open(os.path.join(CAT, "episode_inventory.csv")))
           if r["source"] == "paper_experiments" and r["feedback"] in ("fb", "nofb")}
    labels = []
    for r in csv.DictReader(open(os.path.join(OUT, "replay/replay_labels.csv"))):
        fb = {"with": "fb", "without": "nofb"}[r["cond"]]
        eid = f"{r['task']}/{fb}/{r['demo']}"
        labels.append(dict(
            episode_id=eid, task=r["task"], fb=fb, T=int(r["T"]),
            success=r["official_success"] == "True",
            success_h010=(r["stored_success_h010_final"] == "True") if r["task"] == "pick_egg" else r["official_success"] == "True",
            primary=r["primary_safe_success"] == "True", secondary=r["secondary_safe_success"] == "True",
            compat=r["compat_safe_success"] == "True",
            worst_entity_loss=float(r["worst_entity_loss"]), first_damage_step=r["first_damage_step"] or None,
            max_contact_force_N=float(r["max_contact_force_N"]) if r["max_contact_force_N"] else None,
            final_egg_height_m=float(r["stored_egg_height_final_m"]) if r["stored_egg_height_final_m"] else None,
            replay_vs_stored_final_health_absdiff=float(r["replay_vs_stored_final_health_absdiff"]),
            label_changed_vs_stored=dict(primary=r["stored_paper_rule"] != r["paper_rule_all_obj_gt95"],
                                         compat=r["stored_code_rule"] != r["code_rule_mean_ge95"],
                                         strict=r["stored_strict"] != r["strict_no_loss"]),
        ))
    rd = contract["resolved_decisions"]
    cand = {c["task"]: c for c in draft["candidate_tasks"]}
    files = {}
    for t in ("pick_egg", "wipe_countertop"):
        files[t] = cand[t]["dataset"]["files"]

    def robocasa_entry(task, status, justification):
        rows = [r for r in labels if r["task"] == task]
        split_of = {m: s for s, v in splits["tasks"][task]["splits"].items() for m in v["members"]}
        group_of = {m: g["group_id"] for g in splits["tasks"][task]["groups"] for m in g["members"]}
        c = cand[task]
        preds = {
            "primary_paper_all_entities_gt95": outcome_counts(rows, "primary", "success"),
            "secondary_strict_no_loss": outcome_counts(rows, "secondary", "success"),
            "compatibility_code_mean_ge95": outcome_counts(rows, "compat", "success"),
        }
        sens = None
        if task == "pick_egg":
            sens = {
                "success_threshold_0.10m": {
                    "primary_paper_all_entities_gt95": outcome_counts(rows, "primary", "success_h010"),
                    "secondary_strict_no_loss": outcome_counts(rows, "secondary", "success_h010"),
                    "compatibility_code_mean_ge95": outcome_counts(rows, "compat", "success_h010"),
                    "evidence": "final egg height above counter 0.1001-0.1104 m in all 60 demos (stored final state; force replay "
                                "reproduces object state to <=2.5e-4); margin to the 0.10 m threshold is as small as 0.1 mm",
                }}
        meets = {k: all(v["total"][x] >= 10 for x in ("safe_success", "damaging_success")) for k, v in preds.items()}
        meta = dict(c["metadata"])
        meta["collection_session_inference"] = (
            "teleop DamageDataCollectionWrapper.reset re-applies the previous ep_meta (teleop_robocasa.py:92-107), so a whole "
            "collection run shares one base pose (and wipe dirt layout) while object placement is re-drawn; verified in the "
            "simulator (reset_banks_v1.json determinism.T4_sticky_ep_meta). pick_egg: one session per condition; wipe: nofb one "
            "session of 30, fb six sessions of 5.")
        episodes = []
        for r in sorted(rows, key=lambda r: (r["fb"], int(r["episode_id"].rsplit("_", 1)[1]))):
            e = inv[r["episode_id"]]
            episodes.append(dict(
                episode_id=r["episode_id"], file=f"paper_experiments/teleop_data/{task}/{'with' if r['fb']=='fb' else 'without'}_live_feedback.hdf5",
                demo=r["episode_id"].rsplit("/", 1)[1], feedback=r["fb"], T=r["T"], split=split_of[r["episode_id"]],
                group=group_of[r["episode_id"]], init_base_pos=json.loads(e["init_base_pos"]),
                dirt_hash=e["dirt_hash"] or None, actions_sha16=e["actions_hash"], state0_sha16=e["state0_hash"],
                official_success=r["success"], primary_safe_success=r["primary"], secondary_safe_success=r["secondary"],
                compat_safe_success=r["compat"], worst_entity_loss=round(r["worst_entity_loss"], 3),
                first_damage_step=r["first_damage_step"], max_contact_force_N=r["max_contact_force_N"],
                final_egg_height_m=r["final_egg_height_m"],
                replay_label_changed_vs_stored={k: v for k, v in r["label_changed_vs_stored"].items() if v} or None))
        bk = banks["tasks"][task]
        return dict(
            task=task, registry_name=c["registry_name"], backend="robocasa", status=status, justification=justification,
            dataset=dict(hf_repo="ut-robin-lab/oopsieverse-demos", revision="c0ec29395fdcf898de2aed8dc0b3e786dc2a37d3",
                         files=files[task], n_episodes=len(rows),
                         note="all_data.hdf5 is exactly the union of the two condition files (action hashes); the study uses the condition files",
                         episode_ids=[x["episode_id"] for x in episodes]),
            success=dict(
                rule=contract["task_success"]["pick_egg" if task == "pick_egg" else "wipe_counter"]["predicate"]["value"],
                primary_threshold=("0.05 m (pinned code)" if task == "pick_egg" else None),
                sensitivity=("0.10 m (data-collection-era code) reported as a sensitivity row in every table" if task == "pick_egg" else None),
                verification="force (state-forcing) replay with the collection-time gripper, robocasa_check_success on the last step: 60/60 official successes; no quarantine",
                episode_protocol=rd["episode_protocol"]["value"]),
            damage=dict(mechanism=("impact + quasistatic damage to the egg (robot links: quasistatic only)" if task == "pick_egg"
                                   else "quasistatic contact force on robot gripper/finger links only (sponge sensitivities are 0)"),
                        tracked_entities=contract["entity_set"]["pick_egg" if task == "pick_egg" else "wipe_counter"]["value"],
                        observed_damaged_entities=c["damage"].get("observed_damaged_entities_episode_counts"),
                        safety_labels="primary / secondary / compatibility predicates of metric_contract.json v1.0, replay-verified"),
            outcome_counts_replay_verified=preds,
            outcome_counts_sensitivity=sens,
            meets_feasibility_target_10_each=meets,
            metadata=meta,
            backend_readiness=dict(
                status="READY (RoboCasa)",
                env=f"{manifest['venv']} (python {manifest['python']}, mujoco {manifest['packages']['mujoco']}, robosuite {manifest['packages']['robosuite']}, robocasa {manifest['packages']['robocasa']})",
                sources={k: v["sha"] for k, v in manifest["sources"].items() if isinstance(v, dict)},
                assets=dict(path=manifest["assets"]["path"], revision=manifest["assets"]["revision"]),
                verified=["construction/reset/stepping/success/health/EGL rendering smoke test (env/smoke_test.json)",
                          "force replay of all 60 demos reproduces state to ~1e-13 and labels (replay_report.md)",
                          f"seeded resets deterministic and enumerated for all {bk['n_resets_enumerated']} bank seeds (reset_banks_v1.json)"],
                quirks=["EGL rendering needs __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d; GPUs 0 and 2 give corrupt frames (use 1, 3-7)",
                        "replay/eval must apply damagesim.robosuite.io_utils.patch_gripper_for_position_control() (collection-time continuous gripper)",
                        "stored model_file/ep_meta asset paths need the path-only remap (path_remap.py) for playback",
                        "a used env with ep_meta set is not a fresh reset: call unset_ep_meta() and reseed env.rng (reset_banks_v1 procedure)",
                        "teleop obs[t] is post-actions[t]: train on (obs[t], actions[t+1]) (alignment_audit.md)"]
                + (["DamageableWipeCounter.reset() returns a stale pre-IK observation: re-read env.get_observations() after reset"] if task == "wipe_countertop" else []),
                missing_observations=("no camera images are stored in the teleop files (low-dim obs + health only); the official "
                                      "learning configs use segmentation images from re-rendered playback. The study's native "
                                      "observation modality is the stored low-dim state; images would require re-rendering on "
                                      "EGL GPUs 1/3-7. No stored rewards/success for pick_egg (replay-derived)."),
            ),
            splits_ref="roster/splits_v1.json#tasks." + task,
            reset_bank_ref="roster/reset_banks_v1.json#tasks." + task,
            episodes=episodes,
        )

    tasks = [
        robocasa_entry("pick_egg", "primary-candidate (INCLUDED as primary policy task)",
                       "Official success replay-verified 60/60. Both safe and damaging successes reach the >=10 target under all three "
                       "predicates (primary 29/31, secondary 23/37, compatibility 29/31) and under the 0.10 m success sensitivity. "
                       "Damage is the intended task semantics (egg broken by impact or squeeze). Caveats: feedback condition is strongly "
                       "confounded with safety (nofb 4 safe / 26 damaging), so the within-nofb stratum is exploratory; 60 episodes, "
                       "not the paper's 90; one collector proxy; one base pose per condition."),
        robocasa_entry("wipe_countertop", "diagnostic-only (EXCLUDED from the primary macro average)",
                       rd["wipe_countertop_role"]["reason_fixed_before_policy_comparison"]),
    ]
    tasks[1]["diagnostic_reason_verbatim_from_contract"] = rd["wipe_countertop_role"]["reason_fixed_before_policy_comparison"]
    tasks[1]["diagnostic_basis"] = rd["wipe_countertop_role"]["basis"]
    for t in ("shelve_item", "add_firewood"):
        c = cand[t]
        tasks.append(dict(
            task=t, registry_name=c.get("registry_name"), backend="behavior1k",
            status="pending backend test",
            justification=("Paper IL task with released teleop data (" + ("45 fb + 45 nofb" if t == "shelve_item" else "30 fb + 30 nofb") +
                           " episodes) but no stored health or success; labels require OmniGibson replay. BEHAVIOR-1K readiness is "
                           "being tested separately (outputs/oopsieverse_full_study/b1k/). The task can enter the roster only through an "
                           "entry in `amendments`, with replay-verified labels, splits and reset banks built under the same rules, "
                           "before any policy comparison on it."),
            dataset=dict(c["dataset"], n_episodes=c["dataset"].get("n_episodes")),
            success=c.get("success"), damage=c.get("damage"), metadata=c.get("metadata"),
            outcome_counts_replay_verified=None,
            backend_readiness=dict(c.get("backend_readiness", {}), status="PENDING (separate B1K readiness test)"),
            draft_splits_ref=f"catalog/splits_draft.json#{t} (DRAFT, not frozen)",
        ))
    blocked = []
    for o in draft["other_tasks"]:
        blocked.append(dict(task=o["task"], backend=o["backend"], status="blocked (no usable training data)",
                            justification=o["justification"], damage_mechanisms=o.get("damage_mechanisms"),
                            success_predicate_source=o.get("success_predicate_source"),
                            showcase_files=o.get("showcase_files")))
    roster = dict(
        status="FROZEN v1", version="roster_v1", created="2026-10-06",
        frozen_before="any policy training or comparison",
        pins=dict(
            dataset="ut-robin-lab/oopsieverse-demos @ c0ec29395fdcf898de2aed8dc0b3e786dc2a37d3",
            dataset_file_sha256={f["path"]: f["sha256"] for t in ("pick_egg", "wipe_countertop") for f in files[t]},
            sources={k: v["sha"] for k, v in manifest["sources"].items() if isinstance(v, dict)},
            metric_contract=dict(version=contract["version"], status=contract["status"], sha256=sha(os.path.join(OUT, "metric_contract.json"))),
            replay_labels_sha256=sha(os.path.join(OUT, "replay/replay_labels.csv")),
            splits_v1_sha256=sha(os.path.join(ROS, "splits_v1.json")),
            reset_banks_v1_sha256=sha(os.path.join(ROS, "reset_banks_v1.json")),
            env_lockfile="outputs/oopsieverse_full_study/env/uv_lock.txt",
        ),
        status_vocabulary={
            "primary-candidate": "included in the primary policy matrix and macro average",
            "diagnostic-only": "run and reported separately; excluded from the primary macro average; reason fixed before policy comparison",
            "pending backend test": "data exist but labels need a backend not yet validated; may be added only via an amendment",
            "blocked (no usable training data)": "0-2 showcase demos per task; not trainable",
        },
        feasibility=dict(
            target="spec 5: four eligible tasks over >= 2 damage mechanisms, minimum three for the multi-task headline",
            current=("1 primary task (pick_egg, RoboCasa, impact/squeeze), 1 diagnostic (wipe_countertop, robot quasistatic), "
                     "2 pending B1K backend test (shelve_item, add_firewood). Below the 3-task minimum: unless amendments add B1K "
                     "tasks, the study is a single-backend, single-primary-task study and the generality claim must be narrowed "
                     "(spec 5: do not manufacture coverage)."),
        ),
        tasks=tasks,
        blocked_tasks=blocked,
        amendments=[],
        amendment_policy=("Append-only. Each amendment records date, task, reason, replay-verified label counts under the frozen "
                          "metric contract, new splits/reset banks (built with freeze_splits.py / build_reset_banks.py rules), file "
                          "hashes, and confirmation that no policy result on the affected task was inspected. Existing entries are "
                          "never edited; a B1K task becomes eligible only through such an amendment."),
    )
    body = json.dumps(roster, indent=1)
    roster["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    json.dump(roster, open(os.path.join(ROS, "roster_v1.json"), "w"), indent=1)
    print("wrote roster_v1.json")
    for t in tasks:
        print(t["task"], t["status"])
        if t.get("outcome_counts_replay_verified"):
            for k, v in t["outcome_counts_replay_verified"].items():
                print("   ", k, {fb: (x["safe_success"], x["damaging_success"], x["failed_or_unknown"]) for fb, x in v.items()})


if __name__ == "__main__":
    main()
