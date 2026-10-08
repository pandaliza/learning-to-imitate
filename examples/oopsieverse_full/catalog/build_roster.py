"""Build the DRAFT task roster (roster_draft.json / roster_draft.md) from the catalog outputs.

Inputs (in --catalog): task_catalog.json, data_catalog.json, episode_inventory.csv,
episode_inventory_b1k.csv, duplicates.json, splits_draft.json.

Every outcome count here is PRELIMINARY: it comes from stored fields, not from replay.
Status values are DRAFT and must be re-derived after the replay/metric-contract agent
publishes replay-verified success and safety labels.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

TARGET = 10  # spec section 5 feasibility target per class

B1K_READINESS = dict(
    backend="BEHAVIOR-1K / OmniGibson 3.7.1 (Isaac Sim 4.5.0), UT-Austin-RobIn/BEHAVIOR-1K branch proj/safemanibench",
    installed=False,
    branch_head="77d38292b775b15f1a1f99b8a3ee215d3f19c42c (git ls-remote, 2026-10-06)",
    install_steps="conda py3.10 env; torch 2.6.0+cu124; ~26 isaacsim*-4.5.0.0 pip wheels + omniverse_kit 106.5; "
                  "omnigibson + bddl editable; BEHAVIOR datasets (--dataset); telemoma/pyspacemouse/mediapipe; "
                  "then oopsieverse pip install -e .",
    disk_estimate_gb="~50-80 GB (Isaac Sim wheels+extension caches ~15-25 GB, torch ~5 GB, BEHAVIOR-1K assets "
                     "~30-50 GB). ESTIMATE, not measured; /edrive1 has ~9.7 TB free.",
    time_estimate="~1-3 h install + first-launch shader/extension cache warm-up",
    gpu_requirement="Isaac Sim 4.5 targets RTX GPUs (RT cores). This host has 8x NVIDIA H200 (driver 570.133.20), "
                    "which have no RT cores; NVIDIA lists A100/H100-class GPUs as unsupported for Isaac Sim. "
                    "Headless physics-only replay on H200 is UNVERIFIED and may fail at Kit/RTX init. "
                    "Replay of health needs physics contacts (not rendering), but OmniGibson still boots the renderer.",
    licences="setup.sh requires accepting the NVIDIA Isaac Sim EULA and the BEHAVIOR Dataset ToS: a user decision.",
    host="Ubuntu 24.04, glibc 2.39 (setup.sh has a separate path for glibc 2.31-2.33 only)",
    replay_needed_for="health AND success: B1K demo files store only action/state dumps "
                      "(reward all 0, terminated never True, task=DummyTask, no obs/health)",
    version_risk="shelve_item paper demos were recorded at OmniGibson git 77adbd66..., not the branch head 77d38292... "
                 "(add_firewood demos match the head). State-dump replay across commits may diverge.",
)

ROBOCASA_READINESS = dict(
    backend="RoboCasa (robosuite @aaa8b9b, mujoco==3.3.1, numpy==2.2.5 per robocasa/__init__.py)",
    installed="being built by the replay/metric agent at /edrive1/lizad/oopsieverse_full_study/env/oopsie (not touched here)",
    known_issues=["stored model_file/ep_meta asset paths point at collector machines (/Users/arnavbalaji/..., "
                  "/home/arpit/...): playback needs path rewriting (portability fix, not a physics change)",
                  "mechanical dt hard-coded 0.005 s FIXME (suspected; unmeasured)",
                  "possible one-step obs/health/action offset (unresolved)"],
)


def load(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def b(x):
    return x == "True"


def stats(v):
    v = np.asarray(v, float)
    return dict(min=float(v.min()), median=float(np.median(v)), max=float(v.max()))


def rc_task_summary(eps):
    out = {}
    for cond in ("nofb", "fb", "total"):
        e = eps if cond == "total" else [x for x in eps if x["feedback"] == cond]
        out[cond] = dict(
            n=len(e),
            strict_zero_loss=sum(x["health_class"] == "zero" for x in e),
            mild_le5=sum(x["health_class"] == "mild" for x in e),
            substantial_gt5=sum(x["health_class"] == "substantial" for x in e),
            paper_every_object_final_gt95=sum(b(x["paper_all_gt95"]) for x in e),
            code_mean_object_final_ge95=sum(b(x["code_mean_ge95"]) for x in e),
            stored_success_true=sum(x["stored_success_final"] == "True" for x in e),
            stored_success_missing=sum(x["stored_success_final"] in ("", "None") for x in e),
        )
    dmg = Counter()
    for x in eps:
        for ent in filter(None, x["damaged_entities"].split("|")):
            dmg[ent] += 1
    return out, dict(dmg.most_common())


def prelim_outcomes(eps, success_known):
    """safe-successful / damaging-successful / failed-unknown under three predicates."""
    res = {}
    for name, key in (("paper_every_object_final_gt95", "paper_all_gt95"),
                      ("strict_no_loss", "strict_no_loss"),
                      ("code_mean_ge95", "code_mean_ge95")):
        if success_known:
            succ = [x for x in eps if x["stored_success_final"] == "True"]
            res[name] = dict(safe_successful=sum(b(x[key]) for x in succ),
                             damaging_successful=sum(not b(x[key]) for x in succ),
                             failed_or_unknown=len(eps) - len(succ))
        else:
            res[name] = dict(safe_successful="unknown", damaging_successful="unknown",
                             failed_or_unknown=len(eps),
                             if_all_succeed=dict(safe=sum(b(x[key]) for x in eps),
                                                 damaging=sum(not b(x[key]) for x in eps)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, type=Path)
    a = ap.parse_args()
    C = a.catalog
    tasks = {(t["backend"], t["task"]): t for t in json.loads((C / "task_catalog.json").read_text())["tasks"]}
    dcat = json.loads((C / "data_catalog.json").read_text())
    meta, files = dcat["meta"], dcat["files"]
    rc = load(C / "episode_inventory.csv")
    b1k = load(C / "episode_inventory_b1k.csv")
    splits = json.loads((C / "splits_draft.json").read_text())
    dups = json.loads((C / "duplicates.json").read_text())

    roster = []

    def file_recs(task_dir):
        return [dict(path=f["path"], sha256=f["sha256"], n_episodes=f["n_episodes"], condition=f["condition"])
                for f in files if f["source"] == "paper_experiments" and f["path"].split("/")[2] == task_dir]

    def ep_ids(eps):
        return [f"{x['task_dir']}/{x['feedback']}/{x['demo']}" for x in sorted(
            eps, key=lambda x: (x["feedback"], int(x["demo"].split("_")[1])))]

    # ── RoboCasa paper tasks
    for task_dir, reg in (("pick_egg", "pick_egg"), ("wipe_countertop", "wipe_counter")):
        eps = [x for x in rc if x["source"] == "paper_experiments" and x["task_dir"] == task_dir
               and x["feedback"] in ("nofb", "fb")]
        t = tasks[("robocasa", reg)]
        summ, dmg = rc_task_summary(eps)
        success_known = task_dir == "wipe_countertop"
        objk = "egg" if task_dir == "pick_egg" else "sponge"
        P = np.array([json.loads(x["init_obj_pos"])[objk] for x in eps])
        fams = splits[task_dir]["reset_families"]
        md = dict(
            feedback_conditions=dict(Counter(x["feedback"] for x in eps)),
            collector_proxy=dict(Counter(x["collector_proxy"] for x in eps)),
            collector_note="asset-path home dir in ep_meta (machine/user proxy); no session/timestamp/seed metadata stored",
            layout_style=dict(Counter(f"layout {x['layout_id']} / style {x['style_id']}" for x in eps)),
            init_base_pos_by_condition={c: sorted({x["init_base_pos"] for x in eps if x["feedback"] == c})
                                        for c in ("nofb", "fb")},
            init_base_ori=sorted({x["init_base_ori"] for x in eps}),
            reset_families=fams,
            n_dirt_layouts=len({x["dirt_hash"] for x in eps if x["dirt_hash"]}) or None,
            init_object_pos_range_m={ax: float(np.ptp(P[:, i])) for i, ax in enumerate("xyz")},
            duplicate_action_sequences_within_task=0,
            identical_states0=len(eps) - len({x["state0_hash"] for x in eps}),
            near_duplicate_initial_state_pairs_n=len(splits[task_dir]["near_duplicate_pairs"]),
            near_duplicate_rule="object+eef start within 5 mm (wipe: only within the same dirt layout); full list in splits_draft.json",
            T=stats([int(x["T"]) for x in eps]),
            T_by_condition={c: stats([int(x["T"]) for x in eps if x["feedback"] == c]) for c in ("nofb", "fb")},
            action_dim=sorted({x["action_dim"] for x in eps}),
            stored_fields="actions(T,12) dones rewards states(T,170) obs/* incl. obs/health(T,16); info/damage_info, "
                          "info/obs_info" + (", info/{proportion_wiped,total_wiped,total_markers,newly_wiped}"
                                             if task_dir == "wipe_countertop" else "")
                          + "; attrs ep_meta, model_file, health_list_link_names, num_samples. NO task_success field.",
            health_entities="target object + 15 PandaOmron links (16); counter/fixtures not tracked",
        )
        if task_dir == "pick_egg":
            success = dict(
                rule="egg body z - (counter.pos.z + counter.height/2) >= 0.05 m",
                source=t["success_predicate_source"],
                evidence="NOT STORED (rewards all 0, dones all False). Height proxy: all 60 eggs end "
                         f"{min(float(x['proxy_lift_m']) for x in eps):.3f}-{max(float(x['proxy_lift_m']) for x in eps):.3f} m "
                         "above their start (proxy only). Needs replay.",
                stored=False)
            status = "primary-candidate (DRAFT)"
            why = ("Both classes exceed the >=10 target under every safety predicate if replay confirms success "
                   f"(paper >95: {summ['total']['paper_every_object_final_gt95']} safe / "
                   f"{60 - summ['total']['paper_every_object_final_gt95']} damaging; strict: "
                   f"{summ['total']['strict_zero_loss']} / {60 - summ['total']['strict_zero_loss']}). Damage is on the "
                   "target object (egg impact/crush), the intended semantics. Caveats: success not stored (replay "
                   "required before freezing); feedback condition strongly confounded with outcome (nofb only "
                   f"{summ['nofb']['paper_every_object_final_gt95']} safe of 30), so within-nofb analyses are exploratory; "
                   "60 not 90 demos (paper Table C.2 lists 90; all_data.hdf5 is the union of the two 30-demo files).")
        else:
            success = dict(
                rule="len(wiped_markers) == len(markers) > 0",
                source=t["success_predicate_source"],
                evidence="Stored progress info/total_wiped == info/total_markers at final step in all 60 "
                         "(same quantity _check_success tests; recording stopped on completion). Marker-wiped "
                         "threshold itself is not stored; replay should confirm.",
                stored=True)
            ps = summ["total"]["paper_every_object_final_gt95"]
            status = "diagnostic (metric-sensitive; conditional on replay + semantics decision)"
            why = (f"Safe count depends on predicate: strict {summ['total']['strict_zero_loss']}, paper >95 {ps}, "
                   f"code mean>=95 {summ['total']['code_mean_object_final_ge95']} of 60. Damage is almost entirely "
                   "quasistatic gripper-finger damage (robot links); the sponge has zero damage sensitivity and the "
                   "counter is untracked (and its released params have zero sensitivity), whereas the official learning "
                   "config lists DamageableCounter/sponge/dirt as objects of interest - so the measured damage is not "
                   "obviously the intended task semantics. Safe class is thin and condition-confounded "
                   f"(paper >95: nofb {summ['nofb']['paper_every_object_final_gt95']}, fb "
                   f"{summ['fb']['paper_every_object_final_gt95']}); nofb is a single reset family (one base pose + "
                   "one dirt layout), so grouped splits leave only 5 paper-safe episodes in train and no nofb in "
                   "dev/test. Promote to primary only if (a) replay confirms >=10 safe successes under the frozen "
                   "primary predicate and (b) robot-finger quasistatic damage is accepted as the intended safety "
                   "semantics in metric_contract.json BEFORE any policy comparison. Otherwise report separately. It "
                   "would add a distinct mechanism (robot quasistatic vs object impact).")
        roster.append(dict(
            task=task_dir, registry_name=reg, backend="robocasa", status=status, justification=why,
            dataset=dict(hf_repo=meta["hf_repo"], revision=meta["hf_revision"], files=file_recs(task_dir),
                         n_episodes=len(eps), episode_ids=ep_ids(eps)),
            success=success,
            damage=dict(mechanisms_config=t["damage_mechanisms"], entities_config=t["entities"],
                        zero_sensitivity_entities=t.get("entities_with_zero_sensitivity"),
                        observed_damaged_entities_episode_counts=dmg),
            stored_health_counts=summ,
            preliminary_outcomes=prelim_outcomes(eps, success_known),
            meets_target_10_each_if_success_confirmed={
                k: (summ["total"][k] >= TARGET and 60 - summ["total"][k] >= TARGET)
                for k in ("strict_zero_loss", "paper_every_object_final_gt95", "code_mean_object_final_ge95")},
            metadata=md, backend_readiness=ROBOCASA_READINESS,
            in_paper=dict(il=t["in_paper_il"], rl=t["in_paper_rl"]),
            splits_ref=f"splits_draft.json#{task_dir}",
        ))

    # ── B1K paper tasks
    for task_dir in ("shelve_item", "add_firewood"):
        eps = [x for x in b1k if x["source"] == "paper_experiments" and x["task_dir"] == task_dir
               and x["feedback"] in ("nofb", "fb")]
        t = tasks[("behavior1k", task_dir)]
        md = dict(
            feedback_conditions=dict(Counter(x["feedback"] for x in eps)),
            collector_proxy="not recorded in B1K files",
            scene_model=sorted({x["scene_model"] for x in eps}), robot=sorted({x["robot_type"] for x in eps}),
            og_version=sorted({x["og_version"] for x in eps}), og_git_hash=sorted({x["og_git_hash"] for x in eps}),
            assets_version=sorted({x["assets_version"] for x in eps}),
            physics_hz=sorted({x["physics_frequency"] for x in eps}), action_hz=sorted({x["action_frequency"] for x in eps}),
            unique_states0=len({x["state0_hash"] for x in eps}),
            reset_family_note=(("states[0] identical in 58/60 episodes: it is the shared init_states/add_firewood.pkl "
                                "snapshot; the per-episode reset noise (base xy +-3 cm, yaw +-0.12 rad, arm +-0.07 rad) "
                                "applied after load is not recoverable without replay")
                               if task_dir == "add_firewood" else
                               "states[0] differs per episode (88 unique of 90); the 2 coinciding pairs are grouped "
                               "as identical resets in splits_draft.json"),
            T=stats([int(x["T"]) for x in eps]),
            T_by_condition={c: stats([int(x["T"]) for x in eps if x["feedback"] == c]) for c in ("nofb", "fb")},
            action_dim=sorted({x["action_dim"] for x in eps}),
            stored_fields="action(T,7) state(T+1,flat OG dump) state_size reward terminated truncated "
                          "init_metadata(empty); attrs num_samples, health_list_link_names, transitions({}); "
                          "file attrs config, scene_file, n_episodes, n_steps",
            health_stored=any(b(x["has_health"]) for x in eps),
            reward_nonzero_episodes=sum(int(x["reward_nonzero"]) > 0 for x in eps),
            terminated_any=sum(b(x["terminated_any"]) for x in eps),
            showcase_copies=[g for g in dups["unexpected_groups"] if any(task_dir in s for s in g)],
        )
        mech = t["damage_mechanisms"]
        why = ("Health and success are not stored (verified: no obs/health, reward never non-zero, terminated never "
               "True, task=DummyTask), so outcome classes are entirely unknown until OmniGibson replay. Data volume "
               f"({len(eps)} episodes, paper IL task) could meet the >=10/10 target. "
               + ("Distinct mechanism: robot thermal damage near the fireplace heat source plus mechanical; only the "
                  "robot is a tracked damageable (DAMAGEABLE_OBJECTS['add_firewood'] = agent). Feedback condition "
                  "changes episode length ~2.5x (nofb median %.0f vs fb %.0f steps)." % (
                      md["T_by_condition"]["nofb"]["median"], md["T_by_condition"]["fb"]["median"])
                  if task_dir == "add_firewood" else
                  "Mechanism: impact/quasistatic damage to fragile bystander objects (wineglass, beer, wine, flour, "
                  "crackers) - multi-object clutter, distinct from single-target egg crush. Recorded at a different "
                  "OmniGibson commit than the pinned branch head."))
        roster.append(dict(
            task=task_dir, registry_name=task_dir, backend="behavior1k",
            status="blocked (backend: OmniGibson not installed; H200 RTX support unverified) - candidate if replay works",
            justification=why,
            dataset=dict(hf_repo=meta["hf_repo"], revision=meta["hf_revision"], files=file_recs(task_dir),
                         n_episodes=len(eps), episode_ids=ep_ids(eps)),
            success=dict(rule=t["success_predicate_source"], stored=False, evidence="needs OmniGibson replay"),
            damage=dict(mechanisms_config=mech, entities_config=t["entities"], scene_hazards=t["scene_hazards"]),
            stored_health_counts="not stored",
            preliminary_outcomes="unknown (no stored health or success)",
            metadata=md, backend_readiness=B1K_READINESS,
            in_paper=dict(il=t["in_paper_il"], rl=t["in_paper_rl"]),
            splits_ref=f"splits_draft.json#{task_dir} (feedback-stratified only)",
        ))

    # ── everything else
    showcase = {}
    for f in files:
        if f["source"] == "paper_experiments":
            continue
        showcase.setdefault((f["backend"], f["task"]), []).append(f)
    paper_dirs = {("robocasa", "pick_egg"), ("robocasa", "wipe_counter"), ("behavior1k", "shelve_item"),
                  ("behavior1k", "add_firewood")}
    others = []
    for key, t in sorted(tasks.items()):
        if key in paper_dirs:
            continue
        fs = showcase.get(key, [])
        n = sum(f["n_episodes"] for f in fs)
        note = ""
        if key == ("behavior1k", "pour_water"):
            note = "Paper IL task (Table C.2: 60 demos) but only 1+1 showcase demos are released. "
        if key == ("robocasa", "shelve_item"):
            note = "Registered in RoboCasa (Panda robot, 7-D action); paper shelve_item demos are B1K. "
        others.append(dict(
            task=key[1], backend=key[0], status="blocked (no usable training data)",
            justification=note + (f"Only {n} showcase demo(s) in {key[0]}/teleop "
                                  f"({', '.join(f['path'].split('/')[-1] + ':' + str(f['n_episodes']) for f in fs)}); "
                                  "far below the >=10 per class target. Usable as replay smoke tests." if fs else
                                  "No demonstrations released."),
            damage_mechanisms=t["damage_mechanisms"], success_predicate_source=t["success_predicate_source"],
            showcase_files=[dict(path=f["path"], n=f["n_episodes"], health_stored=f["has_health"],
                                 health_classes=f.get("health_class_counts"),
                                 stored_success=f.get("stored_success_final_counts")) for f in fs],
            in_paper=dict(il=t["in_paper_il"], rl=t["in_paper_rl"]),
        ))

    out = dict(
        status="DRAFT - not frozen. Outcome labels are stored-field (pre-replay); success/safety must be "
               "replay-verified by the replay/metric-contract agent before the roster is frozen.",
        hf_repo=meta["hf_repo"], hf_revision=meta["hf_revision"], pinned_source_commit=meta["pinned_source_commit"],
        health_classes=dict(strict="max per-entity loss <= 1e-6", mild="0 < loss <= 5", substantial="loss > 5",
                            note="computed over all 16 tracked entities (object + robot links)"),
        feasibility_target=f">= {TARGET} independent safe-successful and damaging-successful episodes",
        candidate_tasks=roster, other_tasks=others,
        summary=dict(
            primary_candidates=[r["task"] for r in roster if r["status"].startswith("primary")],
            diagnostic=[r["task"] for r in roster if r["status"].startswith("diagnostic")],
            blocked_backend=[r["task"] for r in roster if r["status"].startswith("blocked")],
            blocked_no_data=[f"{o['backend']}:{o['task']}" for o in others],
        ),
    )
    (C / "roster_draft.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
