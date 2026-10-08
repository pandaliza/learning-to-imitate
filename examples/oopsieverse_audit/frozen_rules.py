"""Frozen similarity / stage / metric rules for the safe-vs-damaging comparison (Stage 0B).

Written before any pair or window metric was computed. Inputs seen when writing it: outcome labels
(any_damage, first_damage_step, damaged entity, phase/gripper state at first damage), initial geometry
spread, session/reset-family structure. No safe-vs-damaging motion comparison had been run.
Do not edit after results exist; add a new version instead.
"""

VERSION = "v1"

H = 8  # repo intent_horizon for *_state_intent configs

SIMILARITY = {
    "same_task": True,
    "same_source_split": True,  # collection condition is a confounder; never pair across it
    "pick_egg": {"max_obj_xy_m": 0.03, "max_eef0_m": 0.05},
    "wipe_countertop": {"same_reset_family": True, "max_obj_xy_m": 0.03, "max_eef0_m": 0.05},
    "safe_side": "no damage AND task_success (pick_egg: lift proxy, flagged)",
    "unsafe_side": "any damage AND task_success (pick_egg: lift proxy, flagged)",
    "matching": "one-to-one linear_sum_assignment on cost = d_obj/0.03 + d_eef0/0.05 over eligible edges; "
                "all eligible edges are also reported",
    "uses_post_damage_state": False,
}

STAGE = {
    "anchor_U": "t_U = first_damage_step_U - 1 - H  (future window t+1..t+H ends one step BEFORE first damage)",
    "lead_windows": [0, 8, 16],  # extra steps earlier than t_U
    "phase": "approach if t < grasp_onset (first step with gripper width < 0.075 m) else manipulate",
    "progress": {
        "approach": "eef-object distance (obs <obj>_to_robot0_eef_pos)",
        "manipulate_pick_egg": "object z rise",
        "manipulate_wipe_countertop": "info/proportion_wiped",
    },
    "match_rule": "comparator time = first t in comparator's same phase whose progress reaches the anchor's "
                  "(approach: distance <= anchor distance; manipulate: progress >= anchor progress); "
                  "approach fallback = grasp_onset-1; window must be fully inside episode and, for a damaging "
                  "comparator, end strictly before its own first damage; else excluded and counted",
}

METRICS = [
    ("intent_norm", "L2 between repo intent labels (mean of min-max-normalised eef pos+quat over t+1..t+H), 7-D"),
    ("intent_pos_m", "L2 between mean future eef positions (m)"),
    ("intent_rot_deg", "angle between normalised mean future quaternions (deg)"),
    ("path_shape_mm", "mean_k ||(P_a[k]-mean P_a) - (P_b[k]-mean P_b)||, the part of the path the mean removes (mm)"),
    ("path_abs_mm", "mean_k ||P_a[k]-P_b[k]|| (mm)"),
    ("speed_mm", "mean_k |v_a[k]-v_b[k]|, v = per-step eef displacement norm (mm/step)"),
    ("grip_cmd", "mean_k |a6_a - a6_b| over executed actions t..t+H-1 (commanded gripper, -1 open)"),
    ("grip_width_mm", "mean_k |w_a - w_b| finger opening over t+1..t+H (mm)"),
    ("obj_force_N", "|max_k F_obj,a - max_k F_obj,b|, stored filtered contact force on target object (N)"),
]

NULL = ("For each damaging anchor window U, comparators C = other episodes in the same condition passing the "
        "geometric rule (primary) or any geometry (secondary), stage-matched to U. "
        "AUC_f(U) = P(d_f(U, safe C) > d_f(U, damaging C)), ties 0.5; averaged over anchors with >=1 comparator "
        "of each kind; 95% CI by bootstrap over anchors (2000). AUC~0.5 => feature distance does not separate "
        "outcome at matched stage.")

COUNTEREXAMPLE = ("matched cross-outcome pair whose intent_norm <= median intent_norm of damaging-vs-damaging "
                  "comparisons (same anchors), while at least one of speed_mm, grip_cmd, path_shape_mm, obj_force_N "
                  ">= 75th percentile of the damaging-vs-damaging distribution of that metric")

FIGURES = "per task: 3 matched pairs with lowest geometric cost + up to 2 counterexamples with lowest intent_norm"

PROBE = {
    "label": "y_t = 1 iff first_damage_step in [t+1, t+H]",
    "windows": "all t with full obs history and future window; damaging episodes only t < first_damage_step",
    "features": {
        "obs": "repo obs (object-state, eef pos, eef quat, gripper qpos) at t-1 and t, repo min-max normalised",
        "obs+intent": "obs + repo mean-future-EEF intent (7)",
        "obs+intent+profile": "obs+intent + future speed (H) + future gripper cmd (H) + future path residual (3H)",
    },
    "model": "L2 logistic regression (lambda=1e-2), class-balanced, standardised, 300 full-batch LBFGS steps",
    "cv": "grouped by partition group_id; leave-one-group-out",
    "metrics": "AUROC and balanced accuracy, pooled over held-out folds",
    "status": "privileged diagnostic: future intent/profile never available at deployment",
}

# ERRATA v1.1 (implementation fix, applied after the first run revealed every anchor classified as
# 'manipulate'; no thresholds or metrics changed): the stored gripper starts partly closed (~0.054 m)
# and is opened by the operator in the first steps, so "first step with width < 0.075 m" was always
# step 0. grasp_onset is now the first step with width < 0.075 m AFTER the first step with width
# >= 0.075 m, which is what the 'approach' phase was meant to capture.
ERRATA = [
    "v1.1: grasp_onset = first closing after first fully-open step (was: first step below 0.075 m)",
    "v1.1 additions (diagnostics added, nothing replaced): two-way bootstrap CI (anchors x safe comparator "
    "episodes) alongside the frozen anchor-only CI; a second probe label 'damage in [t+H+1, t+2H]' using only "
    "windows whose intent window is fully pre-damage.",
]
