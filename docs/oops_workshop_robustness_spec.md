# Oops, I Erred: deadline-scoped robustness study

Coding-agent handoff — 5 October 2026

## 1. Objective and scope

Prepare verified experimental evidence for a four-page workshop submission about **future-intent conditioning and spatial robustness in robot policies**.

User-provided submission deadline: **6 October 2026, 06:00 America/New_York**. Treat this as the working deadline; verify the portal's exact cutoff, template, anonymity, reference allowance and archival policy separately. The public workshop page confirms an October 5 date but not the precise cutoff time. Do not automatically submit or contact authors/organizers.

This specification is separate from `intent_selection_experiment_spec.md`. Do not run that specification for this submission. That document concerns oracle selection of sampled candidates on square-mh-state. This document concerns standard end-to-end evaluations of existing pi-0.5 baseline and intent-conditioned checkpoints on LIBERO-Goal under initial object-position perturbations.

The research question is:

> Does conditioning a policy on predicted future representations improve success under spatial shifts relative to a matched policy without intent conditioning?

The answer is not predetermined. Existing notes suggest limited or inconsistent improvement; recover the evidence and report its actual direction and uncertainty. Do not optimize the experiment to obtain a negative result.

This work concerns execution failures under shifted configurations. It does not test learning from imperfect demonstrations, recovery after mid-episode interventions, or whether failure data fixes a policy. Do not make those claims.

No repository or raw rollout logs were available to the author of this spec. Inspect the actual code, checkpoint metadata and logs. Follow repository instructions. The project PDF is a map to existing experiments, not an authoritative source of implementation details or publication-ready counts.

## 2. Priorities

**P0 — required:** recover one coherent baseline/intent checkpoint pair; audit evaluation and physical validity; export matched per-episode outcomes; produce one table, one perturbation plot, and an evidence report.

**P1 — only if cheap:** fill missing cells with matched evaluations; collect a small systematic sample of failure videos; inspect results by task.

**P2 — defer:** new training, slot-design ablations, auxiliary-loss sweeps, new encoders, DSRL, oracle intent selection, new benchmarks, and the known-buggy RoboCasa campaign.

Do not spend tonight rebuilding a general evaluation framework. Extend the working native evaluator with minimal changes and verify those changes.

## 3. Recover and choose the comparison

Search the repository and run directories using rg, configs, manifests and logged artifact references. Produce `checkpoint_inventory.csv` with:

- Run ID, checkpoint path/hash, training step, training seed if available.
- Base checkpoint identity: distinguish pi05_base from pi05_libero.
- Dataset/version/task suite, demonstration count and split.
- Trainable/frozen modules, LoRA settings, action expert configuration, training budget.
- Intent representation/teacher, whether frozen or co-trained, conditioning location.
- Observation keys/cameras/history, action representation/normalization and horizons.
- Evaluator code/config/version, reset IDs, policy seeds, task names and success semantics.
- Available raw results, videos and configs; missing information.

Promising search leads from the source notes:

- Pages 63–70 and 72–79: pi05_base-derived control and frozen/decoupled intent comparisons, roughly 94–96% unperturbed success, followed by XY perturbations.
- Pages 60–62: an earlier pi05_libero-derived family; do not mix this with the pi05_base family.
- Pages 91–94: a separate co-prediction family; not the default scope tonight.

Run labels M0/M1/M3 change meaning across the notes. Identify runs by actual configuration and hash, never shorthand alone. The percentages above are retrieval clues, not expected outputs to reproduce by tuning.

Choose the pair with the cleanest provenance and comparable training/evaluation, before looking for the most attractive robustness gap. If training budgets, backbone or observation inputs differ materially, either find a better pair or label the comparison as confounded; do not attribute the difference solely to intent.

Preserve the documented original checkpoint choice. Do not select the best checkpoint on shifted test outcomes. Existing legitimate multiple training seeds can be included without new training; report them individually.

## 4. Audit the native evaluation

Check that both models use the correct inference configuration, observation normalization, action conversion, camera order, instruction, execution horizon, control frequency, timeout, flow schedule and success predicate.

Future images/EEF data must not be passed to the intent model at evaluation. It must predict intent from information available to the deployed policy. Use eval mode and the native sampling distribution. Do not inject oracle intent or tune sampling temperature.

Run a minimal parity check on two unperturbed resets per arm: the wrapper used here must match the native evaluator for matched randomness. A few rollouts do not need to match historical aggregate SR exactly; unexplained action/trajectory discrepancies do block the campaign.

At delta=0, the perturbation setup must reproduce the original configuration and success semantics. Check that the code does not accidentally reposition a target, rewrite a goal, advance extra physics, or reset robot state differently in one arm.

Record official success separately from environment `done`, timeouts and exceptions. If native evaluation defines success as ever achieved during the rollout, use and document that definition consistently. Do not quietly change success semantics to improve the paper.

## 5. Perturbation protocol: recover first, rerun only if necessary

Prefer the existing radial XY perturbation experiment if it is coherent and auditable. Recover exactly which object is shifted in every task, the coordinate frame, delta units, directions, reset states, settling procedure, and whether robot/object velocities are restored.

For a new/fill-in evaluation, use a predeclared task manifest. Choose tasks by the ability to define a valid object translation and restore the goal correctly, not by model performance. Start with the existing eligible task set, apparently up to eight tasks in the notes; resolve actual names and eligibility from code. Do not assume integer task indices are stable across datasets.

Each manifest row contains:

    task_name, instruction, manipulated_body, moved_subtree,
    source_region, destination_region, fixed_goal_geometry,
    perturb_frame, eligibility_reason, known_scene_constraints

For an eligible task, apply before policy execution:

    p_xy_shifted = p_xy_original + delta * [cos(theta), sin(theta)]

Preserve z and orientation unless the recovered original protocol specifies otherwise. Update free-joint/model state correctly for the installed simulator; use the native forward/update mechanism. Do not change collision geometry, goal regions, or destination positions inadvertently. If a task's success goal refers to the moved object's original location, inspect that logic explicitly.

Default new grid, only if reruns are needed:

- delta in {0, 0.05, 0.10} metres.
- Ten equally spaced angles for each nonzero delta, fixed before evaluation.
- Up to three distinct benchmark reset IDs per task for the first coherent rerun.
- One native policy seed per physical case per arm; preserve pairing of physical cases. Additional policy seeds are secondary if time permits.

At delta=0, evaluate each reset only once. Ten identical zero-displacement angles are not ten distinct initial states. If historical runs repeat delta=0 with different policy seeds, retain them as stochastic repeats but cluster them by the same reset family.

For eight tasks and three resets each, this grid has 8*3*(1+10+10)=504 physical cases, or at most **1,008 rollouts across two arms**, before rejection of invalid configurations. This is a ceiling, not an instruction to rerun everything. Reuse compatible verified data first.

Include delta=0.02, 0.15 or 0.20 from existing trustworthy logs as supplementary points if appropriate. Do not spend the deadline evaluating extreme shifts dominated by invalid placements. Never hide their rejection rates.

These initial-position changes test robustness to spatial shifts. Call them formally out-of-distribution only if you can substantiate the relationship to the training distribution. Do not copy the note claiming zero x-variance without checking training states.

## 6. Physical and task validity is a required audit

The notes show cells with nonzero overall SR but zero SR on the purported valid subset. This is a warning to investigate masks, denominators, task weighting and reset semantics. It is not automatically a mathematical contradiction and it must not be copied without explanation.

Determine validity **before running either policy**, using identical physical cases for both. Do not use policy outcomes, failure to reach, or video appearance after execution to declare an initial configuration invalid.

Check and record:

- Object support and whether it is off-table, embedded or interpenetrating another solid body. Distinguish normal support contact from invalid penetration; use documented geometry/contact tolerances.
- Stability after the same short, native-compatible settling procedure for both arms; record actual post-settle positions and drift.
- Whether the shift leaves the task already successful.
- Whether the instruction, target identity and success predicate still describe the intended task.
- Scene-specific constraints such as objects that must begin inside a drawer or on a particular support.
- Whether a shift crosses a documented hard workspace limit. Do not equate a policy's inability to reach with physical infeasibility; unknown reachability remains unknown.

Prefer existing benchmark validity rules where available. Any added thresholds are task-specific engineering choices and must be logged, reviewed on a small sample, and frozen before comparing outcomes.

Use statuses `valid`, `invalid`, and `unknown`, with reason codes and evidence. Unknown cases are reported separately; do not silently count them as valid or invalid. A simulator exception is separate from physical invalidity.

Store a state hash and, where available, the full initialized state for every physical case. Both policies must start from that same state, including controller reset and observation history. No mid-rollout branch-restoration system is needed here.

For existing logs, if validity cannot be reconstructed from saved states or deterministic reset/perturbation metadata, label those results unverified. Rerun a smaller auditable subset if feasible. Do not infer physical validity from a final success label.

Required cross-tab per task/delta:

    attempted_cases, valid_cases, invalid_cases, unknown_cases,
    valid_successes_A, valid_successes_I,
    invalid_successes_A, invalid_successes_I,
    unknown_successes_A, unknown_successes_I,
    infra_errors_A, infra_errors_I

Inspect surprising successes on invalid configurations for immediate goal satisfaction, fallen objects, incorrect object identity or reset errors. These cases can illustrate an evaluation pitfall only after the mechanism is verified.

## 7. Matched evaluation and missingness

Use a case ID from task name, benchmark reset ID, perturbation vector, configuration version and initialization state hash. Both arms must evaluate the same case list. Matching policy-seed integers ensures reproducibility, not identical semantic model randomness.

Write outcomes atomically and support resume. Include checkpoint hash, evaluator commit/config hash and physical case ID in the result key. Never reuse records from incompatible configurations.

Retry infrastructure failures deterministically once. Report unresolved ones and coverage by arm. Do not count an infrastructure error as task failure or silently remove only the worse arm's trial. Analyze the common completed physical-case set and report any excluded/missing pairs. Nonfinite/illegal policy output is a policy failure, separately counted.

If historical arms use different physical cases, they are unpaired. Do not fabricate pairs using row order; either analyze them explicitly as unmatched exploratory results or rerun matched cases.

Do not pool new and old runs unless checkpoint, evaluator, initialization and all relevant semantics match. Otherwise show separate labeled cohorts.

## 8. Metrics and statistical reporting

Primary outcome: success among **predeclared valid physical cases**, per task and perturbation magnitude, with both arms evaluated on the same cases.

For task t and delta d:

    SR[a,t,d] = successes[a,t,d,valid] / completed_matched_valid_trials[t,d]
    gap[t,d] = SR[intent,t,d] - SR[baseline,t,d]

Report valid counts and raw numerator/denominator for every cell. Empty cells are NA, never 0%. Separate physical-case counts from repeated policy-seed trial counts.

Main curve: equal-weight task macro-average over a fixed task set with valid observations at every displayed delta. Print that task set. If no adequate common set exists, make per-task panels the main result and avoid an aggregate curve with changing task composition. Show micro-averages only as secondary, with the weighting labeled.

Report both absolute performance and each policy's change from delta=0. A weaker initial policy can have a smaller degradation without being more robust in absolute success. Different valid subsets across deltas also change evaluation composition; do not interpret the curve as a causal within-case deterioration without that caveat.

Useful small table:

    model | training seed(s) | SR(delta=0) | SR(0.05) | SR(0.10)

Include denominators in the caption or a companion table. A separate native full-suite clean SR table is permissible but must not share a column with a subset result as though they used the same task set.

Uncertainty:

- With sufficient reset families, use a paired cluster bootstrap resampling reset IDs within each task, preserving directions, policy repeats, arms and delta values. Recompute macro-averages for 2,000 resamples and report 95% intervals. Handle resamples with empty cells transparently; never substitute zero.
- These intervals are conditional on the evaluated task set and checkpoint(s); they do not represent training-seed variability.
- With only one or a few distinct resets per task, meaningful reset-generalization uncertainty is limited. Report raw counts and label the study exploratory. Do not bootstrap directions as independent reset families to manufacture narrow intervals. Add more independent resets only if time allows.
- With multiple actual training seeds, report seed-level outcomes. Do not call rollout seeds training seeds.

Do not claim equivalence or statistical proof of no effect from an interval crossing zero. Use wording such as 'we did not observe a consistent improvement in these tested configurations,' with scope and uncertainty. If effects differ by task, show that rather than hiding them in a mean.

## 9. Failure examples: optional and honest

The old no_reach/no_grasp/no_place classifier was flagged as possibly wrong. Exclude its aggregate claims unless independently repaired and validated.

For a cheap qualitative audit, choose up to 12 matched physical cases using a deterministic rule spanning tasks and nonzero deltas. Review both arms, including a mixture of successes and failures. If sampling specifically from failures, label the sample as failure-conditioned and do not estimate overall prevalence from it.

Manual tags may include approach miss, contact without stable grasp, loss during transport, placement error, timeout, and unclear. Use 'unclear' instead of guessing. For a moving target, calculate distance to its time-varying pose. A fixed distance threshold or an object-height change alone does not establish grasp.

Export two or three representative paired image sequences with task, delta, initial validity, outcome and a short literal description. Selected examples illustrate behavior; they do not establish dominant failure modes. Avoid claiming recovery unless recovery behavior is actually evaluated.

## 10. Figures and paper-ready evidence

Export clean PDF/SVG plots plus PNG previews and the exact data tables used to generate them.

Required:

1. Main paired success-versus-delta plot, baseline and intent, valid-case counts and clear aggregation/uncertainty captions.
2. Compact performance table in CSV and LaTeX, with exact denominators and training-seed count.
3. Validity/rejection breakdown by delta, a small figure or table. Keep detailed reasons in supporting material if four-page space is tight.
4. `methods_facts.md`: exact base checkpoint, training data/budget, modules trained, intent definition/conditioning, task set, perturbation geometry, success criteria, validity rules and seeds.
5. `evidence_report.md`: verified findings, conflicting results, limitations, missing data, and which candidate claims are supported or unsupported.

Optional: paired failure montage. A method diagram can be created from verified architecture facts, but do not invent a new method or diagram every historical variant.

All numbers in paper-ready outputs must trace to a run/config/checkpoint and saved episode records. No interpolated measurements, reconstructed seed variance, inferred sample counts, or copied notebook percentages without provenance. A valid historical aggregate can be reported as such if its provenance is verified, but cannot support per-episode analyses it does not contain.

## 11. Suggested minimal implementation and configuration

Adapt these names to the existing repository:

    audit_existing_runs.py      # inventory, cohort selection, consistency checks
    export_episode_records.py  # native log parsing and provenance
    eval_spatial_shifts.py     # minimal extension to existing working evaluator
    analyze_robustness.py       # paired counts, fixed-task aggregation, plots/tables

Minimum episode schema:

    cohort_id, run_id, checkpoint_hash, training_seed,
    evaluator_hash, task_name, reset_id, case_id, initialized_state_hash,
    delta_m, angle_rad, original_xyz, initialized_xyz, moved_body,
    validity_status, validity_reason, validity_version,
    arm, policy_seed, success, success_step, steps, timeout,
    policy_failure, infrastructure_error, video_path, source_log_path

```yaml
experiment: oops_workshop_spatial_robustness_v1
deadline_local: '2026-10-06T06:00:00-04:00'  # user-provided
timezone: America/New_York
mode: audit_then_reuse_then_minimal_fill
training_allowed: false
suite: libero_goal
checkpoint_baseline: RESOLVE
checkpoint_intent: RESOLVE
task_manifest: RESOLVE_AND_FREEZE
delta_m: [0.0, 0.05, 0.10]  # rerun default; preserve recovered protocol otherwise
nonzero_angles: 10
reset_ids_per_task: 3       # exploratory minimum, not independent training seeds
policy_seeds_per_case: 1
same_initialized_states_for_both_arms: true
validity_determined_before_policy: true
primary_population: valid_matched_cases
aggregation: fixed_task_macro_average
bootstrap_unit: reset_family_within_task
max_new_rollouts: 1008      # for eight eligible tasks; prefer fewer via data reuse
stop_new_jobs_local: '2026-10-06T01:00:00-04:00'
freeze_evidence_local: '2026-10-06T02:00:00-04:00'
gpu_budget: RESOLVE_FROM_AVAILABLE_ALLOCATION
resume: true
```

The stop times are proposed writing buffers, not promises about run speed. Resolve compute authorization/allocation from the user/session. Do not cancel unrelated jobs or exceed resources. Do not ask for permission again if the relevant compute use is already authorized.

## 12. Execution order and deadline discipline

1. **First 45–60 minutes:** locate the cleanest pair, audit evaluator/validity, export existing outcomes, report what is usable. If there is no defensible pair, say so immediately.
2. **Next:** run a tiny sanity batch, then estimate throughput including queue delay, model loading, two-policy memory needs and video overhead. Prefer loading each policy once per batch over repeated initialization.
3. **Reuse first:** generate provisional tables/figures from verified existing evidence immediately so writing can proceed. Label provisional counts; update from finalized evidence later.
4. **Fill only essential gaps:** launch complete predeclared task/reset blocks predicted to finish before the stop time. Interleave or shard paired arms so an interruption does not leave an entire arm missing. Keep task coverage balanced. Do not prioritize favorable cells.
5. **By 02:00 local, proposed:** freeze evidence and export final paper-ready assets. Record incomplete experiments as not included; make no statements about their eventual outcomes. Finish writing and validation with a submission buffer.

If only a smaller coherent subset finishes, narrow the paper's scope and disclose how it was chosen. If validity or checkpoint provenance remains unresolved, provide a blocker report rather than a confident negative-result table.

## 13. Acceptance criteria

The handoff is complete when:

- One matched, interpretable comparison has traceable configurations and checkpoints.
- Physical validity, task semantics, success and denominators have been checked.
- Both arms have auditable coverage on the reported cases; missingness is explicit.
- Every figure/table is regenerated from exported data, with correct task weighting.
- The findings are written with appropriate uncertainty and single-seed limitations.
- Known-buggy campaigns and unverified failure labels are not used as evidence.

Return one of: `READY_FOR_DRAFT`, `READY_WITH_EXPLICIT_LIMITATIONS`, or `BLOCKED`, with the reasons and exact artifact paths. The goal is a credible short empirical workshop paper, not a rushed demonstration of a predetermined claim.
