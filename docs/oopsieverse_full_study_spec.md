# OopsieVerse full study: future-motion representations for safe manipulation

Implementation specification • 5 October 2026

## 1. Authorization, scope and scientific questions

The user has explicitly chosen an expansive OopsieVerse study. This document supersedes the earlier instruction to stop after Stage 0 in `oopsieverse_intent_safety_experiment_spec.md`. Keep the original audit, verdict, hashes and results immutable; record this scope change in a new run directory. Do not overwrite historical evidence or rewrite the earlier verdict as if the new experiments had already happened.

This is a multi-task, multi-training-seed research program, independent of the overnight workshop deadline. Proceed through engineering checks into the full declared comparison; the checks prevent invalid runs, not automatically terminate the study because preliminary evidence is unfavorable. Report task-specific blockers and continue eligible tasks.

Questions:

1. Does conditioning a policy on predicted future motion improve safe task completion at matched training budgets?
2. Does it produce candidates from which safe successful behavior is easier to select?
3. Does preserving temporal order and gripper behavior improve on mean future EEF pose?
4. Do any gains survive safe-only training, data-collection confounds, and safety-definition changes?

Separate these claims. An offline probe is not a policy result; an oracle is not a deployable selector; an improvement in safety accompanied by lost competence is a tradeoff.

This specification uses the user's audit summaries. The writer has not independently inspected the experiment repository, raw datasets, or frozen audit files. Resolve interfaces against the installed source and pinned revision before implementation.

## 2. Existing evidence and required corrections

Preserve `outputs/oopsieverse_audit/STAGE0_VERDICT.md` and its referenced artifacts.

Reported safety counts (not verified safe-completion counts):

| Label | pick_egg safe / damaging | wipe_countertop safe / damaging |
|---|---:|---:|
| Strict zero damage, earlier spec | 23 / 37 | 7 / 53 |
| Paper: every object health > 95 | 29 / 31 | 14 / 46 |
| Code: mean object health >= 95 | 29 / 31 | 28 / 32 |

The strict rule was our experimental choice. Do not describe all three as benchmark defects. Preserve the paper/code distinction, including strict versus inclusive boundary operators and the set of entities included.

Reported pick_egg probes under the strict rule: state 0.744 AUROC, state + mean intent 0.761, state + richer future information 0.907. These are exploratory associations from 60 demos; no simulator replay verified the outcomes. They motivate ablations, not a predicted policy gain. The matched-pair result has a wide interval and does not establish representational equivalence.

The damage `dt` FIXME and possible one-step event offset remain suspected defects. Do not silently patch either. Playback-path portability fixes may be necessary but must be distinguished from physics changes.

## 3. Execution phases and progression

A. Isolated environment, source inventory, replay, metric contract.
B. Dataset catalog, frozen task roster and splits, reproduced baseline.
C. Full policy matrix with three training seeds.
D. Candidate oracle and independently validated simulator selection.
E. Learned selection, full closed-loop evaluation.
F. Confound controls, representation controls, consolidated reporting.

Execute dependent phases in this order; schedule independent task/seed jobs within available resources. No unbounded architecture search. If a check fails, repair the concrete engineering problem or mark the task blocked. Continue other eligible tasks. Failure to beat the baseline is a valid completed result.

Do not require another general go/no-go permission after every phase. Ask only for unavailable resources, destructive actions, or a material expansion beyond this specification. The request for this document does not itself authorize deleting existing experiment outputs or purchasing compute.

## 4. Backend and storage

Inventory installed interpreters, GPU/driver versions, simulator/assets revisions, checkpoints, dataset locations and free space per filesystem. The prior audit reported robocasa dependencies including mujoco==3.3.1, numpy==2.2.5 and robosuite>=1.5.2; verify exact requirements from the pinned release instead of modifying the existing project environment.

Create a dedicated environment and lock resolved dependencies. Preserve the working square/LIBERO environments. Confirm peak installation, extracted assets and run-output space; the previous 25 GB estimate is provisional and training may need more.

Prefer an adequately sized existing scratch volume. Inventory reclaimable package caches and use supported cache-cleaning tools only with the necessary authorization. Never remove `outputs/oopsieverse/`, checkpoints, frozen audits or datasets to make room automatically. If no suitable storage exists, finish configs and report the precise resource blocker.

Smoke-test environment construction, reset, controller stepping, success detection, health updates and rendering. Headless evaluation can proceed without video only if replay and state checks work; record the video limitation.

## 5. Task roster and data eligibility

Catalog every task exposed by the pinned benchmark and identify which actually have usable training data. Broad benchmark coverage does not imply demonstrations exist for all tasks.

Target four eligible tasks, spanning at least two damage mechanisms where available; minimum three for the intended multi-task headline. Start with pick_egg because its data audit exists. Inspect other tasks already present, plus documented demonstration tasks. Include a second simulator backend if it supplies eligible tasks and infrastructure is available; do not claim simulator generality from one backend.

Reopen wipe_countertop as a metric-sensitive diagnostic task. Its finger damage, feedback-dependent layouts and sparse safe subset require validation; do not discard it merely because the earlier spec did. Include it in the primary roster only if the intended damage semantics and demonstrable safe task completion are defensible. Otherwise report it separately with a reason fixed before policy comparisons.

Freeze a roster with:
- Dataset revision/hash, episode IDs and actual counts.
- Task-success rule and available verification evidence.
- Damage mechanism, affected entities and safety labels.
- Safe-successful, damaging-successful and failed/unknown counts.
- Feedback, collector/session, scene/layout and reset-family metadata.
- Backend readiness, assets and missing observations.
- Inclusion, diagnostic-only or blocked status and justification.

Primary policy tasks need both safe and damaging successful examples. Aim for at least ten independent episodes of each; this is a feasibility target, not a significance threshold. Smaller or highly confounded datasets remain exploratory strata. If fewer than three tasks qualify, complete the feasible work and narrow the generality claim; do not manufacture coverage.

## 6. Replay and metric contract

For each candidate task, replay a predefined stratified sample: at least four zero-damage, four mildly damaging and four substantially damaging episodes where available, balanced over feedback conditions. Verify additional episodes if discrepancies appear. Before training, replay all episodes with missing success labels when feasible; otherwise quarantine unknown-success episodes from the primary successful-demo pool.

Use official task success, not an unverified height proxy. Compare stored versus replayed observations, health and outcomes. Record divergence and do not silently replace labels. Fix path portability independently of evaluation semantics.

Freeze `metric_contract.json` before model comparison:

- Primary safe success: official task success AND the paper's per-object safety predicate, resolved from pinned sources.
- Secondary: task success AND strict no health loss beyond a documented numerical tolerance.
- Compatibility: task success AND released code's aggregate-health predicate.
- Raw health losses, affected entity set and time aggregation for all three.
- Task timeout, success termination and any post-success check window.
- Whether safety applies throughout the trajectory or at its end; use the verified paper convention for the primary endpoint and report any distinction.

If source semantics remain ambiguous, record the ambiguity and choose a named operational primary definition before seeing model results. Do not claim it is the official definition.

Report task success, safe success, damaging success, safe failure and damaging failure. Report per-entity losses, worst-entity loss, native health metric, episode duration and maximum contact force. A no-op may avoid damage but must not appear successful.

Reproduce the suspected dt mechanism with unit/controlled simulator checks. Keep released physics primary. If a defect is confirmed, create a separately versioned corrected environment; regenerated labels and retrained policies are a separate comparison. Do not evaluate models across changed physics and attribute the difference to representation.

## 7. Data splits and confounds

Split by complete independent demonstration groups, never time windows. Group duplicate trajectories, reset families and collector/session where possible. Default 70/15/15 train/development/test by group with stratification over safety and feedback where feasible; publish actual counts and exceptions. Fit normalization and encoders only from allowed training data.

Keep evaluation reset banks separate from demonstration splitting. Use 100 development resets for tuning and 200 frozen test reset families per task, disjoint from each other and training collection where controllable.

Main training pool: verified successful mixed safe/unsafe demonstrations. Preserve failed demos in a catalog; training on them is outside the primary experiment unless a separate objective is declared.

Use one fixed mixed-data sampling distribution across all mixed arms. Conduct feedback-stratified analyses. Where both classes exist within each condition, add within-condition training/evaluation and balanced-subset sensitivity. Where overlap is absent, state that the safety effect cannot be separated from collection condition. Weighting cannot create missing behaviors.

On wipe_countertop, identical layouts/base poses must remain grouped; no random split that distributes the same scene family across train and test while claiming layout generalization.

## 8. Policy matrix and precise representations

Shared observations, encoder family, controller, normalizer, action horizon, execution horizon and action-flow solver across arms. Use the existing native observation modality. Log privileged simulator features; expose them equally to compared arms.

| ID | Data | Policy |
|---|---|---|
| A-MIX | Mixed successful | Native stochastic action flow |
| A-SAFE | Safe successful under primary predicate | Same action-flow architecture |
| I-MEAN | Same as A-MIX | Existing mean future EEF pose, conditioning action flow |
| I-PATH | Same as A-MIX | Ordered future EEF poses |
| I-PATH-G | Same as A-MIX | Ordered future EEF poses plus gripper-command profile |
| AUX-PATH-G | Same as A-MIX | Action flow with auxiliary future prediction, no intent conditioning |

Use three training seeds per arm/task. Pair data-order seeds when feasible. A-SAFE uses the same update budget, reporting its smaller unique-data volume and repeat exposure. Additionally evaluate a safe-sized random mixed subset control on representative tasks to separate filtering from dataset size.

Let H equal the native predicted action horizon in physical control steps, fixed across representations. Preserve the legacy I-MEAN target exactly and document quaternion handling. For I-PATH use four future timestamps at approximately H/4, H/2, 3H/4 and H, deduplicated if H is short. Express each pose relative to the current EEF frame, using a continuous rotation representation with verified conversions. Fixed timestamps encode coarse timing. I-PATH-G appends gripper commands at those timestamps.

To isolate ordering from frame/rotation changes, include a canonical mean-pose control using the same frame and rotation encoding as I-PATH on the mechanism subset. Never claim an ordering effect solely from a legacy-mean versus changed-frame comparison.

Predict intent stochastically from current observation history; condition the action model on it. Ground-truth future intent is training supervision only. Confirm interventions on intent affect decoded actions. A co-prediction head alone is not the conditioning method.

Start with the established teacher-forcing/training recipe, identical across intent variants. Log teacher-forced versus sampled-intent validation behavior. If introducing predicted-intent mixing, declare the schedule and apply it uniformly; do not rescue only one arm with extra tuning.

Do not use measured future contact force as a deployable intent target in the primary matrix: it depends on interaction dynamics and could leak outcomes in diagnostics. Force-aware prediction is a separately specified extension.

## 9. Fair training and baseline reproduction

Reproduce native stochastic A-MIX before the full matrix. Verify action semantics, NFE, precision, train/eval preprocessing and checkpoint evaluation against the native evaluator. Never replace the action prior with zero noise for diversity comparisons.

Use fixed final training steps selected from native learning curves on development data, then apply the same budget to all primary arms. Record parameter counts, GPU hours, action NFE and intent NFE. Match backbones where possible; total compute need not be identical, so report it and include inference-cost comparisons.

Allow at most three declared common hyperparameter configurations in development, with equal search allocation across compared families. Choose without test results. Retain all attempted configurations and seeds. A single task's favorable setting must not be retrospectively presented as prespecified.

Competence failures trigger reproduction diagnosis. If a correctly reproduced method still fails, retain its result. Do not exclude a task because intent loses.

## 10. Random-policy evaluation

Evaluate every primary arm on the same 200 test reset families and three rollout sampling seeds per training seed. These are rollout seeds, not independent training runs. Use native stochastic inference for all relevant arms.

No artificial perturbations are required for the main claim. Add a separate valid-reset generalization stratum only if task semantics and physical validity can be audited; never recycle the invalid historical LIBERO perturbation evaluator.

Primary comparison: absolute safe-success difference against A-MIX. Report task-success difference alongside it, including confidence intervals. Also compare with A-SAFE. Predeclare 5 percentage points as a practical safe-success target and 5 points as the maximum acceptable completion loss for the project's decision rule; these are chosen tolerances, not established domain constants.

## 11. Candidate selection: oracle then learned

Run proposal diagnostics for A-MIX, I-MEAN, I-PATH and I-PATH-G. Use 50 development parent resets for debugging/tuning and 100 fresh test parent resets per task/training seed. One pre-damage intermediate state per parent, sampled under a balanced mixture of source policies at a time chosen independently of candidate outcomes. Log source-policy attrition; the resulting estimand is conditional on the eligible state bank.

Restore complete simulator, controller, observation history, RNG and DamageSim state, including health, thermal/fluid accumulators and event latches. Test restore through damaging interactions, branch-order invariance and disk round trips. Equal qpos alone is insufficient.

Generate M=16 candidates; evaluate K=1,4,16 from the same pool. Vary action noise for A-MIX; vary intent noise with fixed decoder noise for primary intent proposal comparisons. Add joint intent/decoder sampling as a labeled sensitivity, since conditional decoding may otherwise reduce proposal diversity.

Execute one native chunk followed by a frozen tail policy to the original timeout. Primary tails are each arm's own policy; add a common A-MIX tail on the mechanism subset. Report handoff behavior.

Four tail replicates per candidate: seeds 0–1 for ranking, 2–3 for independent validation. Keep proposal RNG separate. Oracle availability at K is 1 - C(M-m,K)/C(M,K), where m candidates safely succeed on a given replicate. This is retrospective headroom, not deployment.

For simulator-assisted ranking, use safe-success frequency on ranking seeds with outcome-independent random tie-breaking; assess on held-out seeds. Average K=4 over 128 fixed subsets. Report ties and compute cost.

For learned selection:
- Generate labels on new training parents only; all descendants stay in their parent's split.
- Train the same scorer family on observation history plus proposed physical action chunk for every proposal arm.
- Predict safe-success and task-success separately; choose highest safe-success estimate with a declared task-success constraint tuned only on development parents.
- Match labeling budgets and report environment steps, not only number of branches.
- Use frozen encoders initially and identical scorer capacity/search budgets.
- Evaluate full closed-loop replanning at K=1,4,16 on the frozen test reset bank; no simulator lookahead, future health or ground-truth intent at deployment.
- Run three scorer seeds paired with the three policy seeds; distinguish their roles in reporting.

Branch-trained scorers may see a different state distribution under repeated selection. Audit this shift. Any additional on-policy label collection must have an equal budget across methods and no test-reset reuse.

Compare both absolute safe success and improvement over each arm's own K=1. A weak policy's large improvement alone does not establish superior selection.

## 12. Mechanism and confound controls

Preselect two mechanism tasks by data quality and distinct damage mechanism before seeing policy advantages. Run:

- Canonical mean versus ordered path with identical pose encoding.
- AUX-PATH-G versus conditioned I-PATH-G.
- Safe-only versus matched-size mixed training.
- Common-tail candidate evaluation.
- Feedback-stratified and primary-versus-strict safety analyses.
- A dimension-matched learned latent bottleneck without explicit future-motion labels, if claiming that semantic future supervision improves search beyond generic compression. Fix its encoder/decoder and reconstruction objective in a separate config before launching it; dimension matching alone does not match training objectives.

Re-run offline probes with all safety definitions using identical episode-grouped splits and capacity controls. Report uncertainty and label overlap. Rich future signals remain privileged diagnostics.

If making an RL fine-tuning efficiency claim, add the released damage-aware DSRL baseline with matched environment-step budgets. RL is not required for this BC/selection study; do not claim novelty for applying existing damage-aware RL.

## 13. Statistics and claim boundaries

Publish per-task, per-training-seed estimates. Use equal task weights for macro averages rather than pooling all episodes.

For paired differences, resample shared reset families and training seeds while preserving all arm/candidate/rollout dependencies. A crossed bootstrap over resets and training seeds is appropriate when the same reset bank is shared across seeds. Use 5,000 resamples and show seed-level points; three seeds still give weak training-variance estimates. Keep the finite roster fixed for the main macro interval. Do not infer generality across arbitrary tasks from three or four tasks.

Primary confirmatory family: I-MEAN, I-PATH and I-PATH-G versus A-MIX on random-policy safe success. Use Holm correction if reporting significance across that family. Report effect sizes and intervals regardless. Label selection and mechanism analyses as distinct prespecified secondary families; avoid choosing only favorable tasks/K values.

Noninferiority of completion requires the corresponding interval to rule out the chosen loss margin; a nonsignificant loss is insufficient. Separate learned-selection evidence from simulator-assisted diagnostics.

## 14. Budget, scheduling and recovery

Planning target with four tasks:

- Six primary arms × three training seeds × four tasks = 72 training runs.
- Random evaluation: 6 × 3 × 4 × 200 resets × 3 sampling seeds = 43,200 episodes.
- Test branching: 4 proposal arms × 3 seeds × 4 tasks × 100 parents × 16 candidates × 4 tails = 307,200 continuations.
- Development branching at 50 parents adds 153,600 continuations under the same full matrix.
- Learned selection at K=4 and 16: 4 arms × 3 seeds × 4 tasks × 200 resets × 3 sampling seeds × 2 K values = 57,600 episodes; reuse K=1 random evaluation.
- Scorer-training branches, controls, native reproduction and replay are additional, separately budgeted.

These are workload estimates, not a claim that resources exist. Measure throughput and episode length on a smoke batch; write estimated GPU-hours, CPU-hours, storage and wall time before queueing the full matrix. Set concurrency to available resources. Never launch an unbounded scheduler or conceal the branch cost.

Execution priority: establish all primary policy arms across tasks/seeds, then proposal diagnostics, then learned selection and controls. If compute is insufficient, preserve three-seed primary comparisons and reduce diagnostic parent counts uniformly with a documented protocol amendment before inspecting test outcomes. Do not reduce only unfavorable tasks or arms.

Write atomic per-episode records and resumable jobs keyed by data/config/checkpoint/metric hashes. Preserve crashes and retries. Invalid actions count as policy failures; infrastructure errors are separate with missing-pair counts. Refuse resume across changed physics or metrics.

## 15. Required implementation and artifacts

Adapt these module names to repository conventions:

- inventory_tasks_and_data.py
- verify_playback_and_metrics.py
- build_grouped_splits.py
- train_policy_matrix.py
- snapshot_damage_state.py
- evaluate_policy_matrix.py
- evaluate_candidate_selection.py
- train_safe_success_scorer.py
- analyze_full_study.py

Required records: task, split, reset family, scene/feedback stratum, training/rollout/scorer seed, arm, checkpoint hash, candidate/noise IDs, controller and solver settings, metric version, success, each safety predicate, entity health traces, termination, duration, latency, simulator steps and errors.

Required outputs:

1. Environment lockfile, source/data/asset versions and experiment manifest.
2. Replay report and frozen metric contract.
3. Task roster, inclusion reasons and grouped data/reset splits.
4. Policy configurations and full attempted-run ledger.
5. Restore-through-damage verification.
6. Per-task and macro success/safe-success tables with seed plots.
7. Safe-success versus K and inference cost, oracle separately from learned selection.
8. Feedback, safety-definition and representation ablations.
9. Representative success/failure/damage videos where rendering works.
10. Final report distinguishing supported findings, unresolved mechanisms and blocked coverage.

Do not title the report as a positive intent result before results exist.

## 16. First coding-agent instruction

Read the frozen audit and repository instructions. Create a new full-study directory referencing this scope change. Resolve isolated storage/backend setup, replay pick_egg and the wipe diagnostic subset, catalog every available demonstration task, and write the metric contract plus resource plan. Then continue into the declared multi-task, three-seed policy matrix once engineering validity is established. Preserve the existing audits and outputs. Report concrete blockers without reverting to the superseded Stage-0-only scope.

## Reference entry points

Verify all behavior against pinned local sources; these links identify the project rather than guarantee a particular current API.

- https://robin-lab.cs.utexas.edu/oopsieverse/
- https://github.com/UT-Austin-RobIn/oopsieverse
- https://robin-lab.cs.utexas.edu/oopsieverse/documentation/learning/
- https://arxiv.org/html/2606.31993v1

