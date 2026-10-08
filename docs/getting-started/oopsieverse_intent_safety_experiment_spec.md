# OopsieVerse: intent selection for safer successful manipulation

Coding-agent specification · 5 October 2026

## 1. Goal and scope

Test this hypothesis:

> An explicit future-motion representation makes successful, low-damage behavior easier to select than direct action sampling, when both policies learn from the same mixture of safe and unsafe demonstrations.

This is a new, staged research experiment. It is not evidence for the current workshop submission until results actually exist. Do not displace the deadline-scoped spatial-robustness write-up or start a large overnight training campaign automatically.

Use this document as the standalone scope. Reuse verified engineering from `intent_selection_experiment_spec.md` when applicable, but OopsieVerse health state and task semantics need their own audits. Do not run `oops_workshop_robustness_spec.md` as part of this experiment.

No repository, actual OopsieVerse data or checkpoints were inspected when writing this specification. Discover their paths and installed interfaces; follow repository instructions. All budgets below are staged defaults, not claims that compute is available.

First deliverable: a demonstration/representation audit and a baseline-reproduction report. Train or evaluate the intent comparison only if those gates pass.

## 2. Sources and differentiation

Official sources checked:

- Project: https://robin-lab.cs.utexas.edu/oopsieverse/
- Paper: https://arxiv.org/html/2606.31993v1
- Code: https://github.com/UT-Austin-RobIn/oopsieverse
- Learning: https://robin-lab.cs.utexas.edu/oopsieverse/documentation/learning/
- Playback: https://robin-lab.cs.utexas.edu/oopsieverse/documentation/quickstart/

OopsieVerse measures task completion separately from damage. The paper defines safe completion as completion without damage events and already demonstrates damage-guided DSRL. Our proposed distinction is the representation used for candidate selection, not the introduction of damage-aware selection or RL.

The learning docs require playback HDF5s rather than raw teleoperation dumps. Paper and documentation disagree on at least one demonstration count: Pick Egg is listed as 90 episodes in paper Table C.2 and 60 in the learning docs. Inventory the downloaded version and exact episode IDs; do not assume either count.

Live-feedback collection splits are not safety labels: a demonstration collected with feedback can still incur damage. Determine outcomes using recorded/replayed task and health metrics.

## 3. Questions and permissible conclusions

Q1. Within comparable states and stages, do safe and unsafe behaviors differ in future motion represented by our intent labels?

Q2. Do frozen intent-based candidate proposals offer more safe-success headroom than stochastic action proposals at equal candidate budget?

Q3. Can a selector exploit that headroom on unseen initial conditions while preserving task completion?

Stage 0 tests feasibility and representation relevance; Stage 2 tests proposal headroom; Stage 3 tests usable selection. None alone proves training-sample efficiency, real-world safety, cross-task generality, or that future supervision rather than architecture/dimension caused an advantage.

DamageSim is an operational simulation metric. Report safety under its configured damage model; do not present it as a physical safety guarantee.

## 4. Stage 0A: choose one task by evidence and implementation readiness

Start by inventorying already-installed tasks and historical runs. The project notes contain preliminary `open_single_door`, `pick_egg`, `dishes_to_sink`, and `nav_lift_bowl` runs. Their reported percentages are unverified leads, not a basis for selecting a winner.

Prefer the existing working RoboCasa setup if a task has enough demonstrations, meaningful damage signals and reproducible task completion. Inspect `open_single_door` first because prior runs exist, but verify its damage mechanism and usable data. Do not assume it is in the paper's main IL dataset.

If that task fails the data gate, inspect another already-installed task. `pick_egg` and `wipe_counter` have documented paper-training paths. `shelve_item` is conceptually useful for route choice but the paper training configuration uses BEHAVIOR-1K; do not port between simulators or install a second backend just to make a favorable task fit.

Record the choice and reasons before model comparison. Required properties:

1. Both successful safe and successful damaging demonstrations, or reproducible policy rollouts of both types, actually exist.
2. At least some safe/unsafe differences concern motion before damage, rather than only unobserved material properties or controller forces.
3. The baseline can complete the task often enough to separate safety from competence.
4. The environment's success, health update and termination functions are accessible and auditable.

If none meets these requirements, return a blocker report. Do not cycle through every task.

## 5. Stage 0B: data and damage audit

Use official playback scripts or verified existing playback files. Confirm required observation/action/EEF/health fields are present, temporally aligned and correctly normalized. Replay at least two alleged safe and two alleged unsafe episodes. Check that damage events and completion agree with stored labels; record replay mismatches instead of silently relabeling the entire dataset.

Create `demo_inventory.csv`:

    task, simulator_version, dataset_hash, source_split, episode_id,
    length, task_success, any_damage, safe_success,
    first_damage_step, damaged_entities, initial_health, final_health,
    observation_keys, action_schema, playback_version, replay_verified

Partition complete episodes into train/development/test, grouping common reset families, collector/session identity and duplicated trajectories where known. Never split adjacent windows across these partitions. With small datasets, explicitly report class counts; if the held-out set is too small for inference, use it as an exploratory audit rather than pretending it is conclusive.

For the first main comparison, train both arms on the same mixed collection of task-successful demonstrations, retaining both safe and damaging successes. If the intended source also contains failed-task demonstrations, report them and set them aside in the primary protocol; using them changes the training question and should be a separate, labeled experiment.

Inspect safe/unsafe trajectories from similar initial geometry and comparable task stages. Similarity rules must be frozen before examining model outcomes and must not match solely on post-damage states. Report when no comparable pairs exist: collection conditions or initial states may explain label differences.

Extract the existing intent definition exactly: mean future EEF position/quaternion over the configured horizon. Determine whether those mean labels distinguish route/approach differences, or whether similar means hide different speeds, gripper actions, contact forces or paths.

Export physical-space path overlays, speed/gripper profiles and damage timestamps for a small predefined set of paired examples. Show counterexamples where intent labels are similar despite different damage outcomes. Do not infer information loss merely from one scatter plot.

Optional lightweight probe, only if data volume permits: compare prediction of damage in the upcoming horizon from current observations alone versus observations plus future-intent labels, with episode-grouped splits and balanced metrics. This is a privileged-label diagnostic; future intent/health must not become runtime input. Any probe gain may be an association, not causal proof or evidence that an intent predictor can recover it at deployment. No probe is required to launch a feasible pilot.

### Stage 0 decision

- If meaningful safe/unsafe differences are visibly captured by the current intent representation, proceed.
- If differences appear mostly in timing/force that the mean erases, record that the existing representation is poorly motivated here. Do not redesign the method silently. A separate ordered-waypoint or force-aware study would require a new hypothesis and budget.
- If the dataset lacks comparable safe/unsafe behaviors, stop the proposed test on that dataset.

## 6. Stage 1: reproduce competence, then form a fair policy pair

Reproduce the supplied CFM policy pipeline/configuration for the selected task, or evaluate a traceable matching pretrained checkpoint. Check observations, controller mode, action conversion, normalization, sampler prior, NFE, chunk prediction/execution horizon and task timeout. Use the supplied observation modality; do not substitute an impoverished handcrafted state and call it reproduction.

The previous project audit found zero-noise sampling for one action baseline. Here the selection baseline MUST sample the native trained stochastic prior. Log sampling mode and verify distinct noise and action candidates. A zero-noise deployment baseline can be reported separately, never used as the only diversity comparator.

Pilot competence screen: 30 valid initial configurations from development seeds. Report success, safe success and damage. A reasonable operational gate is at least 10 successful episodes and at least 5 successful-damaging episodes, plus verified safe successful demonstrations. These are planning thresholds, not significance tests. If policies almost never complete the task, first resolve reproduction; safety comparisons at that floor are not informative.

Build or reuse a second policy that samples intent and explicitly conditions action prediction on it. An auxiliary/co-prediction head does not automatically provide an intervention interface: verify that clamping/resampling intent changes the actions as intended.

Primary policy pair:

| Arm | Training data | Proposal |
|---|---|---|
| ACTION_MIX | Same mixed safe/unsafe successful episodes | Action chunk from stochastic action flow |
| INTENT_MIX | Exact same episodes/split | Stochastic future-motion intent, then conditioned action chunk |

Match observation access, encoder family, data preprocessing, batch size, number of training updates and checkpoint-selection rule as closely as possible. Record total parameters, training time and NFE rather than claiming perfect architectural matching. Use a fixed final step or a common development-set criterion declared in advance. Never compare 50k versus 190k updates as an isolated intent ablation.

One training seed is acceptable for a feasibility pilot, with conclusions conditional on that pair. Broader claims require additional seeds. Do not silently extend only the weaker arm's training to search for parity.

Later essential practical comparator: ACTION_SAFE, trained on the safe subset using the same declared training protocol. Report its smaller dataset size and repeat exposure. This tests whether sophisticated selection improves over simply filtering unsafe demonstrations. It need not block the initial frozen-policy headroom diagnostic.

## 7. Outcome definitions: prevent 'do nothing' from looking safe

Resolve the installed benchmark's exact event thresholds, health normalization and terminal rules and freeze them in `metric_contract.json`.

Primary outcome:

    safe_success = task_success AND no benchmark-defined damage event

Report alongside it:

- Task-success rate.
- Successful-but-damaging rate.
- Failure-without-damage rate and failure-with-damage rate.
- Native average environment health, with averaging/time conventions documented.
- Per-entity final health loss and maximum health loss as diagnostics; include collateral objects and robot, not only the target.
- Episode length/time-to-success and any safety-triggered termination.

Never claim benefit from lower damage alone if completion collapses. Keep the benchmark evaluation horizon and termination rules identical across arms. Avoid a damage-only selector that prefers inaction. Health aggregation can hide damage to a single fragile object, so retain entity-level logs and any-damage indicators.

Use the official task semantics to handle post-success execution. Do not let one arm stop at the instant of success while another is exposed to additional damage. If adding a settling/check window, define it as a separate uniform protocol before comparison.

## 8. Stage 2: oracle candidate selection diagnostic

Use the same valid states for both arms. Primary state bank: one pre-damage intermediate snapshot per parent reset, collected under a balanced ACTION/INTENT source-policy mixture. Choose the target replanning time independently of candidate outcomes. Record missing states due to earlier success/damage/termination and report attrition by source. Keep RESET states as a separate secondary stratum if evaluated; do not pool them invisibly.

For intermediate states, distinguish whole-episode from remaining-episode safety. Primary states must have no prior damage; retain accumulated health and counters exactly. Never heal an already damaged scene to create a safe starting point. A future recovery study on damaged states would need a different metric.

### Full snapshot requirements

Reuse verified physics/controller/wrapper/history restoration, then add ALL DamageSim state: entity/link health, accumulated damage, event histories/latches, thermal/fluid state, time-dependent evaluators and task bookkeeping. Record normalized and raw observations and recurrent state where relevant.

Replay identical actions after restore through a damaging contact: physical traces, health deltas, damage events and success must agree. Test branch-order invariance, disk round-trip and post-chunk caching. Physical-state equality with different health is a failed restoration. Do not assume the previous square-task tests cover this backend.

### Candidate protocol

- M=16 IID candidate samples per arm per state; report K={1,4,16}.
- Match physical execution horizon and control rate across arms.
- INTENT: vary intent noise; keep action-decoder noise fixed if the decoder is stochastic.
- ACTION: vary action noise from the trained prior. Never use sixteen identical zero-noise samples.
- Generate/store candidate noises, decoded intents and physical action chunks before scoring.
- Restore the same snapshot, execute one native action chunk, then continue with a frozen tail policy until the original episode budget ends.
- Primary tails: each arm's native policy; if promising, add a common ACTION tail diagnostic to test continuation dependence. Treat handoff degradation explicitly.
- Use four continuation replicates per candidate. Share keyed continuation seeds across candidates within an arm; keep candidate RNG separate from physics, health and tail-policy RNG.
- Verify precision and batch-size behavior; generate candidates with a fixed batching scheme and cache them. Do not mix native evaluation at one precision with candidate evaluation at another without auditing the impact.

Outcome tensor includes task success, safe success, damage and entity health for every state/candidate/replicate. Do not score only immediate chunk damage; a harmless first chunk can lead to later harm.

### Oracle and independent validation

For each state/replicate, let m be the number of safe-successful candidates among M. Oracle availability at K is:

    C_safe(K) = 1 - choose(M-m, K) / choose(M, K)

Average across states/replicates, keeping strata separate. Report the corresponding task-success availability. This observes rollout outcomes and is not deployable inference.

Cross-validated simulator selection: rank candidates by safe-success frequency on continuation seeds 0–1, then evaluate the selected candidate on seeds 2–3. Break ties using a seeded outcome-independent permutation. Report held-out task success and damage too. Use K=1 as random selection and K=16 as all candidates; average K=4 across 128 fixed candidate subsets. With two ranking replicates, ties will be common; report tie rate and ranking uncertainty.

This independently checked selection result is still simulator-assisted, not a learned scorer. It asks whether good candidates remain good under new continuation randomness. An oracle gain without independent-validation gain may reflect ranking noise or fragile trajectories.

## 9. Stage 3: learned selection, only after useful headroom

Collect branch outcomes on NEW training parent resets. Use separate development/test parents; every descendant state/candidate remains in its parent's partition. Never fit a selector on evaluation rollout labels and then report those same parents as held out.

Minimal neutral selector design: predict safe-success and task-success probability from observation history plus the proposed physical action chunk. Use the same architecture, observation access, label budget and candidate count for ACTION and INTENT. This first tests proposal quality with a shared scoring interface. Frozen observation encoders reduce training cost.

Choose the candidate with highest estimated safe-success probability, with a task-success tie-break. Evaluate actual closed-loop rollouts, not just classification metrics. At every deployment replan, generate K candidates and score them without simulator lookahead or future health access.

Report task completion and safe completion together; optionally tune a task-success constraint on development data only. If using current health as input, expose the same current information to both arms and declare the privileged simulator observation. Never expose future damage, simulator counterfactual outcomes, or intent ground truth at deployment.

Compare:

1. ACTION_MIX, random candidate.
2. INTENT_MIX, random candidate.
3. ACTION_MIX, learned selection at K.
4. INTENT_MIX, same learned-selection protocol at K.
5. ACTION_SAFE, random candidate, when available.

Log both inference cost and total label-generation simulator steps. Equal K is not equal inference cost. All selector-training rollouts count toward the method's data budget.

A selector operating directly on intent instead of action chunks is a separate representation ablation, not an apples-to-apples proposal comparison. Later, a same-dimensional unstructured latent control can distinguish future-motion supervision from lower-dimensional search. Damage-aware DSRL is already prior work; implement that comparison only if the claim becomes RL adaptation efficiency.

## 10. Statistics, budgets and stopping rules

Bootstrap parent reset families, preserving arms, candidates, replicas and dependent snapshots; use 2,000 resamples for 95% paired intervals. Report training-seed results separately; one seed gives conditional evidence. Avoid counting 16 candidates as 16 independent environments.

Suggested stages:

- Data audit: one task, all episode metadata, 4–12 targeted replay checks and no new policy training.
- Reproduction: 30 development resets for task competence/safety gap.
- Branch smoke test: 2 states, 3 candidates, 2 replicas, 2 arms = 24 continuations.
- Branch pilot: 12 parent resets, one eligible state each, 16 candidates, 4 replicas, 2 arms = at most 1,536 continuations.
- Confirmation only if promising: 50 NEW parent resets under the same protocol = at most 6,400 continuations per checkpoint pair. Additional reset strata multiply this budget and must be explicit.

Measure actual throughput, queue delay and average remaining horizon before confirmation. Reuse stored candidates/outcomes for all K, not separate runs. Stop on configured wall time/environment-step budget and record incomplete pairs.

Decision thresholds are project choices, not universal significance requirements. Freeze a practical target before confirmation, for example a 5 percentage-point safe-success improvement with no more than a 5-point task-success loss. Report intervals; failing to detect a task-success loss does not prove noninferiority.

Proceed when intent has useful absolute safe-success headroom or a reproducible selection advantage over ACTION at the same budget. Larger uplift from a much weaker intent starting policy alone is insufficient.

If task completion is near zero, diagnose competence. If everything is safe already, diagnose ceiling. If outcomes remain dominated by force/timing absent from intent, reconsider that representation. If properly functioning controls show no useful advantage with adequate precision, stop this formulation rather than add tasks indefinitely.

## 11. Engineering outputs

Required manifests: repository commit/diff hash, simulator/assets/data versions, checkpoint hashes, training updates/seeds, observation/action schemas, controller mode, intent definition, sampler distribution/NFE, precision, damage config, evaluation horizon, partitions and budget.

Suggested modules, adapted to repo conventions:

    audit_oopsie_demos.py
    reproduce_oopsie_baseline.py
    health_snapshot_adapter.py
    evaluate_safety_candidates.py
    analyze_safe_selection.py
    train_safety_selector.py      # gated later stage

Minimum outcome record:

    experiment_id, stage, task, parent_reset_id, state_id, split,
    arm, checkpoint_hash, candidate_id, candidate_noise_hash,
    intent_value_path, action_chunk_path, continuation_seed, tail_mode,
    task_success, safe_success, any_damage, first_damage_step,
    initial_health_path, final_health_path, entity_damage_path,
    steps, termination_reason, latency, infrastructure_error, policy_error

Write atomic records and deterministic job keys; refuse resume if checkpoint, config or damage semantics change. Retry infrastructure errors deterministically once, preserve logs and report missing paired data. Invalid model actions count as policy failures, not infrastructure exclusions.

Deliverables:

1. `demo_representation_audit.md`, inventory and safe/unsafe paired examples.
2. `baseline_reproduction.md`, resolved interfaces and competence/safety rates.
3. Frozen protocol/config plus restoration-through-damage checks.
4. Safe-success and task-success versus K, with paired intervals and exact counts.
5. Independent continuation-selection results and tie rates.
6. Entity-level damage breakdown and inference/data costs.
7. An explicit verdict: blocked by data, blocked by competence, representation mismatch, promising proposals, promising learned selection, or no advantage under this protocol.

## 12. Immediate instruction to the coding agent

Start with Stage 0 only: inspect the already-installed OopsieVerse environment, downloaded playback datasets, damage semantics and existing runs. Produce the demonstration/representation audit and identify one task with both safe and damaging successes. Check whether mean future EEF pose preserves the relevant behavioral distinction. Report concrete evidence and estimated baseline-reproduction cost. Do not launch multiple tasks or redesign the representation before that audit is complete.
