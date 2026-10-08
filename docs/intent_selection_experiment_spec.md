# Intent selection: experiment specification for the coding agent

Version: 1.0 — 5 October 2026

## 1. Assignment and scope

Implement an evaluation harness to answer:

> At an identical robot/environment state and equal candidate budget, does a pretrained intent-conditioned policy offer more useful candidate action chunks than a pretrained action-only flow policy, and does this advantage survive independent continuation randomness?

Use the existing **Robomimic square-mh-state** checkpoints first. This is a diagnostic of existing checkpoints. Do not train new policies, add encoders, launch DSRL, or start RoboCasa experiments in this phase.

The source project notes report a 23-dimensional observation containing object state, EEF position/quaternion and gripper state. The architecture figure distinguishes high-dimensional action noise from a 7-dimensional intent-noise input. These are leads to inspect, NOT authoritative configuration. Read actual checkpoint metadata and source. Prediction horizon, execution horizon, history length, normalization, rotation representation, and noise dimension must be resolved from the repository.

No repository or checkpoints were available when this specification was written. Names below are proposed interfaces, not claims about existing code. Adapt to the repository's conventions and AGENTS.md; preserve existing evaluation paths.

Deliver a correct harness, smoke-test report, and small pilot first. Estimate cost before the confirmation stage. Do not silently launch all proposed stages.

## 2. Hypotheses and limits

H1: Sampling intent-conditioned candidates yields higher successful-continuation coverage at equal K than action-only candidate sampling.

H2: Candidates differ in their reproducible probability of success; the apparent benefit is not solely luck in subsequent policy sampling.

H3, reserved for a later experiment: A learned selector can exploit those differences on unseen states with less data or computation.

This experiment tests H1 and provides evidence about H2. It does not establish H3, semantic intent, generalization, novelty, or an advantage caused specifically by intent supervision. Checkpoint quality, architecture and dimensionality remain confounders.

A null H1 result does not logically disprove easier learning in intent space: equal coverage could still be easier to select. It does weaken the proposed coverage-based salvage route. An unsuccessful one-chunk intervention also does not disprove interventions over longer horizons.

## 3. Policy arms and immutable manifest

Use two frozen policies in evaluation mode:

| Arm | Candidate generation | Execution |
|---|---|---|
| ACTION | Independently sample native action-flow initial noise; infer one action chunk given the fixed observation history | Execute the next H_exec physical actions |
| INTENT | Independently sample native intent-flow initial noise; decode intent; infer the conditioned action chunk | Execute the next H_exec physical actions |

If INTENT has a stochastic action decoder, hold its action-decoder noise fixed across intent candidates at each state. Record that fixed seed. Candidate variation must then come from intent noise only. A joint intent-plus-action sampling variant is optional and must have a separate arm name. With an MLP decoder this distinction is unnecessary.

Use native sampling distributions, flow schedules and normalization from the checkpoint. Do not apply the RL actor's tanh transform, noise magnitude, mean shift, or temperature changes. This evaluates unsteered BC policies.

Record in manifest.json:

- Code commit and dirty diff hash; dependency and simulator versions; hardware.
- Checkpoint path and SHA-256; training seed, training data identity, selected training step if known.
- Exact observation keys/order/dimensions, history length and normalization hashes.
- Action prediction horizon, execution horizon, physical control frequency, environment action dimension and internal action representation.
- Intent definition, horizon, reference frame and normalization; sampler noise dimensions, schedules and number of function evaluations.
- Environment/task config, model XML/assets identity, controller settings, episode timeout and official success predicate.
- Snapshot schema version; all seed namespaces; state-bank split and hash; continuation mode.
- Every unresolved field. Fail before rollout if an unresolved field affects inference or action execution.

Choose checkpoint steps using existing validation information, never the new test bank. Do not search checkpoints for a desired result. If multiple existing training seeds are available, preserve paired seed identities; a single pair supports a checkpoint-specific conclusion only.

## 4. Stage A: audit and meaningful sanity checks

### A1. Reproduce native evaluation

Run the original evaluator and the new adapter with identical resets/seeds for each arm, initially 10 episodes each. Confirm matching predicted chunks and rollout behavior for matched randomness. Compare to saved performance, but do not require an exact historical percentage when original seeds are unavailable.

Any material disagreement requires explanation before branching. Log predicted actions before and after normalization/controller conversion. Never sum normalized actions into a supposed physical trajectory.

### A2. Verify exact restoration

A simulator qpos/qvel vector alone may be insufficient. Save and restore every piece of state required by the actual environment:

- Simulator integration state, time, actuator state and relevant mutable model parameters.
- Environment/task bookkeeping, elapsed steps, termination flags, success latch and randomized initial configuration.
- Controller internal state (targets, integrators, filters, interpolation state where applicable).
- Wrapper state, observation normalization state, observation history, recurrent caches and action queues.
- Environment stochastic state and policy/random generators, kept in separate namespaces.

Do not advance physics to obtain observations after restoration. If an observation update mutates state, account for it explicitly. Check native state APIs in the installed version; do not assume an API name from this document.

Mandatory restoration tests:

1. Save a snapshot. Execute a predetermined legal action sequence. Restore and repeat. Physical trajectories, rewards and termination must match within a documented numerical tolerance.
2. Restore the same snapshot twice with identical candidate and continuation seeds: identical results.
3. Permute branch evaluation order: results indexed by state/candidate/seed remain identical.
4. Repeat with different batch sizes and worker order: candidate identity and outcomes remain unchanged within documented tolerance.
5. At a native replanning boundary, adapter sampling matches native policy sampling. Clear only the previous action queue, while restoring the correct history/recurrent state. Do not clear history and pretend an intermediate state is an initial state.

For CPU float64 state replay, 1e-8 may be a useful initial position tolerance; choose tolerances appropriate to the actual runtime and report them. A reproducibility failure blocks the scientific run.

### A3. Intent causality audit

At five fixed pilot states, sample 16 native intents and log their decoded values and resulting action chunks. Verify that changing intent changes the action when intended. Check for ignored conditioning, accidental broadcasting, shared noise across a batch, stale caches, collapsed samples and future-observation leakage.

Do not feed ground-truth future intent at inference. Do not average/sign-flip/renormalize quaternion labels differently from training as part of evaluation. Record suspicious training representations for a separate audit.

## 5. State bank: identical states, independently chosen

Keep two strata separate; do not pool them into a single headline result.

### RESET stratum

Use official valid test resets drawn with a fixed reset seed list, disjoint from training demonstrations where the benchmark permits. Save the initial snapshot and the native initial history (including native frame padding).

### MID stratum

Use the same parent reset IDs. Generate one source rollout per parent, balancing source policy ACTION/INTENT across parent IDs. Choose source assignment by deterministic hashing, not outcomes.

Before rollout, choose one target replanning index per parent from a preregistered interval of the episode, e.g. the nearest native boundary to a uniform time in [0.2T, 0.6T]. Capture that state if the rollout has not already terminated or succeeded. Reject already-successful states. Do not replace a missing state by searching for a difficult-looking one; record why it is missing. Add preregistered reserve parents to reach the target count if necessary.

Record source policy, source seed, timestep and parent reset ID. Balance achieved source counts or report imbalance. Save the source observation history. For recurrent policies, reconstruct each evaluated policy's own cache from that history or replay prefix; do not copy one model's hidden state into another model.

Do not select by intent variance, clustering, success after branching, or visual appeal. If a later difficulty-enriched bank is desired, define it using separate screening rollouts before candidate testing and label its results exploratory. It is not needed for the first experiment.

Split development/test by **parent reset ID** before inspecting branching outcomes. Every snapshot and candidate descendant of a parent stays in the same split. For learned selectors later, use a third training split with new parents.

## 6. Candidate generation and branching semantics

Default K_max=16; report K in {1,4,16}. Sampling is IID under each arm's native proposal distribution. Generate and store all candidate noises, decoded intents if applicable, and final physical action chunks before evaluating outcomes.

H_exec is the native number of physical environment steps executed between replans, resolved from code. Both arms must use the same H_exec and control rate. If native execution horizons differ, document the mismatch: use an explicit matched-horizon comparison and retain native-horizon diagnostics. Never silently call one 8-step candidate equivalent to one 10-step candidate.

For each candidate:

1. Restore the identical full starting snapshot and correct policy history.
2. Execute exactly H_exec actions from its precomputed chunk, unless official success/termination occurs sooner.
3. Update observation history at each physical step according to the native wrapper. Save the post-chunk snapshot and history.
4. Discard the unused suffix of the predicted chunk, as in native receding-horizon execution.
5. Continue from that state using a specified frozen tail policy, with fresh deterministic-by-ID seed streams, until success, failure termination or the original episode timeout.

The total physical horizon is T minus the snapshot's elapsed steps. Do not give MID branches a fresh full episode. Store actual EEF/object traces from the simulator for analysis.

Never clamp one short-horizon intent for the rest of the episode. Never replace independent sampling with arbitrary hand-edited intent coordinates.

## 7. Continuation protocol and randomness

### Primary: native tails

After the intervention, ACTION continues with its own native policy and INTENT with its own native policy. This measures the practical opportunity for selection within each full policy. It does NOT isolate candidate quality from continuation-policy quality.

### Required follow-up if the primary result is promising: common tail

Re-evaluate stored candidates using the same frozen ACTION tail for both arms. Preserve identical continuation mechanics and horizon. This asks whether differences survive holding downstream control fixed.

Policy handoff can itself be harmful. Report K=1 handoff performance and compare to native tails. If the effect vanishes only under a poor handoff, call the result continuation-dependent, not automatically invalid. A reciprocal INTENT common-tail test can resolve this only if needed.

### RNG implementation

Derive deterministic seeds using a stable hash, never Python's process-dependent hash(). Use separate namespaces for state collection, candidate sampling, decoder noise, continuation policy sampling, environment stochasticity, bootstrap and video selection.

Candidate seeds depend on (experiment, arm, checkpoint pair, state ID, candidate ID). Continuation policy seeds depend on (experiment, state ID, continuation ID, relative replan index, policy identity), deliberately excluding candidate ID. Environment stochastic seeds likewise exclude candidate ID and should be shared across arms where meaningful.

Sharing seed integers across different-dimensional policies does not create identical semantic noise. The purpose is reproducibility and coupling continuation randomness within an arm. Prefer stateless keyed draws per replan to avoid RNG consumption changing after early termination or across worker scheduling.

If the environment is stochastic during the intervention, repeat the intervention per environment seed. The shortcut of caching one post-chunk state is valid only after confirming deterministic transitions conditional on snapshot and actions.

## 8. Outcomes and metrics

Let Y[a,s,j,r] be binary task success for arm a, state s, candidate j and continuation replicate r. Log time-to-success, termination reason, task-native return, and physical traces as secondary fields. Use official success; distinguish simulator exceptions from legitimate policy failures.

### M1. Random-candidate success

For each state, p_s = mean over j,r of Y[s,j,r]. Average states equally within a stratum. This is the K=1 reference distribution. It need not equal historical full-episode SR, especially for MID states or common-tail handoffs.

### M2. Oracle availability at equal K

For a fixed state and continuation replicate, let m be the number of successful candidates among M=16. Define:

    C_s,r(K) = 1 - choose(M-m, K) / choose(M, K)

Use choose(n,K)=0 for n<K. This exactly averages the event 'at least one success' over all size-K subsets of the sampled candidate pool. Average over states and continuation replicates within each stratum. Compute combinations stably; test all-zero/all-one/K=1/K=M cases.

This is finite-pool oracle availability under the specified rollout protocol, not a deployable selection score or the maximum of true success probabilities. It is deliberately optimistic because it observes outcomes. Do not replace it with 1-(1-global_SR)^K: that erases state heterogeneity.

Report both C_ACTION(K), C_INTENT(K), and their paired difference. Also report per-arm gain C(K)-C(1). A larger gain alone may just reflect a weaker K=1 policy.

### M3. Cross-validated simulator selector

Use R=4 continuation replicates for confirmation, divided in advance into selection seeds {0,1} and held-out seeds {2,3}.

For each subset S_K of candidate IDs:

    j_star = argmax over j in S_K of mean(Y[j, selection_seeds])
    score = mean(Y[j_star, heldout_seeds])

Break ties with a seeded random permutation independent of held-out outcomes. For K=1 and K=16, compute directly; for K=4 use 128 reproducibly sampled distinct subsets. For K=1 average all candidates, giving the random-candidate reference. Compare selected success against the mean held-out success of candidates in the same subsets. Integrate tie-breaking analytically or with one fixed documented permutation; never use test outcomes to break ties.

This selection procedure uses expensive simulator trials and is NOT a learned or deployable selector. It tests whether a successful candidate remains good under independent continuation randomness. With only two selection replicates, ranking is noisy; report the tie fraction and split sensitivity. Increase R to 8 on a preregistered follow-up subset only if this uncertainty changes the decision, not selectively on favorable states.

### M4. Diagnostics

- Per-state success counts and candidate success probabilities estimated from replicates.
- Candidate duplication rate in physical action space, using a documented tolerance.
- Achieved EEF/object displacement per candidate, intent-target versus achieved-motion relation where well-defined.
- Episode timeout/failure breakdown and invalid-action frequency.
- Runtime, inference time, sampler evaluations and total simulator steps by arm.
- Rescue analysis on a separate independently screened failure bank only, if requested later. Do not select failed states using the same outcomes you use to estimate rescue rates.

Diversity plots are diagnostic. Do not use VDR, silhouette or UMAP as primary evidence of useful strategies.

## 9. Statistics and comparison limits

Use paired comparisons because both arms start from the same snapshots. Bootstrap parent reset IDs with replacement, preserving all their states, arms, candidate outcomes and replicate assignments. Use 2,000 bootstrap resamples for 95% percentile intervals. Report each stratum separately. Candidates and continuation repetitions are NOT independent test states.

Intervals from one checkpoint pair describe uncertainty across evaluated reset families conditional on those checkpoints; they do not establish training-seed robustness. Repeat with at least three independent checkpoint pairs before a broader paper claim if resources permit. Report seed-level results, not just pooled counts.

The pilot is not a powered hypothesis test. Do not infer equivalence from a nonsignificant difference. Report effect sizes and intervals, and label small/inconclusive results as such.

A difference between these arms is a system-level difference. If the result survives, future controls should include equal-dimensional unstructured latent policies, matched training budgets and inference costs, and repeated training seeds. Do not claim intent semantics caused the gain from this experiment alone.

## 10. Staged budget and decision rules

### Smoke test

Two parent resets, one RESET and one available MID snapshot each; K=3, R=2, two arms. At most 48 branch continuations. Run restoration and order-invariance checks. No scientific conclusion.

### Pilot

12 new parent resets: 12 RESET and up to 12 MID snapshots; K=16, R=2, native tails. At most 1,536 continuations, plus source collection and native-evaluation audit. Use this to estimate runtime, feasibility, success saturation and metrics. These are development states and never enter the confirmation result.

Pilot cannot validate M3 robustly. Its R=2 can provide a one-selection/one-validation preview only, labeled exploratory.

### Confirmation

60 new parent resets: 60 RESET and up to 60 MID snapshots; K=16, R=4, two arms, native tails. At most 15,360 continuations per checkpoint pair. Collect preregistered reserve parents if MID attrition is large. Reuse one outcome tensor for all K values; do not rerun separately for each K.

For example, 15,360 continuations at an average of 200 remaining physical steps is about 3.07 million environment steps. Actual runtime must be measured from the pilot. If the allotted budget cannot support this, reduce parent count transparently and label the result exploratory. Do not shrink R silently or substitute candidate counts for independent state counts.

Primary endpoint: paired INTENT minus ACTION difference in C(16) on MID. RESET, K=4 and cross-validated M3 are supporting endpoints. Freeze this choice before confirmation. Report every endpoint regardless of sign.

Practical continuation rule, chosen as a project decision threshold rather than a universal publication standard:

- Promising coverage: approximately 5 percentage points or more advantage in C(16), with uncertainty supporting a positive effect, and a credible M3 gain over random selection. Run common-tail follow-up and replicate on another checkpoint seed/task.
- Equal coverage but easier selection: if M3 is reliably better despite similar C(16), pursue the narrower selectability hypothesis. This remains a legitimate positive outcome.
- Weak starting policy: if INTENT has a large uplift but its selected success still trails ACTION, call it useful headroom within a weaker policy; do not claim superiority. Decide whether improving the decoder is worth a new experiment.
- No advantage with a sufficiently narrow interval: stop expanding the coverage-based branch on these checkpoints. Record that it does not refute every intent formulation.
- Ceiling, floor, wide intervals, broken restoration, or unstable evaluation: inconclusive. Resolve that specific limitation within the agreed budget; do not call it a scientific negative.

Do not automatically escalate K to hundreds, change tasks, tune samplers or train critics to rescue an unfavorable confirmation result.

## 11. Proposed modules and interfaces

Suggested structure, adapted to the existing repo:

    intent_selection/
      adapters.py          # native policy sampling and physical action conversion
      snapshots.py         # complete env/controller/wrapper capture and restore
      state_bank.py        # deterministic collection, provenance and splits
      candidates.py        # frozen candidate pool generation
      branch_eval.py       # intervention and seeded continuation execution
      metrics.py           # oracle availability, independent selector, bootstrap
      report.py            # CSV tables, figures, readable experiment summary
      configs/             # smoke, pilot, confirmation
      tests/               # replay, isolation, metrics and resume correctness

Required logical interfaces:

    capture_full_state(env, wrapper, history) -> Snapshot
    restore_full_state(snapshot, env, wrapper) -> ObservationHistory
    adapter.sample_candidate(history, noise_seed, decoder_seed) -> Candidate
    execute_candidate(snapshot, candidate, h_exec) -> PostChunkState
    run_tail(post_chunk_state, tail_policy, continuation_id, remaining_budget) -> Outcome
    collect_state_bank(config) -> BankManifest
    evaluate_shard(bank, candidate_pool, job_ids) -> OutcomeRecords
    analyze(manifest, records) -> TablesAndFigures

Candidate must expose noise, decoded intent when applicable, normalized model output, physical action sequence, hashes and sampling latency. Snapshot serialization must be versioned. Avoid coupling candidate generation to environment stepping.

Pseudocode for deterministic intervention environments:

```python
for state in assigned_states:
    for arm in arms:
        candidates = load_or_generate_candidates(state, arm, M=16)
        for candidate in candidates:
            post = load_or_execute_candidate(state, candidate, H_exec)
            for continuation_id in continuation_ids:
                key = result_key(state, arm, candidate, continuation_id, tail_mode)
                if verified_result_exists(key):
                    continue
                restore_post_chunk_state(post)
                install_keyed_continuation_rng(state.id, continuation_id)
                result = run_tail_or_return_terminal(post, tail_for(arm, tail_mode))
                atomic_write_result(key, result)
```

Terminal success during the candidate chunk yields success for every continuation replicate. Terminal task failure yields failure. A numerical/runtime error is not a success/failure sample until classified: retry once deterministically, preserve error logs, and block the aggregate if unresolved failures can bias arm comparisons. An illegal/nonfinite policy action is a policy failure and must also be counted explicitly; do not discard it as infrastructure noise.

## 12. Configuration template

```yaml
experiment_id: intent_selection_square_mh_v1
phase: pilot
task: square-mh-state
checkpoints:
  action: REQUIRED_ABSOLUTE_PATH
  intent: REQUIRED_ABSOLUTE_PATH
checkpoint_pair_id: seed_pair_0
freeze_models: true
sampling: native_prior
task_config_source: checkpoint_and_native_eval
execution_horizon: RESOLVE_FROM_REPO_AND_MATCH
episode_horizon: RESOLVE_FROM_NATIVE_EVAL
candidate_pool_size: 16
report_k: [1, 4, 16]
continuation_mode: native
continuation_replicates: 2   # confirmation: 4
state_bank:
  parent_resets: 12         # confirmation: 60 NEW parents
  strata: [reset, mid]
  mid_source_mix: {action: 0.5, intent: 0.5}
  mid_time_fraction: [0.2, 0.6]
  mid_states_per_parent: 1
  reject_already_successful: true
  split_unit: parent_reset_id
  selection_uses_branch_outcomes: false
analysis:
  primary_stratum: mid
  primary_k: 16
  subset_samples_k4: 128
  bootstrap_unit: parent_reset_id
  bootstrap_resamples: 2000
  practical_effect_threshold: 0.05
runtime:
  max_env_steps: REQUIRED_BUDGET
  max_wall_hours: REQUIRED_BUDGET
  resume: true
  deterministic_job_ids: true
  render_all_rollouts: false
```

REQUIRED/RESOLVE values must be resolved and persisted before running. The agent should return the proposed resolved config and measured pilot cost before a large campaign. Existing user-provided compute authorization can cover launch; do not add an unnecessary permission step.

## 13. Storage, resuming and outputs

Store manifests, state bank and candidate pool independently of results. Write atomic per-job records so interruptions do not lose a campaign. A job key includes experiment/config hash, checkpoint hash, state ID, candidate ID, continuation ID, tail mode and snapshot version. Refuse resume when identity changes. Record simulator steps consumed for budget enforcement.

Minimum outcome columns:

    experiment_id, config_hash, checkpoint_pair_id, arm, checkpoint_hash,
    split, stratum, parent_reset_id, state_id, source_policy, elapsed_steps,
    candidate_id, candidate_seed, decoder_seed, continuation_id, tail_mode,
    success, termination_reason, steps_executed, success_step, return,
    candidate_latency_ms, tail_latency_ms, error_status, candidate_hash

Keep trace files separate and reference them by key. Retain candidate and intent arrays for every job; render only a small predefined diagnostic sample. For evidence videos, show paired examples chosen by a declared rule, including both arms' failures, and state that examples are selected.

Required outputs:

1. `audit_report.md`: resolved policy semantics, native parity, restoration checks, unresolved limitations.
2. `manifest.json`, `state_bank_manifest.json`, candidate arrays, outcome records.
3. `metrics_by_state.csv`, `aggregate_metrics.csv`, bootstrap intervals and per-seed tables.
4. Oracle availability versus K, separated by RESET/MID and continuation mode.
5. Cross-validated selected success versus random success, with intervals and tie rates.
6. Paired per-state plot comparing arm coverage, plus latency and simulator-step costs.
7. `experiment_report.md`: exact claim supported, effect sizes/uncertainty, confounders, decision under the frozen rule.

## 14. Definition of done and next stage

The first implementation is done when the audit passes, all smoke-test records are reproducible, the pilot finishes/resumes correctly, and analysis reconstructs every reported number from saved records. A successful engineering run is not a positive research result.

If confirmation is promising, the NEXT separately scoped experiment is a learned selector: train a small Q-like scorer on candidate/rollout outcomes from new training parents, tune on development parents, and evaluate on untouched test parents. Compare intent-candidate selection to action-candidate selection using equal data, candidate counts and clearly reported inference costs. Use the same observation access in both arms; privileged simulator state must not enter a vision-only selector. Online DSRL is a subsequent option, not a prerequisite for this diagnostic.

Return an honest verdict: promising coverage, promising selectability, useful only within a weaker policy, inconclusive, or no advantage under this protocol. Do not turn a larger spread in trajectories into a claim of successful behavioral diversity without outcome evidence.
