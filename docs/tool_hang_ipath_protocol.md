# tool_hang (ph, state): I-PATH vs no intent — protocol

Fixed before any I-PATH result was seen (2026-11-01).

## Question
On a competent policy (no-intent MIP + MLP, best-checkpoint success 84–92%, matching the upstream release),
does I-PATH intent help, and in particular does it solve starts the no-intent policy consistently misses?

## Arms (identical except intent)
| | no intent | I-PATH |
|---|---|---|
| runs | `state_mip_s{0,1,2}` | `state_ipath_s{0,1,2}` |
| task config | `tool_hang_ph_state` | `tool_hang_ph_state_ipath` |
| obs | 53-D state (object, eef pos/quat, gripper), 2 steps | same + 36-D intent appended to each step |
| actions | delta_legacy (delta OSC, 6D rot), horizon 10, execute 8 | same |
| policy | MIP loss, `network=mlp` (512), 1-step inference from zeros | same |
| budget | 300k steps, batch 1024, lr 1e-4 cosine, EMA 0.995 | same |
| training evals | 50 episodes every 20k, env streams seeded by the training seed | same streams (same seeds) |

I-PATH (recipe of `examples/oopsieverse_full/matrix/models.py`, target of `intent_targets.py` "ipath"):
- target: future EEF poses at t+2, 4, 6, 8 in the current EEF frame, pos + 6D rot = 4 × 9 = 36-D, MinMax
  to [-1, 1] over all training frames;
- action policy teacher-forced on the ground-truth intent;
- intent generator `IntentFlowPredictor` (rectified flow, t ~ Beta(1.5, 1), 650K-param MLP on the obs
  window, own Adam with the same lr/cosine schedule); at eval the intent is sampled with 16 Euler steps from
  N(0, I), then the action policy runs on [obs, sampled intent].
- Difference from OopsieVerse: MIP's state MLP has no shared trunk, so the intent loss does not shape the
  action network's features; and the action/intent nets have separate optimizers (no shared parameters,
  so the summed loss would only change clipping). Data order is not RNG-paired with the baseline.

## Checkpoint selection (same rule for both arms)
Primary: `models/model_best.pt` = the first training-eval checkpoint (of 15, at 20k…300k) reaching that
run's maximum 50-episode success. Secondary: `models/model_latest.pt` (300k).

## Fresh-reset evaluation
- Pool: reset seeds 100000–100499 (500 starts), shared by all 6 policies, disjoint from training-eval resets
  (`log.eval_reset_seed_base`; robosuite placement uses `np.random.seed(reset_seed)`, verified identical across
  processes).
- Each selected checkpoint × rollout seeds {0, 1, 2} (torch/numpy seed → intent-sampling noise). The
  no-intent policy is deterministic per reset; its repeats are a determinism check.
- Runner `scripts/tool_hang_fresh_eval.sh RUN ROLLOUT GPU [best|latest]`; analysis
  `scripts/tool_hang_intent_analysis.py`.

## Analysis
1. Success on the pool per arm × seed; paired per-reset difference (I-PATH − no intent) with a bootstrap
   CI over resets.
2. Hard starts, cross-fitted: for each held-out no-intent seed j, H_j = starts both other no-intent seeds
   fail on every rollout; compare held-out seed j vs I-PATH (all seeds) on H_j, pooled over j. This avoids
   the bias of picking starts with the baseline's own failures (which makes the baseline 0% by
   construction). The naive "all 3 seeds fail" set is reported descriptively.
3. The reverse (starts I-PATH consistently misses) with the same construction.

## Addendum A (2026-11-02): teacher-forced I-PATH failed; sampled-intent arm
Result of the arm above: best training eval 6/10/4%, 2/4/0% at 300k, vs no intent 90/92/92% (80/84/86% at
300k). Offline on 2,048 dataset samples (all three seeds alike), action MSE is 0.0001 with the GT path,
0.024 with the flow-sampled path, 0.0005 for the no-intent policy; the sampler is decent (MSE 0.006 vs
target variance 0.06). The teacher-forced policy decodes its actions from the GT path, so sampling error
dominates. The 500-reset fresh evaluation was not run for this arm.

New arm `state_ipath_sampled_s{0,1,2}` (`tool_hang_ph_state_ipath_sampled`): identical (target, four
waypoints, delta actions, architecture, budget, training evals, seeds), except that from step 20k the action
policy is conditioned on the flow predictor's sample (no gradient, 16 Euler steps) instead of the GT path.
The first 20k steps use the GT path (same switch point as the repo's DexMG `wp4_sampled` arm). Checkpoint
rule and analysis are unchanged; the 500 fresh resets are reserved for a variant that is competitive with
the baseline on training evals.
