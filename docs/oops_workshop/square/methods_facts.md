# Methods facts: spatial-shift robustness, flow-matching policies with vs without predicted-intent conditioning (robomimic square-mh-state)

Substitute study, run on the local workstation. The pi0.5 / LIBERO-Goal checkpoints named in
`docs/oops_workshop_robustness_spec.md` exist only on the cluster, which was unavailable for this submission. The
question is the same: does conditioning on a *predicted* future representation improve success under spatial
shifts relative to a matched policy without it? Every fact below was read from code, checkpoints or the simulator
in this repository on 2026-10-05.

## Models (one training run each, training seed 0)

| | Baseline (A) | Intent-conditioned (I) |
|---|---|---|
| checkpoint | `checkpoints/square_mh_state_flow_mlp_512_h10_seed0_success62.pt` | `checkpoints/square_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success64.pt` |
| sha256 | `a687c8f78925f7792e8f7c85540a62080dbe3c87dce42373ad16b44c801943e4` | `1e740970bad86b4afb92c1b107086b641e012c572f16d0d63e680c6ff4f21b14` |
| agent | `TrainingAgent`: flow-matching action policy, MLP emb 512 | `FlowIntentAgent` (Config A): shared MLP obs encoder (0.8 M) -> flow model samples a 7-D intent from noise given the observation (17.5 M) -> deterministic MLP action decoder given (obs embedding, intent) (2.4 M) |
| intent definition | — | mean future end-effector pose (position + quaternion) over the next 8 steps (`intent_type: mean`, `intent_horizon: 8`); trained independently of the action loss (`intent_training_mode: independent`); at test time **sampled from the policy's own flow model given only the current 2-step observation window** (no future or oracle information) |
| task config | `task=square_mh_state` | `task=square_mh_state_flow_intent +network.arch_variant=flow_intent` |
| training data | robomimic square **mh** low-dim, 300 demos (HF `ChaoyiPan/mip-dataset`, `robomimic/square/mh/low_dim.hdf5`), no validation split | same |
| observations / actions | 23-D state (nut object state 14, EEF pos 3, EEF quat 4, gripper 2); 2-step history; absolute actions (pos + rotation-6d + gripper), horizon 10, 8 executed per replan | same |
| training budget | **49,999 gradient steps** | **189,999 gradient steps** (budget differs: comparison is confounded by training length) |
| checkpoint selection (documented, not changed) | training-time eval every 10k steps, 50 episodes, NFE 9: 0.32, 0.16, 0.32, 0.30, **0.62** (final = best) | 19 evals: 0.40-0.58 range, mean 0.47, final **0.64** (final = best) |

## Evaluation (identical for both arms)

- Native rollout code: the unmodified `collect_rollouts()` from `examples/collect/collect_diversity_rollouts.py`
  (the function behind `examples/eval/eval_success_rate.py`), called through a one-env adapter
  (`examples/openpi/oops/square_eval.py`). Note: `eval_success_rate.py` itself currently crashes (it unpacks 4 of
  `collect_rollouts`' 5 return values, and its ROOT path is stale since it moved to `examples/eval/`).
- Flow sampling steps (NFE) = 9 for both arms: the primary metric of the native training-time evaluator, which
  produced the checkpoint labels. (`eval_success_rate.py` defaults to NFE 1, where the documented baseline SR
  is 0.18 vs 0.44 for the intent model; that setting is not used.) Baseline uses the native `sample_mode: zero`
  (deterministic given the observation); the intent model samples its intent stochastically.
- 400 environment steps; success = any step with reward > 0 (task success; the episode is not terminated early).
- Per-episode seed `oops_common.episode_seed(...)` seeds numpy and torch identically for both arms; CPU inference.

## Physical cases and perturbation

- 50 benchmark resets: reset k is the native seeded reset `np.random.seed(42 + k); env.reset()` (42 = native
  eval seed); its full flattened MuJoCo state is stored and hashed.
- Moved body: the square nut (free joint `SquareNut_joint0`, flattened index 10). World-XY translation of the stored
  state by delta [cos theta, sin theta]; z and orientation preserved. Pegs, table and robot unchanged.
- Grid (fixed before any rollout): delta in {0, 0.05, 0.10} m (primary) plus 0.02 m (pre-declared supplementary);
  10 directions theta_k = 2 pi k / 10; delta = 0 once per reset. 50 x (1 + 3 x 10) = 1,550 physical cases; one
  policy seed per case per arm (delta = 0 additionally has 3 seeds in the parity cohort).
- Start: seeded native reset (reproduces robot pose and controller references), then `reset_to(stored state)`.
  Every delta, including 0, uses this same procedure.
- Demonstrated start positions (300 demos): nut x in [-0.115, -0.110] m (5 mm wide), y in [0.111, 0.225] m
  (11.4 cm), yaw random. Every shifted case (delta >= 0.02) starts outside this bounding box (median distance
  outside: 1.3 / 3.7 / 7.9 cm at delta = 0.02 / 0.05 / 0.10). Shifts are "outside the demonstrated start-position
  range of the nut"; the rest of the scene is in distribution.

## Validity (policy-free, decided before rollouts)

Probe (never part of a rollout): robot held kinematically, 250 MuJoCo substeps to settle + 250 to test stability;
rules = `VALIDITY_RULES` v1 thresholds (penetration > 5 mm, XY drift > 4 cm, Z change > 3 cm, still moving > 1 cm,
pre-satisfied goal, off-table). Result: **1,550/1,550 cases valid** (no penetration, drift < 1 mm, no
pre-satisfied goal). The open square-nut scene has no obstacle within 10 cm of the nut, so the validity filter is
inactive here, unlike LIBERO-Goal, where up to 41% of 10 cm placements are physically invalid.

## Parity with the native evaluator

- Bitwise: at delta = 0 the adapter's first observation and first action chunk equal the native path's exactly.
  Later chunks differ (max |Delta a| 5.6e-3 baseline, 1.3e-2 / 2.6e-2 intent). Cause, verified: the extra `mj_forward`
  inside `reset_to`. A single redundant forward pass in the native path alone changes later actions by up to 4e-2,
  and the adapter without that forward pass matches native bit-for-bit. Rollouts are numerically chaotic, so
  bitwise parity is unattainable for any state-setting perturbation method.
- Statistical (50 resets x 3 seeds, delta = 0): baseline adapter 0.46 vs native 0.56 (deterministic policy; 9
  discordant resets, 2 vs 7, sign test p ~ 0.18); intent adapter 0.43 vs native 0.42 (18 discordant, 10 vs 8).
- Neither pipeline reproduces the checkpoint labels (0.62 / 0.64). Those are single 50-episode evaluations that
  happened to be each run's maximum; the intent run's 19-evaluation mean is 0.47.

## Analysis

Primary population: valid matched cases (all cases here). SR per delta, gap I - A, change from delta = 0, paired
cluster bootstrap over the 50 resets (2,000 resamples). One task, so macro = that task. Intervals are conditional on
these two checkpoints; one training seed per arm, so no training-seed variability is captured. Pre-registered
secondary breakdowns: direction class (y- vs x-dominant shift) and distance outside the demonstrated start box
(`square_breakdowns.py`).
