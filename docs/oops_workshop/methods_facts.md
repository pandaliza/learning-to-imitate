# Methods facts: spatial-shift robustness of intent-conditioned pi0.5 (LIBERO-Goal)

Provenance tags: **[code]** read from code/launchers in this repo · **[sim]** measured in the LIBERO-Goal
simulator/data during this audit · **[doc]** stated in project notes/docs, not yet checked against checkpoint
configs · **[RESOLVE]** has to be filled from the cluster (`audit_existing_runs.py` -> `checkpoint_inventory.csv`)
before it can be written in the paper.

## Compared models

| | Baseline (A, "M0") | Intent-conditioned (I, "M1") |
|---|---|---|
| openpi config | `pi05_base_nointent` [code] | `pi05_base_intent` [code] |
| checkpoint | `.../ldahiya_checkpoints/pi05_base_nointent/base_nointent_goal_run1/29999` [code] | `.../pi05_base_intent/base_intent_goal_run1/29999` [code] |
| checkpoint hash | [RESOLVE] | [RESOLVE] |
| base weights | `pi05_base` (gs://openpi-assets/checkpoints/pi05_base) [doc] / config `weight_loader` [RESOLVE] | same |
| training | JAX `external/openpi/scripts/train.py`, batch 32, 30,000 steps (`--resume`), norm stats recomputed per config [code: `slurm-scripts/train/pi05/train_pi05_base_{nointent,intent}.sbatch`] | same |
| trainable modules | LoRA r=16, alpha=16 on Gemma-2B; SigLIP and Gemma-300M action expert fully finetuned [doc] / [RESOLVE] | same + intent projection into adaRMS [doc] |
| training seed | one run per arm (`run1`); seed value = openpi config default [RESOLVE] | same |
| training data | LIBERO-Goal, 10 tasks x 50 demos = 500 [sim: 50 demos per hdf5] ; LeRobot repo id [RESOLVE] | same demos + per-frame `intent` column [code] |
| intent at train time | none | 64-D z-hat **sampled per frame by the deployable generator** and baked into the dataset (`convert_libero_to_lerobot_intent.py`) [code] |
| intent at eval time | none | same generator (`mip.pi05_intent.IntentGenerator`, task config `libero_goal_suite_image_slot_intent`, ckpt `libero-goal-suite-image/slot_intent/models/model_best.pt`, EMA weights) sampled from the **current/past observation window only**: 128 px agentview + wrist, 15-D proprio [code]. No future frames, no oracle intent [code] |
| intent definition | — | slot-attention (K=4 slots, 64-D) target from future agentview frames (horizon 8) learned by a ResNet slot encoder; a flow map p(z given s) is trained to predict it from current observations; only the flow map is used at train/eval of pi0.5 [code: config; doc] |
| conditioning location | — | action-expert adaRMS conditioning [doc] / config [RESOLVE] |
| sidecar hash | — | [RESOLVE] |

Known context, not a result of this study [doc, aggregate only]: an earlier intent-override test on the
frozen-VL variant (M3) reported real/zero/random intent SR = 0.944/0.944/1.0, i.e. the action expert largely
ignored the intent channel. The M1 equivalent has not been verified for this checkpoint.

## Evaluation (identical for both arms) [code]

- Env: LIBERO `OffScreenRenderEnv`, 256x256 agentview + wrist, controller and control frequency = LIBERO defaults.
- Policy input: both images (no 180 degree rotation for these finetunes), 8-D state (EEF pos, axis-angle, 2 gripper qpos), task instruction.
- Execution: first 5 actions of each predicted chunk, then replan; 10 dummy-action settle steps
  (`[0]*6 + [-1]`) before the policy acts; cap 300 policy steps.
- Success: LIBERO env `done` (BDDL goal predicate); the episode ends at the first success ("ever achieved"). Timeouts,
  non-finite policy output (policy failure) and infrastructure errors are recorded separately.
- Native sampling distribution, eval mode; per-episode seed `oops_common.episode_seed(task, reset, delta, angle, policy_seed)`
  seeds numpy, torch and the JAX policy key, so it is the same integer for both arms (reproducibility only).
- Parity: `eval_spatial_shifts.py` vs native `eval_libero_intent.py --episode-seeding 0` on 2 unperturbed resets
  per arm (`parity_check.py`). Status: [RESOLVE — must PASS before the campaign].

## Perturbation protocol [code, sim]

- Tasks: the 8 LIBERO-Goal tasks whose instruction names a free-body object; moved body =
  that object (bowl, wine bottle, plate, cream cheese). Excluded by eligibility (no free-body target): "open the middle drawer of
  the cabinet", "turn on the stove". Task ids are resolved from instruction strings at runtime.
- Shift: `p_xy <- p_xy + delta [cos theta, sin theta]` on the object's free joint in the world frame (flattened state index
  `jnt_qposadr + 1`); z and quaternion unchanged; applied to the stored benchmark init state before `set_init_state`, then the
  native 10-step settle. The robot state, other objects and goal fixtures are untouched [sim: verified `perturb_state` changes
  exactly two entries].
- Grid: delta in {0, 0.05, 0.10} m; 10 fixed directions theta_k = 2 pi k / 10; benchmark resets 0, 1, 2; delta=0 once per reset
  -> 21 physical cases per reset, 63 per task, 504 total; one policy seed per case per arm.
- Goals in LIBERO-Goal are relational (e.g. `On(bowl, stove_cook_region)`) and never reference the moved object's original
  location [sim: parsed BDDL goal states].
- Relation to training data [sim, 500 demos]: every object's demonstrated start x and y each have std ~0.009 m and range
  ~±0.015 m around a fixed mean (the note claiming zero x-variance is **false**). At delta >= 0.05 m the moved object
  starts >= ~0.029 m from its demonstrated mean, outside the range of demonstrated start positions for that object. Other scene
  elements remain in distribution. Use "outside the demonstrated initial-position range", not a general OOD claim.

## Physical validity (decided before any policy runs) [code]

`build_case_manifest.py` builds every case, stores the exact init state + sha256, and classifies with simulator physics only
(`VALIDITY_RULES`, version `vv1`; the xy/z thresholds are inherited from the earlier harness defaults):

| rule | invalid | unknown |
|---|---|---|
| deepest contact penetration of the moved subtree at placement | > 5 mm | 2-5 mm |
| settled XY drift from requested XY (native settle) | > 0.04 m | 0.02-0.04 m |
| settled Z vs the same reset's unperturbed settle | drop or rise > 0.03 m ("fell" / "support changed") | 0.015-0.03 m |
| any other free object displaced vs unperturbed settle | > 0.02 m | 0.01-0.02 m |
| still moving between settle end and +40 dummy steps | > 0.01 m | — |
| goal predicate already satisfied after settle | yes | — |
| requested XY off the table-top footprint | yes | — |
| simulator exception during the pre-pass | — | always |

Reachability is recorded (distance to robot base) but never used to declare invalidity. Normal resting contact
measured at ~1e-5 m penetration [sim].

## Analysis [code]

Primary population: valid cases with both arms completed. SR per task and delta = successes / matched valid trials;
main curve = equal-weight macro-average over the tasks with valid matched data at every displayed delta (task set
printed in `summary.json`). Uncertainty: paired cluster bootstrap over benchmark reset ids within task (2,000 resamples).
With 3 resets per task this is labelled exploratory; intervals do not reflect training-seed variability (one training run per arm).
