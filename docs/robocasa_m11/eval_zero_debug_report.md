# M11 RoboCasa A1 Co-Prediction Eval Zero Debug Report

**Date**: 2026-08-04  
**Status**: Investigation in progress  
**Problem**: Phase-A eval shows 0/900 success across ALL tasks/schedules/checkpoints, despite healthy training loss (~0.015)  

---

## Executive Summary

A trained A1 (tied) checkpoint from 30k training steps with decreasing BC loss (final ~0.015) scores 0% on all RoboCasa tasks in evaluation. The pattern suggests **train/eval mismatch** not model quality.

**Key observations from eval logs**:
- Action dims 0-3 (base motion): consistently output ~0 (normalized)
- Action dim 4 (control_mode): consistently output -0.999 (normalized)
- Action absmax: ~1.0 (within valid normalized range)
- Prompt: correct, derived from task names ("turn on electric kettle")

---

## Hypothesis Summary (Prioritized)

### 1. Quantile Normalization Mismatch (CRITICAL - investigate first)
**Status**: Partially ruled out by checkpoint asset check

The V1 integration report documented Bug #10: degenerate quantile ranges (q01≈q99) for action dims 0-4 causing loss spikes during smoke tests. A fix was applied by widening quantile ranges to ±1 around midpoint in `norm_stats.json`.

**Finding**: 
- Checkpoint at 30000/ has its own `assets/robocasa/norm_stats.json` 
- It contains the POST-qfix version (widened quantiles)
- Matches current in-repo `assets/pi05_robocasa_copred/robocasa/norm_stats.json`
- Therefore training and eval use IDENTICAL norm_stats

**Remaining concern**: Even though versions match, the quantile ranges themselves may still be wrong:
- Action dim 4: q01=-2.0, q99=0.0 (unusual range for binary control_mode)
- Action dims 0-3: q01=-1.0, q99=1.0 (corresponds to base_motion)

### 2. Double Normalization / Unnormalization Dance (MEDIUM - likely innocent)

**Finding**:
- Training pipeline: dataset applies z-score norm → training **unnormalizes** → openpi Normalize (quantile) applies
- Eval pipeline: raw obs from env → openpi Policy (should apply quantile norm internally)

The training explicitly unnormalizes to match openpi's transform stack. Eval should do the same, but unclear if it does.

### 3. Action Slicing / Dimension Mismatch (LOW - checked)

- Policy outputs 32D actions padded
- Eval correctly slices first 12D before env.step()
- Confirmed in eval_robocasa_intent.py line 484

### 4. Obs Construction / Keys (MEDIUM - spotted in addendum)

V1 Integration Report Addendum (Bug #6) fixed eval obs keys:
- Incorrect: `robot0_eef_pos_rel/quat_rel` (don't exist)
- Correct: `robot0_base_to_eef_pos/quat`

Current eval_robocasa_intent.py uses correct keys (line 169-170). But image flipping issue was noted (OpenGL convention).

---

## Diagnostic Plan

### Tier 1: GT-Action Replay Test (HIGHEST PRIORITY)
**Why**: Definitively separates env/controller bugs from policy bugs

- **Test**: Load 3 demo episodes from LeRobot dataset
- **Method**: Reset env, replay raw demo actions open-loop
- **Expected**: If GT actions fail → bug is in env/controller wiring (not policy)
                If GT actions succeed → bug is in policy-side (obs, norm, or model)

**Script**: `examples/openpi/debug_gt_action_replay.py`

### Tier 2: Normalization Audit
**Why**: Confirm which norm_stats file is loaded and applied

**Script**: `examples/openpi/debug_norm_audit.py`
- Traces config → asset_id → norm_stats loading
- Checks for degenerate quantile ranges
- Compares actual vs. stored stats

### Tier 3: Observation Construction
**Why**: Verify obs dict keys, shapes, dtypes match expectations

**Script**: `examples/openpi/debug_obs_construction.py`
- Dumps raw obs from env.reset()
- Checks all required keys present
- Validates state/image shapes and dtypes

### Tier 4: Training Data Sampling
**Why**: Verify actual demo action ranges match quantile stats

**Script**: `examples/openpi/debug_training_data_sample.py`
- Samples actual action values from LeRobot parquet files
- Computes per-dim quantiles
- Compares against norm_stats.json q01/q99

---

## Current File State

### Norm Stats Versions
**File**: `/home/ldahiya/max_vla/much-ado-about-noising/assets/pi05_robocasa_copred/robocasa/norm_stats.json`
- Updated: 2026-07-31 03:00
- Version: POST-qfix (widened quantiles)
- Action dims 0-3: q01=-1.0, q99=1.0
- Action dim 4: q01=-2.0, q99=0.0

**Backup**: `norm_stats.json.pre_qfix`
- Pre-qfix version (degenerate quantiles)
- Action dims 0-3: q01=0.0, q99=0.0 (constant!)
- Action dim 4: q01=-1.0, q99=-1.0 (constant!)

**Checkpoint**: `/data/group_data/maxlab/common_datasets/pandaliza/maxvla/m11_checkpoints/m11_a1_tied/30000/assets/robocasa/norm_stats.json`
- Matches POST-qfix version
- Suggests training occurred AFTER qfix was applied (Aug 2-3)

### Config Files
**Training config**: `pi05_robocasa_copred`
- Defined in `external/openpi/src/openpi/training/config.py` line 951
- Uses `asset_id="robocasa"`
- Data config: `LeRobotLiberoDataConfig`
- Uses quantile normalization (pi05 is not PI0, so `use_quantile_norm=True`)

**Eval script**: `examples/openpi/eval_robocasa_intent.py`
- Loads policy via `create_trained_policy(..., norm_stats=None)`
- Falls back to checkpoint's `checkpoint_dir/assets/robocasa/norm_stats.json`
- Uses the same post-qfix version

---

## Key Code Paths

### Training normalization (train_pi05_m11.py)
```python
# Line 239-242: build_observation
st = ds.normalizer["obs"]["state"].unnormalize(batch["obs"]["state"].numpy())
act = ds.normalizer["action"].unnormalize(batch["action"].numpy())
# Then applied to openpi transforms.Normalize (quantile norm)
```

### Eval obs construction (eval_robocasa_intent.py)
```python
# Line 458-465: extract_state_obs
state = np.concatenate([
    obs["robot0_base_pos"],           # 3D
    obs["robot0_base_quat"],          # 4D
    obs["robot0_base_to_eef_pos"],    # 3D
    obs["robot0_base_to_eef_quat"],   # 4D
    obs["robot0_gripper_qpos"],       # 2D
])  # → 16D

# Action applied directly to policy.infer()
action_chunk = np.asarray(policy.infer(element)["actions"])
# Should be (replan_steps=5, 12)
```

---

## Unexplained Phenomena

1. **Why is action dim 4 pinned at -0.999?**
   - This is the normalized output from the model
   - Unnormalizes to approx -2.0 (near q01=-2.0 lower bound)
   - Suggests model learned to output "minimum control_mode"
   - But training should have learned diverse values if demos varied

2. **Why are dims 0-3 all ~0?**
   - Normalized 0 maps to unnormalized 0 (midpoint of q01=-1, q99=1)
   - But base_motion should vary to solve tasks
   - Possible mode collapse: policy learned to output mean/median across all tasks

3. **What is action dim 4 really?**
   - Docs say "control_mode 4:5" — binary flag
   - But quantile stats show range [-2.0, 0.0]
   - This doesn't match binary semantics

---

## Next Steps

1. **RUN: GT-action replay test**
   - Verdict will determine if env/policy/both are broken

2. **IF GT test fails**: 
   - Investigate env controller wiring
   - Check action_dim interpretation
   - Verify action scaling/clipping

3. **IF GT test succeeds**:
   - Investigate obs normalization
   - Check if policy sees correct obs structure
   - Trace quantile normalization in eval path

4. **Sanity checks**:
   - Confirm norm_stats loading in eval
   - Print one obs/action through both pipelines
   - Verify action unnormalization logic

---

## Files to Check/Run

### Diagnostic scripts (created 2026-08-04)
- `examples/openpi/debug_gt_action_replay.py` — GT action replay
- `examples/openpi/debug_norm_audit.py` — Norm stats audit
- `examples/openpi/debug_obs_construction.py` — Obs structure dump
- `examples/openpi/debug_training_data_sample.py` — Training data histogram

### Related docs
- `docs/robocasa_m11/V1_integration_report.md` — W2 defect history
- `docs/robocasa_m11/D3_eval_report.md` — Eval design spec
- `external/openpi/src/openpi/training/config.py` — Config definition

---

## Report Status

- [x] File inventory completed
- [x] Norm stats versions reconciled
- [x] Config paths verified
- [ ] GT-action replay test (NEXT)
- [ ] Training data sampling
- [ ] Obs construction verification
- [ ] Root cause identified

**Estimated completion**: After GT-action replay test verdict

