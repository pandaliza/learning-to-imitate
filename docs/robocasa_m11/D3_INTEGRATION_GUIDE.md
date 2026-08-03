# M11 RoboCasa Eval Integration Guide

**For**: Integration agent G1  
**From**: D3 (eval harness agent)  
**Date**: 2026-07-29  
**Status**: Eval script complete and smoke-tested; ready for RoboCasa env verification

---

## What Was Delivered

### 1. Main Eval Script
**File**: `examples/openpi/eval_robocasa_intent.py` (434 lines)

A complete port of `eval_libero_intent.py` for RoboCasa kitchen environments. Supports:
- 15 tasks across 3 sets (4 atomic_seen, 5 composite_seen, 6 composite_unseen)
- M10 co-prediction schedules (s1 intent-first, s2 joint, s3 action-first)
- Intent sidecar from MIP flow map (via --intent)
- VL co-training intent (via --vl-cotrain)
- JSON output compatible with LIBERO eval results

### 2. Smoke Test
**File**: `examples/openpi/test_robocasa_eval_smoke.py` (206 lines)

Verifies the eval harness without requiring RoboCasa installed:
- ✓ Task set definitions (15 tasks, 3 sets)
- ✓ JSON output structure
- ✓ Obs extraction (16D state, images)
- ✓ Action handling (12D control)

**Status**: ✓ PASSED on main .venv

### 3. Documentation
**File**: `docs/robocasa_m11/D3_eval_report.md`

Detailed specification of:
- Observation assembly order (16D state + 2 images)
- Action format (12D: base[4] + mode[1] + eef[3] + rot[3] + grip[1])
- Environment adapter assumptions awaiting your confirmation
- JSON output schema
- Integration checklist

---

## What Needs Your Confirmation (G1)

The eval script has one localized env-specific class: `RoboCasaEnvAdapter` (lines 75–246). This depends on:

### **1. Task Instantiation** (Lines 97–113)
```python
task_class = getattr(robocasa.envs, task_name, None)
env = task_class()
```

**To confirm**:
- [ ] Can we instantiate tasks like `robocasa.envs.PickPlaceCounterToCabinet()`?
- [ ] If not, what's the correct import/factory?
- [ ] Do any tasks require additional args (e.g., base_env=None)?
- [ ] Document in G1_env_report.md the exact instantiation syntax

### **2. Observation Keys** (Lines 165–195)
Expected keys in obs dict (from env.reset()/env.step()):
```
"robot0_base_pos", "robot0_base_quat",
"robot0_eef_pos_rel", "robot0_eef_quat_rel", "robot0_gripper_qpos",
"robot0_agentview_left", "robot0_eye_in_hand"
```

**To confirm**:
- [ ] All 7 keys exist?
- [ ] Are images uint8 [0–255] or float [0–1]?
- [ ] Are images HWC or CHW?
- [ ] Do camera names match exactly? (esp. agentview_left vs agentview vs other names)

### **3. Success Detection** (Lines 234–244)
Currently tries (in order):
1. `info["success"]` if present
2. `env.is_success(obs)` if method exists
3. Fallback to `done=True` (conservative)

**To confirm**:
- [ ] Which strategy works? (likely #1 or #2)
- [ ] How is success defined?
- [ ] Can done=True occur mid-episode?

### **4. Control Frequency** (Line 47 comment)
Assumed 20 fps (matching training data).

**To confirm**:
- [ ] Is RoboCasa env at 20 fps or different?
- [ ] If different, MAX_STEPS_PER_SET needs adjustment

### **5. Language Instructions** (Lines 223–227)
Currently checks:
1. `env.language_instruction`
2. `env.task_description`

**To confirm**:
- [ ] Which attribute (if any)?
- [ ] String or object?
- [ ] Same across episodes or per-episode?

---

## Validation Checklist

Once you've verified the above, run this checklist:

### Phase 1: Environment Setup
- [ ] Install robocasa into `.venv-robocasa`
- [ ] Verify all 15 task classes are available
- [ ] Confirm obs keys and image formats
- [ ] Test a single reset/step cycle manually

### Phase 2: Adapter Fixes (if needed)
- [ ] If obs keys differ, update `extract_state_obs()` (line 195)
- [ ] If camera names differ, update `extract_images()` (line 206)
- [ ] If success detection differs, update `check_success()` (line 234)
- [ ] If frequency differs, adjust `MAX_STEPS_PER_SET` (line 39)

### Phase 3: Smoke Test with Real Env
```bash
source .venv-robocasa/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

# Run 2 trials on one atomic task with random actions (debug mode):
python examples/openpi/eval_robocasa_intent.py \
  --config-name pi05_base \
  --checkpoint-dir /fake/path \
  --tasks PickPlaceCounterToCabinet \
  --num-trials-per-task 2 \
  --random-policy \
  --out /tmp/smoke_test.json
```

**Expected**:
- Script runs without errors
- Episodes complete (100 steps max)
- JSON file created with structure: config, mean_sr, per_task, schedule

### Phase 4: Full Eval with Checkpoint
Once you have a pi05_base or pi05_base_copred checkpoint:

```bash
# Baseline (no co-prediction):
python examples/openpi/eval_robocasa_intent.py \
  --config-name pi05_base \
  --checkpoint-dir /path/to/checkpoint \
  --task-set atomic_seen \
  --num-trials-per-task 3 \
  --out logs/eval_robocasa_atomic_baseline.json

# Co-prediction s1 (intent-first):
python examples/openpi/eval_robocasa_intent.py \
  --config-name pi05_base_copred \
  --checkpoint-dir /path/to/copred_ckpt/26000 \
  --schedule s1 \
  --task-set atomic_seen \
  --out logs/eval_robocasa_atomic_s1.json
```

---

## Expected JSON Output

Each eval run produces a JSON file like:

```json
{
  "config": "pi05_base_copred",
  "checkpoint": "/path/to/checkpoint",
  "task_set": "atomic_seen",
  "num_tasks": 4,
  "num_trials_per_task": 3,
  "mean_sr": 0.667,
  "total": "4/6",
  "per_task": {
    "PickPlaceCounterToCabinet": 0.667,
    "PickPlaceCounterToStove": 0.667,
    "TurnOnElectricKettle": 0.333,
    "SlideDishwasherRack": 1.0
  },
  "schedule": "s1"
}
```

Use this to:
1. Track per-task success rates
2. Compare schedules (s1 vs s2 vs s3)
3. Aggregate results into `docs/co_prediction_results.md`

---

## Reporting Back to D3

Once you've verified and tested, create `docs/robocasa_m11/G1_env_report.md` with:

1. **Env instantiation**: Exact import/factory call for each task
2. **Obs keys**: Confirmed keys, dtypes, shapes
3. **Success API**: Which strategy works + how to call it
4. **Control frequency**: Actual fps; any needed adjustments
5. **Language instructions**: Attribute name + format
6. **Any env quirks**: Edge cases, initialization gotchas, etc.
7. **Code changes needed** (if any): Exact lines in eval_robocasa_intent.py to fix

Then D3 will apply those fixes and mark the eval as production-ready.

---

## Files Modified/Created

### Created (Additive Only):
- ✓ `examples/openpi/eval_robocasa_intent.py`
- ✓ `examples/openpi/test_robocasa_eval_smoke.py`
- ✓ `docs/robocasa_m11/D3_eval_report.md`
- ✓ `docs/robocasa_m11/D3_INTEGRATION_GUIDE.md` (this file)

### Unchanged (No modifications):
- `examples/openpi/eval_libero_intent.py`
- All other repo files

---

## Quick Start (After You Confirm)

```bash
# 1. Activate RoboCasa venv
source .venv-robocasa/bin/activate

# 2. Run smoke test (2 trials, random actions)
python examples/openpi/eval_robocasa_intent.py \
  --config-name pi05_base \
  --checkpoint-dir /tmp/fake \
  --task-set atomic_seen \
  --num-trials-per-task 2 \
  --random-policy \
  --out /tmp/smoke.json && cat /tmp/smoke.json

# 3. Full eval (when checkpoint ready)
python examples/openpi/eval_robocasa_intent.py \
  --config-name pi05_base_copred \
  --checkpoint-dir /path/to/pi05_copred/26000 \
  --schedule s1 \
  --task-set all \
  --num-trials-per-task 5 \
  --out logs/robocasa_s1_full.json
```

---

## Contact / Questions

If any assumption is wrong or adapter code needs fixes after G1's testing, D3 can apply them quickly once you document the issue in `G1_env_report.md`.

**Status**: Ready for integration. Waiting on G1's env confirmation. 🚀
