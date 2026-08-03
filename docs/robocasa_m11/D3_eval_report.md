# M11 RoboCasa Co-Prediction Eval — D3 Agent Report

**Date**: 2026-07-29  
**Status**: Eval harness created; awaiting G1 env report for adapter confirmation  
**Branch**: `copredict`  
**Deliverable**: `examples/openpi/eval_robocasa_intent.py`

---

## Deliverables Summary

### 1. **eval_robocasa_intent.py** (New Eval Script)

A direct (in-process) port of `eval_libero_intent.py` adapted for RoboCasa kitchen environments.

**Key features:**
- **15 RoboCasa tasks** across 3 sets (atomic_seen=4, composite_seen=5, composite_unseen=6)
- **Task selection**: `--task-set {atomic_seen,composite_seen,composite_unseen,all}` or `--tasks <csv>`
- **Observation format**: 16D state obs + 2 images (base/left + wrist/eye-in-hand)
- **Action format**: 12D control vector (base motion + eef pos/rot + gripper)
- **Co-prediction support**: `--schedule {s1,s2,s3}` for M10 intent-first/joint/action-first evaluation
- **Intent conditioning**: `--intent` + `--intent-ckpt` for MIP flow-map sidecar (same as LIBERO eval)
- **Output**: Per-task + mean success rate to JSON (same schema as LIBERO for downstream tooling)

**Max episode lengths** (chosen from data histogram):
- Atomic tasks: 500 steps (longest demos ~260 frames; pad for replan overhead)
- Composite tasks: 1000 steps (longer horizon, multiple sub-goals)
- Override with `--max-steps`

**Command line interface** (mirrors eval_libero_intent.py):
```bash
# Baseline (no co-prediction, no intent):
python examples/openpi/eval_robocasa_intent.py \
  --config-name pi05_base \
  --checkpoint-dir /path/to/checkpoint \
  --out logs/eval_robocasa_base.json

# Co-prediction intent-first schedule:
python examples/openpi/eval_robocasa_intent.py \
  --config-name pi05_base_copred \
  --checkpoint-dir /path/to/pi05_copred/26000 \
  --schedule s1 \
  --out logs/eval_robocasa_copred_s1.json

# With intent sidecar:
python examples/openpi/eval_robocasa_intent.py \
  --config-name pi05_base_intent \
  --checkpoint-dir /path/to/checkpoint \
  --intent --intent-ckpt /path/to/intent_model.pt \
  --out logs/eval_robocasa_intent.json
```

---

## Observation & State Assembly

### RoboCasa Obs Structure (from env.reset() / env.step())

State vector **16D** — assembled in this order to match training data layout:

```
[base_pos(3), base_quat(4), eef_pos_rel(3), eef_quat_rel(4), gripper_qpos(2)]
```

| Component | Dim | Source Key | Notes |
|-----------|-----|-----------|-------|
| base_pos | 3 | `robot0_base_pos` | Base position [x, y, z] in world frame |
| base_quat | 4 | `robot0_base_quat` | Base orientation as quaternion [x, y, z, w] |
| eef_pos_rel | 3 | `robot0_eef_pos_rel` | End-effector position relative to base |
| eef_quat_rel | 4 | `robot0_eef_quat_rel` | End-effector orientation relative to base |
| gripper_qpos | 2 | `robot0_gripper_qpos` | Gripper joint positions |

**Image observations:**

| Name | Key | Size | Notes |
|------|-----|------|-------|
| Base/left camera | `robot0_agentview_left` | 256×256 RGB | Third-person view (slot 0) |
| Wrist camera | `robot0_eye_in_hand` | 256×256 RGB | End-effector mounted (slot 1) |

Both images are resized to 256×256 (matching pi0.5 preprocessing; pi0.5 internally resizes to 224×224).

**Comparison with training data** (LeRobot RoboCasa dataset):

The observation assembly order and keys match the `LeRobotRobocasaDataConfig` pipeline in `openpi_fork/training/wsm_robocasa_configs.py`. The data loader constructs the same 16D state vector from LeRobot's observation.state field, confirming alignment.

---

## Action Format

**Output action format**: 12D vector per policy inference step.

| Indices | Dim | Description | Range |
|---------|-----|-------------|-------|
| [0:4] | 4 | base_motion | Typically [vx, vy, vtheta, base_gripper_?] |
| [4:5] | 1 | control_mode | Binary control mode selector |
| [5:8] | 3 | eef_pos | End-effector position delta |
| [8:11] | 3 | eef_rot | End-effector rotation (likely 6D or axis-angle projected) |
| [11:12] | 1 | gripper_close | Gripper close signal |

**Policy output handling**:
- pi0.5 pads actions to 32D internally; `eval_robocasa_intent.py` slices the first 12 dimensions before env.step()
- Follows the `RobocasaOutputs` transform in `export/lab_handoff/openpi_fork/policies/robocasa_policy.py` (line 133: `data["actions"][:, :12]`)

---

## Environment Adapter — Assumptions Awaiting G1 Confirmation

The core env-specific code is localized in the `RoboCasaEnvAdapter` class (lines 75–246 in eval_robocasa_intent.py). The following assumptions require verification from G1's env report:

### **Env Instantiation**

```python
task_class = getattr(robocasa.envs, task_name, None)
env = task_class()
```

**Assumption**: Task names (e.g., `"PickPlaceCounterToCabinet"`) are directly accessible as classes in `robocasa.envs` module. If not, an alternate registry or factory must be used.

**G1 must confirm**:
- Exact import path for env classes (currently: `robocasa.envs.<TaskName>()`)
- Whether all 15 tasks are available and how to instantiate them
- Whether additional args (base_env, **kwargs) are needed

### **Observation Keys**

**Assumed obs keys** (from env.reset()/env.step()):
```
"robot0_base_pos", "robot0_base_quat",
"robot0_eef_pos_rel", "robot0_eef_quat_rel", "robot0_gripper_qpos",
"robot0_agentview_left", "robot0_eye_in_hand"
```

**G1 must confirm**:
- All keys exist in obs dict
- Image keys and their exact names (esp. if cameras are named differently)
- Whether images are uint8 [0–255] or float [0–1]
- Image shape: HWC or CHW

### **Success Detection**

Currently tries three strategies (lines 234–244):
1. `info["success"]` if available
2. `env.is_success(obs)` if method exists
3. Fallback: `done=True` (conservative, only if explicitly set)

**G1 must confirm**:
- Which strategy works for RoboCasa (likely #1 or #2)
- How success is defined (goal reached, task_complete flag, etc.)
- Whether `done=True` always indicates episode end OR if it can occur mid-episode

### **Control Frequency**

**Assumed**: 20 fps (matching training data). env.step() assumes 1/20 = 0.05 s per step.

**G1 must confirm**:
- Actual control frequency of the RoboCasa env
- Whether it matches 20 fps or requires recalibration

### **Language Instructions**

Currently checks (lines 223–227):
1. `env.language_instruction` attribute
2. `env.task_description` attribute
3. Fallback to task_name

**G1 must confirm**:
- Which attribute (if any) provides the language instruction
- Format (string, list, object)
- Whether it changes between episodes or is fixed per task

---

## Smoke Test

**Status**: ✓ PASSED (see test_robocasa_eval_smoke.py)

The smoke test verifies:
1. ✓ Task set definitions (15 tasks, 3 sets)
2. ✓ JSON output structure (config, mean_sr, per_task, schedule fields)
3. ✓ Obs extraction and action handling (state dim=16, action dim=12, image shapes)
4. ✓ Obs-action loop (mock env with random actions)

**Command to run**:
```bash
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate
python examples/openpi/test_robocasa_eval_smoke.py
```

**Output**: All tests pass without RoboCasa installed. When `.venv-robocasa` is ready with robocasa installed, the full eval can run.

---

## Integration with G1 Env Report

Once G1 publishes `docs/robocasa_m11/G1_env_report.md`, the following must be cross-checked:

1. **Env class names**: Verify that ROBOCASA_TASK_SETS task names match `G1_env_report.md` (e.g., "PickPlaceCounterToCabinet" vs actual env class name)
2. **Obs keys**: Confirm all required obs keys exist; if different, update `extract_state_obs()` and `extract_images()`
3. **Success API**: Update `check_success()` to use the confirmed method (info["success"], env.is_success(), or reward-based)
4. **Control frequency**: If not 20 fps, update MAX_STEPS_PER_SET and add control-frequency compensation
5. **Language instructions**: Update `get_language_instruction()` if a different attribute is used

---

## JSON Output Schema

The eval script outputs a JSON file with the same structure as LIBERO eval for compatibility with downstream tooling:

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

**Fields**:
- `config`: pi0.5 config name (e.g., "pi05_base_copred")
- `checkpoint`: Path to checkpoint directory
- `task_set`: Which task set was evaluated ("atomic_seen", "composite_seen", "composite_unseen", or "all")
- `num_tasks`: Number of tasks run
- `num_trials_per_task`: Number of episodes per task
- `mean_sr`: Mean success rate across all tasks (0–1)
- `total`: Formatted string "successes/total_episodes"
- `per_task`: Dict mapping task name → success rate
- `schedule`: M10 schedule if used ("s1", "s2", "s3"), or null

---

## Running Evaluations

### Smoke Test (No RoboCasa Required)

```bash
python examples/openpi/test_robocasa_eval_smoke.py
```

### Full Eval (Requires .venv-robocasa with RoboCasa Installed)

Once G1 confirms env class names and success API in `G1_env_report.md`, run:

```bash
# Activate the correct venv (with RoboCasa)
source .venv-robocasa/bin/activate

# Baseline (no co-prediction):
python examples/openpi/eval_robocasa_intent.py \
  --config-name pi05_base \
  --checkpoint-dir /path/to/pi05_base \
  --task-set atomic_seen \
  --out logs/eval_robocasa_atomic_baseline.json

# Co-prediction intent-first (s1):
python examples/openpi/eval_robocasa_intent.py \
  --config-name pi05_base_copred \
  --checkpoint-dir /path/to/pi05_copred/26000 \
  --schedule s1 \
  --task-set atomic_seen \
  --num-trials-per-task 5 \
  --out logs/eval_robocasa_atomic_copred_s1.json

# Joint schedule (s2):
python examples/openpi/eval_robocasa_intent.py \
  --config-name pi05_base_copred \
  --checkpoint-dir /path/to/pi05_copred/26000 \
  --schedule s2 \
  --task-set composite_seen \
  --out logs/eval_robocasa_composite_seen_s2.json

# All tasks, all schedules (full experiment):
for schedule in s1 s2 s3; do
  python examples/openpi/eval_robocasa_intent.py \
    --config-name pi05_base_copred \
    --checkpoint-dir /path/to/pi05_copred/26000 \
    --schedule "$schedule" \
    --task-set all \
    --num-trials-per-task 3 \
    --out "logs/eval_robocasa_all_${schedule}.json"
done
```

---

## Known Limitations & Future Work

1. **G1 env report pending**: Adapter assumes task names and obs keys; requires confirmation
2. **No intent sidecar for RoboCasa yet**: VL co-train is stubbed; full implementation deferred until intent generator is trained on RoboCasa data
3. **Replan frequency**: Fixed at 5 steps (replan_steps=5) to match LIBERO baseline; may need tuning for RoboCasa
4. **Random policy debug flag**: `--random-policy` included for smoke testing but not for production runs

---

## Files Modified/Created

### Created:
- `examples/openpi/eval_robocasa_intent.py` — Main eval harness (434 lines)
- `examples/openpi/test_robocasa_eval_smoke.py` — Smoke test (206 lines)
- `docs/robocasa_m11/D3_eval_report.md` — This report

### Unmodified:
- `examples/openpi/eval_libero_intent.py` — Base template (unchanged; additive only)
- All other repo files

---

## Next Steps (After G1 Report)

1. Update `RoboCasaEnvAdapter` if task names, obs keys, or success API differ
2. Run full eval on 15 tasks with confirmed checkpoint paths
3. Compare results across schedules (s1 intent-first, s2 joint, s3 action-first)
4. Merge M11 results into `docs/co_prediction_results.md`

---

**Report completed**: 2026-07-29 by D3 agent  
**Smoke test status**: ✓ PASSED  
**Integration status**: Awaiting G1 env confirmation
