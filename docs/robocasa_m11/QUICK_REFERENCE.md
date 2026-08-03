# M11 RoboCasa Eval — Quick Reference

## Files
- **Eval script**: `examples/openpi/eval_robocasa_intent.py` (434 lines)
- **Smoke test**: `examples/openpi/test_robocasa_eval_smoke.py` (206 lines)
- **Full report**: `docs/robocasa_m11/D3_eval_report.md`
- **Integration guide**: `docs/robocasa_m11/D3_INTEGRATION_GUIDE.md`

## Smoke Test (No RoboCasa Required)

```bash
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate
python examples/openpi/test_robocasa_eval_smoke.py
# Expected: ✓ All smoke tests PASSED
```

## Full Eval Examples (After G1 Confirms Env)

### Command Template
```bash
source .venv-robocasa/bin/activate
python examples/openpi/eval_robocasa_intent.py \
  --config-name <config> \
  --checkpoint-dir <path> \
  --task-set <set> \
  --schedule <schedule> \
  --out <json_path>
```

### Atomic Tasks, Baseline (No Intent)
```bash
python examples/openpi/eval_robocasa_intent.py \
  --config-name pi05_base \
  --checkpoint-dir /path/to/pi05_base \
  --task-set atomic_seen \
  --num-trials-per-task 5 \
  --out logs/robocasa_atomic_baseline.json
```

### Composite Tasks, Co-Prediction S1 (Intent-First)
```bash
python examples/openpi/eval_robocasa_intent.py \
  --config-name pi05_base_copred \
  --checkpoint-dir /path/to/pi05_copred/26000 \
  --task-set composite_seen \
  --schedule s1 \
  --num-trials-per-task 5 \
  --out logs/robocasa_composite_s1.json
```

### All Tasks, All Schedules (Full Experiment)
```bash
for schedule in s1 s2 s3; do
  python examples/openpi/eval_robocasa_intent.py \
    --config-name pi05_base_copred \
    --checkpoint-dir /path/to/pi05_copred/26000 \
    --schedule "$schedule" \
    --task-set all \
    --num-trials-per-task 3 \
    --out "logs/robocasa_all_${schedule}.json"
done
```

### Debug: Random Policy (Smoke Test with Real Env)
```bash
python examples/openpi/eval_robocasa_intent.py \
  --config-name pi05_base \
  --checkpoint-dir /tmp/fake \
  --tasks PickPlaceCounterToCabinet,TurnOnElectricKettle \
  --num-trials-per-task 2 \
  --random-policy \
  --out /tmp/smoke_test.json
```

## Task Sets

| Set | Count | Tasks |
|-----|-------|-------|
| atomic_seen | 4 | PickPlaceCounterToCabinet, PickPlaceCounterToStove, TurnOnElectricKettle, SlideDishwasherRack |
| composite_seen | 5 | KettleBoiling, LoadDishwasher, PrepareCoffee, PreSoakPan, WashLettuce |
| composite_unseen | 6 | ArrangeTea, CategorizeCondiments, CuttingToolSelection, PanTransfer, WashFruitColander, WeighIngredients |

## Schedules (M10 Co-Prediction)

- **s1**: Intent-first (4 intent steps → 10 action steps) — **the method**
- **s2**: Joint (shared noise, 10 steps) — baseline co-prediction
- **s3**: Action-first (10 action steps → 4 intent steps) — control

## Observation Format

**16D State** (in order):
```
[base_pos(3), base_quat(4), eef_pos_rel(3), eef_quat_rel(4), gripper_qpos(2)]
```

**2 Images** (256×256 RGB):
- `robot0_agentview_left` — base/left camera (slot 0)
- `robot0_eye_in_hand` — wrist camera (slot 1)

## Action Format

**12D Control** (policy outputs 32D, we slice [:12]):
```
[base_motion(4), mode(1), eef_pos(3), eef_rot(3), gripper(1)]
```

## Key Arguments

| Arg | Values | Default | Purpose |
|-----|--------|---------|---------|
| `--task-set` | atomic_seen, composite_seen, composite_unseen, all | — | Which tasks to eval |
| `--tasks` | CSV list | None | Override --task-set with specific tasks |
| `--schedule` | s1, s2, s3, None | None | M10 schedule; None = standard pi0.5 |
| `--num-trials-per-task` | int | 3 | Episodes per task |
| `--max-steps` | int | 500/1000 | Max episode length (per-set default) |
| `--intent` | flag | False | Enable MIP flow-map intent sidecar |
| `--vl-cotrain` | flag | False | Enable VL co-trained intent |
| `--fp32` | flag | False | Force float32 (pi05_base BF16 overflow fix) |
| `--random-policy` | flag | False | Random actions (smoke test only) |
| `--seed` | int | 7 | RNG seed |

## JSON Output

Each eval produces JSON with:
```json
{
  "config": "pi05_base_copred",
  "checkpoint": "/path/to/checkpoint",
  "task_set": "atomic_seen",
  "num_tasks": 4,
  "num_trials_per_task": 5,
  "mean_sr": 0.75,
  "total": "15/20",
  "per_task": {
    "PickPlaceCounterToCabinet": 0.8,
    "PickPlaceCounterToStove": 1.0,
    "TurnOnElectricKettle": 0.6,
    "SlideDishwasherRack": 0.6
  },
  "schedule": "s1"
}
```

Use per_task dict for granular analysis; mean_sr for headline numbers.

## Next Steps (After G1 Confirms)

1. Update `RoboCasaEnvAdapter` if obs keys or success API differ
2. Run smoke test with real env: `--random-policy --tasks <one_task> --num-trials-per-task 2`
3. Run full eval across all 3 schedules (s1, s2, s3) and all task sets
4. Aggregate results into `docs/co_prediction_results.md` section for RoboCasa
5. Compare vs LIBERO baseline

---

**Status**: ✓ Complete and smoke-tested  
**Ready for**: G1 env verification and integration  
**Pending**: `docs/robocasa_m11/G1_env_report.md` with confirmed adapter assumptions
