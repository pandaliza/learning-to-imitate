# Phase-A Evaluation Grid Status

**Submission Date:** 2026-08-03  
**Checkpoint Base:** `/data/group_data/maxlab/common_datasets/pandaliza/maxvla/m11_checkpoints/m11_a1_tied_robocasa/`  
**Branch:** `copredict`  
**Status:** 9 jobs submitted, awaiting completion

---

## Submitted Jobs

### Grid 1: A1@30k × s2 (native schedule for tied model) × all task sets

| Job ID | Arm | Checkpoint | Schedule | Task Set | Tasks | Trials | Output JSON | Expected Path |
|--------|-----|-----------|----------|----------|-------|--------|-------------|----------------|
| **9707957** | A1 | 30k | s2 | atomic_seen | 4 | 20 | `eval_m11_a1_30k_s2_atomic_seen.json` | `logs/eval_m11_a1_30k_s2_atomic_seen.json` |
| **9707958** | A1 | 30k | s2 | composite_seen | 5 | 20 | `eval_m11_a1_30k_s2_composite_seen.json` | `logs/eval_m11_a1_30k_s2_composite_seen.json` |
| **9707959** | A1 | 30k | s2 | unseen (subset) | 2 (ArrangeTea, PanTransfer) | 20 | `eval_m11_a1_30k_s2_unseen.json` | `logs/eval_m11_a1_30k_s2_unseen.json` |

**Grid 1 Notes:**
- Tests Claim 1 (decoupling separates plan from execution) and Claim 5a (base-policy transfer on unseen tasks)
- s2 is the joint schedule where intent and action share noise (M10 co-prediction native)
- Unseen tasks limited to ArrangeTea and PanTransfer per plan §W1.2 (early kill-risk check, LIBERO precedent)

---

### Grid 2: A1@30k × s1 (intent-first schedule) × seen task sets only

| Job ID | Arm | Checkpoint | Schedule | Task Set | Tasks | Trials | Output JSON | Expected Path |
|--------|-----|-----------|----------|----------|-------|--------|-------------|----------------|
| **9707960** | A1 | 30k | s1 | atomic_seen | 4 | 20 | `eval_m11_a1_30k_s1_atomic_seen.json` | `logs/eval_m11_a1_30k_s1_atomic_seen.json` |
| **9707961** | A1 | 30k | s1 | composite_seen | 5 | 20 | `eval_m11_a1_30k_s1_composite_seen.json` | `logs/eval_m11_a1_30k_s1_composite_seen.json` |

**Grid 2 Notes:**
- s1: intent-first schedule (4 intent steps → 10 action steps); the method schedule per M10 spec
- Skips unseen tasks per plan — intent-first benefits primarily measured on seen task distributions
- Allows composition comparison: atomic vs composite within the intent-first frame

---

### Grid 3: A1@30k × s3 (action-first schedule) × seen task sets only

| Job ID | Arm | Checkpoint | Schedule | Task Set | Tasks | Trials | Output JSON | Expected Path |
|--------|-----|-----------|----------|----------|-------|--------|-------------|----------------|
| **9707962** | A1 | 30k | s3 | atomic_seen | 4 | 20 | `eval_m11_a1_30k_s3_atomic_seen.json` | `logs/eval_m11_a1_30k_s3_atomic_seen.json` |
| **9707963** | A1 | 30k | s3 | composite_seen | 5 | 20 | `eval_m11_a1_30k_s3_composite_seen.json` | `logs/eval_m11_a1_30k_s3_composite_seen.json` |

**Grid 3 Notes:**
- s3: action-first schedule (10 action steps → 4 intent steps); control for schedule effects
- s1 vs s3 comparison tests whether intent predicts future behavior or merely follows actions
- Skips unseen per plan

---

### Grid 4: A1@20k × s2 (checkpointing for a learning curve) × seen task sets

| Job ID | Arm | Checkpoint | Schedule | Task Set | Tasks | Trials | Output JSON | Expected Path |
|--------|-----|-----------|----------|----------|-------|--------|-------------|----------------|
| **9707964** | A1 | 20k | s2 | atomic_seen | 4 | 20 | `eval_m11_a1_20k_s2_atomic_seen.json` | `logs/eval_m11_a1_20k_s2_atomic_seen.json` |
| **9707965** | A1 | 20k | s2 | composite_seen | 5 | 20 | `eval_m11_a1_20k_s2_composite_seen.json` | `logs/eval_m11_a1_20k_s2_composite_seen.json` |

**Grid 4 Notes:**
- Uses 20k-step checkpoint (earlier in training) to construct a learning curve for s2
- Only seen task sets per plan
- Will compare against 30k to estimate SR slope over training progress

---

## Task Set Definitions

| Set | Type | Count | Tasks |
|-----|------|-------|-------|
| **atomic_seen** | Seen, single-step | 4 | PickPlaceCounterToCabinet, PickPlaceCounterToStove, TurnOnElectricKettle, SlideDishwasherRack |
| **composite_seen** | Seen, multi-step | 5 | KettleBoiling, LoadDishwasher, PrepareCoffee, PreSoakPan, WashLettuce |
| **unseen (subset)** | Unseen, multi-step | 2 | ArrangeTea, PanTransfer |

Total unique tasks evaluated: **4 + 5 + 2 = 11 tasks**  
Total episodes across all 9 grid cells: **9 cells × mean ~80–120 episodes/cell = ~900 episodes**

---

## Monitoring and Completion

### Check Job Status
```bash
squeue -u ldahiya -l
squeue -j 9707957  # Check specific job
```

### Monitor Progress
```bash
# Watch all eval jobs in real-time
watch -n 5 'squeue -u ldahiya | grep eval-m11'

# Stream logs from a running job (replace JOBID)
tail -f logs/eval_m11_JOBID.out
```

### Expected Wall-Clock Time per Job
- **Atomic tasks (4 tasks × 20 trials = 80 episodes):** ~10–15 minutes (if ~8–10 sec/episode)
- **Composite seen (5 tasks × 20 trials = 100 episodes):** ~15–20 minutes
- **Unseen/composite mix (2 tasks × 20 trials = 40 episodes):** ~8–10 minutes

**Total estimated wall-clock (all 9 jobs in parallel on separate GPUs):** ~20 minutes  
**Total SLURM queue time:** Depends on GPU availability; typically 5–30 min wait per job

---

## Result Collection

Once jobs complete:

```bash
# Verify all output JSONs are present
ls -lh logs/eval_m11_a1_*.json

# Run the collector script to aggregate results
python scripts/collect_phaseA.py

# This produces:
#   - Aggregated markdown table to stdout and file
#   - Throughput metrics from .out logs
#   - Per-cell means and 95% binomial CIs
#   - Atomic vs composite split
#   - Seen vs unseen split
```

See `scripts/collect_phaseA.py` for full documentation.

---

## SLURM Submission Details

**SBATCH Template Used:**  
```bash
slurm-scripts/eval/eval_m11_robocasa.sbatch
```

**Environment Variables Passed per Cell:**
```bash
CKPT=<checkpoint_dir>       # Full path to 30k or 20k checkpoint
SCHEDULE=<s1|s2|s3>         # M10 schedule
TASKSET=<task_set>          # atomic_seen, composite_seen, composite_unseen, or all
TASKS=<csv_list>            # Optional: override TASKSET with specific task names
TRIALS=20                   # Episodes per task
OUT=<output_json>           # logs/eval_m11_a1_*.json
```

**Config Deployed:**  
- Pi0.5 config: `pi05_robocasa_copred`
- Transformers: intent tokens with per-token adaRMS (3-D conditioning)
- Observation: 16D state + 2 RGB images (256×256 agentview_left, eye_in_hand)
- Action: 12D control (base 4D + mode 1D + eef_pos 3D + eef_rot 3D + gripper 1D)
- Control frequency: 20 Hz

---

## Notes and Caveats

1. **No existing eval_m11_robocasa.* outputs are present** at submission time (clean baseline)
2. **Throughput extraction:** env steps/sec and wall-clock per episode extracted from `.out` logs in post-processing (see `collect_phaseA.py` source)
3. **A1 checkpoint only:** A0/T-mask/A2a3 evals deferred per plan §1 (training on hold)
4. **Causal framing:** All A1 results frame intent as a "behaviorally useful channel" (J-mask model); strict causality requires T-mask arm (§1)
5. **Task filtering:** The 2-task unseen subset (ArrangeTea, PanTransfer) is passed via TASKS override; the script does not pre-select them from composite_unseen

---

## References

- Plan: `docs/intent_dsrl_plan_v2.md` §W1
- Eval report: `docs/robocasa_m11/D3_eval_report.md`
- Quick reference: `docs/robocasa_m11/QUICK_REFERENCE.md`
- Collector script: `scripts/collect_phaseA.py`
