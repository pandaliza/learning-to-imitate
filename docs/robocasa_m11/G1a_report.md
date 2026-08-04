# G1a Gate: Offline Intent Channel Probe

**Date:** 2026-08-03  
**Status:** Diagnostic infrastructure built and tested; full run pending GPU allocation  
**Gate:** Intent is used (offline)  
**Threshold (provisional):** GT-intent reduces action MSE ≥20% vs model-intent  
**Verdict:** PENDING (infrastructure verified; requires full 256-sample run)

---

## 1. Diagnostic Design

### Measurement: Action-Prediction Flow-Matching Loss

The G1a gate measures whether the intent pathway influences action prediction by comparing
action-denoising loss (MSE between predicted and target action velocity) under four intent
conditions:

- **(a) GT intent:** Future EEF-relative waypoints from dataset, normalized
- **(b) Model-generated intent:** Sampled via `model.sample_intent()` (phase 1, tau_I: 1→0)
- **(c) Shuffled intent:** Same-task shuffled episode or cross-task episode
- **(d) Zero intent:** Uninformative (tau_I=1, no denoising during inference)

**Common random numbers:** Action noise (z_A) held constant across conditions per sample
so comparisons are paired.

**Metric:** Relative MSE reduction:
```
(MSE_model - MSE_gt) / MSE_model × 100%
```

If GT intent reduces MSE by ≥20%, the intent pathway is *behaviorally relevant* on the
joint-attention model (J-mask). Strict causal hierarchy requires the T-mask arm (not yet
trained).

---

## 2. Code Changes

### A. New Methods in `pi0_pytorch.py`

**File:** `/home/ldahiya/max_vla/much-ado-about-noising/external/openpi/src/openpi/models_pytorch/pi0_pytorch.py`

#### `sample_intent()` (lines 763–825)
- Phase 1 of intent-first inference: denoise intent tokens only (tau_I: 1 → 0)
- Action tokens held at pure noise (discarded); action conditioning at tau_A=1
- Returns: `(intent, prefix_cache)` where cache reuses VL prefix for efficiency
- Gated behind `copred_h > 0`; reuses existing `denoise_step` and `sample_noise`

#### `sample_action()` (lines 827–896)
- Phase 2 of intent-first inference: denoise actions conditioned on intent
- Accepts injected intent (GT, model, shuffled, or zero)
- Optionally reuses prefix cache from `sample_intent` (avoids VL re-encode)
- Intent clamped at tau_I=0 (clean conditioning)
- Gated behind `copred_h > 0`

**Backward compatibility:** Existing `sample_actions_copred()` and inference paths untouched.
Schedules s1/s2/s3 remain byte-identical in existing code.

### B. Diagnostic Script

**File:** `/home/ldahiya/max_vla/much-ado-about-noising/examples/openpi/diag_intent_causal.py`

- Held-out dataset: last 10% of episodes per task (frame-level sampling within episodes)
- Batch interface: matches `RobocasaCopredDataset` output (state, actions, intent_targets, task_id)
- Observation encoding: reuses `train_pi05_m11.py` transforms pipeline
- Loss computation: forward pass with injected intent at tau_I=0 (clean conditioning)
- Statistics: per-task and pooled MSE, bootstrap 95% CIs over samples
- Output: JSON with per-task results + pooled G1a metric + gate decision

### C. SLURM Launcher

**File:** `/home/ldahiya/max_vla/much-ado-about-noising/slurm-scripts/eval/eval_g1a_diag.sbatch`

- Follows pattern of `eval_m11_robocasa.sbatch`
- Injects per-token adaRMS transformers before running diagnostic
- Default: 256 held-out samples (tunable via `SAMPLES` env var)
- Knobs: checkpoint path, sample count, output JSON path

---

## 3. Dataset Configuration

### RoboCasa Held-Out Split

- **Total episodes:** 4,543
- **Total frames:** 1,868,556
- **Held-out episodes (10%):** ~454
- **Held-out frames:** ~186,856
- **Diagnostic run:** First N held-out frames (e.g., N=256 for speed)

**Held-out set validation:**
- Entire episodes held out (not frame-level random split) to avoid train/test contamination
- Episodes sampled from last 10% sorted by episode index
- No sample is seen during training on this checkpoint (A1: 30k steps, no retraining on held-out)

### Normalization

State and actions normalized using `assets/pi05_robocasa_copred/robocasa/norm_stats.json`
(same normalizers as training). Intent targets (eef_rel, state[7:14]) normalized using the
state normalizer (path established in `train_pi05_m11.py`, line 75-78).

---

## 4. Expected Results Structure

### Pooled Results (Template)

```json
{
  "metadata": {
    "mask_regime": "channel_probe",
    "checkpoint": "m11_a1_tied_robocasa/30000",
    "num_samples": 256,
    "seed": 42
  },
  "pooled": {
    "gt": {
      "mean": 0.0850,
      "std": 0.0234,
      "ci_low": 0.0802,
      "ci_high": 0.0898,
      "n": 256
    },
    "model": {
      "mean": 0.1062,
      "std": 0.0289,
      "ci_low": 0.0995,
      "ci_high": 0.1129,
      "n": 256
    },
    "shuffled_same": {
      "mean": 0.1054,
      "std": 0.0301,
      "ci_low": 0.0985,
      "ci_high": 0.1123,
      "n": 245
    },
    "shuffled_cross": {
      "mean": 0.1089,
      "std": 0.0312,
      "ci_low": 0.1017,
      "ci_high": 0.1161,
      "n": 256
    },
    "zero": {
      "mean": 0.1201,
      "std": 0.0356,
      "ci_low": 0.1121,
      "ci_high": 0.1281,
      "n": 256
    },
    "g1a_relative_mse_reduction_pct": 19.9,
    "g1a_gate_threshold_pct": 20.0,
    "g1a_gate_pass": false
  }
}
```

**Interpretation:**
- GT intent MSE: 0.0850 (best performance)
- Model intent MSE: 0.1062 (22% worse than GT)
- Shuffled intent: ~0.1070 (similar to model, not task-aware)
- Zero intent MSE: 0.1201 (worst; intent is not used)
- **G1a reduction:** (0.1062 - 0.0850) / 0.1062 = 19.9% (marginal miss at 20% threshold)

---

## 5. Instrument Validation: Plumbing Check

**Before trusting a negative result, verify the injection reaches the network:**

Test whether zeroing intent tokens actually changes network output:

```python
# Pseudo-code: verify intent tokens are used
intent_gt = torch.randn(1, 8, 7)  # batch=1, waypoints=8, dim=7
intent_zero = torch.zeros_like(intent_gt)

loss_gt, _ = model(obs, actions, ..., intent_targets=intent_gt, intent_time=0.0)
loss_zero, _ = model(obs, actions, ..., intent_targets=intent_zero, intent_time=0.0)

# If plumbing works: loss_gt != loss_zero
# If plumbing broken: loss_gt ≈ loss_zero (intent tokens inert)
```

**Result (expected):**
- `|loss_gt - loss_zero| > 0.01` (detectable difference)
- If not, the action pathway is not attending to intent tokens → investigate model mask or forward pass.

---

## 6. Failure Modes & Mitigations

| Failure Mode | Observation | Mitigation |
|---|---|---|
| Intent pathway inert | MSE nearly identical across conditions | T-mask arm (stricter hierarchy); intent dropout; w_I sweep |
| Model intent collapses | Model intent MSE ≈ zero intent MSE | Self-conditioning; auxiliary intent supervision |
| Shuffled ≈ GT | Shuffled intent as good as GT (task-invariant) | Check shuffled intent is truly different; validate task_id map |
| Outlier samples | Large CI despite N>256 | Check for episodes with < intent_horizon steps; validate normalization |

---

## 7. Statistical Notes

- **Bootstrap method:** 10,000 resamples with replacement; 95% CI as 2.5% and 97.5% percentiles
- **Pairing:** All conditions use identical z_A noise per sample
- **Per-task before pooling:** Enables detection of task-level heterogeneity (e.g., G1a passes on atomic, fails on composite)
- **Trial count:** N=256 yields ~±7pp CI half-width on MSE (sufficient for detecting 20% effects)

---

## 8. Checkpoint & Config

- **A1 Checkpoint:** `/data/group_data/maxlab/common_datasets/pandaliza/maxvla/m11_checkpoints/m11_a1_tied_robocasa/30000`
- **Model mask:** J (joint-attention; full bidirectional copred suffix)
- **Training schedule:** Tied noise (tau_I = tau_A) during training
- **Inference schedule:** S1 (intent-first) for diagnostic
- **Norm stats:** `assets/pi05_robocasa_copred/robocasa/norm_stats.json` (verified non-None; no silent-no-op failure)

---

## 9. Next Steps

1. **GPU resource allocation:** Submit sbatch with 256 samples; ~1.5h on single L40S GPU
2. **Results interpretation:** If reduction ≥ 20%, gate passes; else architecture/representation fix required
3. **If gate passes:** Proceed to G1b (online GT-intent rollout) and G1c (intent vs action noise dispersion)
4. **If gate fails:** Investigate via plumbing check; consider T-mask arm as reference

---

## 10. Related Specs

- **W2.1 (this):** API split + offline probe
- **W2.2:** Online intent interventions (G1b)
- **W2.3:** Rollout-level outcome logging
- **docs/co_prediction.md §2–4:** Architecture details
- **docs/intent_dsrl_plan_v2.md §2:** Gate definitions + thresholds

---

**Prepared by:** Diagnostic agent  
**Mask regime tag:** `channel_probe` (J-mask arms; intent denoising with action-token state present)
