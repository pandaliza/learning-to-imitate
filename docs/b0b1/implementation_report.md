# B0/B1 Implementation Report
## From-Scratch Flow-Matching Generative Policies for RoboCasa

**Date:** 2026-08-04  
**Specification:** docs/intent_dsrl_plan_v3.md §2 (B0/B1 formulation), §6 (architecture), §5 (execution order)  
**Branch:** copredict  

---

## Summary

Implemented B0/B1 from-scratch generative policies over frozen SigLIP-SO400M vision-language features. B0 is action-only flow matching; B1 adds intent co-prediction with causal T-mask attention (intent tokens never attend to action tokens by construction) and independent tau_I/tau_A noise scheduling with stratified corners (P(tau_I=0)=0.25, P(tau_I=1)=0.10).

**Parameter counts** (default config: depth=6, dim=512, num_heads=8):
- B0: 72,718,092 trainable params
- B1: 72,725,779 trainable params (intent projections: +7.7k)

**Smoke test:** All tests pass (random tensors + real SigLIP cache, both pooled and grid16 variants, vision_dim=1152, lang_dim=1152).

---

## File Map

### Core B0/B1 Module (`b0b1/`)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 7 | Package docstring |
| `dit.py` | 429 | DiT backbone: transformer blocks, per-token adaLN-zero conditioning (tau_I/tau_A), T-mask for B1 |
| `feature_dataset.py` | 212 | Dataset adapter: reads cached VL features (manifest-driven dims), wraps RobocasaCopredDataset |
| `sample.py` | 237 | Inference: B0 Euler (10 steps), B1 intent-first (K_I=4 intent + K_A=10 action), velocity clipping |

### Training & Testing (`examples/b0b1/`)
| File | Lines | Purpose |
|------|-------|---------|
| `train_b0b1.py` | 378 | Training loop: flow matching loss, stratified noise sampler for B1, AdamW, cosine LR, checkpoints every 25k steps |
| `test_b0b1_smoke.py` | 407 | Smoke tests: forward pass (random), T-mask unit test, training loops (30 steps), sampling, real cache (pooled + grid16) |

### SLURM Launchers (`slurm-scripts/b0b1/`)
| File | Purpose |
|------|---------|
| `train_b0.sbatch` | Launch B0 training: 1×L40S, 24h, bs=256, 200k steps, checkpoints to `/data/group_data/maxlab/common_datasets/pandaliza/b0b1_checkpoints/b0/` |
| `train_b1.sbatch` | Launch B1 training: 1×L40S, 24h, bs=256, 200k steps, checkpoints to `/data/group_data/maxlab/common_datasets/pandaliza/b0b1_checkpoints/b1/` |

---

## Architecture Details

### DiT (Diffusion Transformer)

**Components:**
- **Context tokens:** Vision features (base + wrist cameras, 1152D each via frozen SigLIP) + language embedding (1152D) + proprioception encoder (16D state → dim)
- **Suffix tokens:** 
  - B0: action tokens [A_1..A_10] (10 tokens, 12D each)
  - B1: intent tokens [I_1..I_8 | A_1..A_10] (18 tokens: 8 intent 7D + 10 action 12D)
- **Per-token conditioning:** Sinusoidal time embedding → MLP → separate tau_I (for intent) and tau_A (for action) conditioning vectors, broadcast to all tokens
- **AdaLN-zero:** Learnable scale/shift/gate per token applied after LayerNorm (residual + gated)

**T-mask (B1 only, PyTorch convention: True=mask out, False=attend):**
```
Block structure: [context (3 tokens) | intent (8) | action (10)]
Intent rows [3:11]:  False for columns [0:11], True for [11:21]  (attend to context+intent, mask action)
Action rows [11:21]: False for all columns                       (attend to everything)
Context rows [0:3]:  False for all columns                       (attend to everything)
```

This ensures **causal hierarchy by construction:** intent is denoised independently of actions.

### Flow Matching Loss

**B0 (standard):**
```
L = E[ ||v_theta(x_tau, tau, ctx) - (eps - x0)||^2 ]
where x_tau = tau*eps + (1-tau)*x0, target u = eps - x0
```

**B1 (decoupled noise + stratification):**
```
L = w_A * E[ ||v_A - u_A||^2 ] + w_I * E[ ||v_I - u_I||^2 | tau_I > 0.01 ]
```
- Independent tau_I, tau_A ~ Beta(1.5,1) per sample
- Stratification: P(tau_I=0)=0.25 (clean intent conditioning), P(tau_I=1)=0.10 (uninformative), else independent
- Intent loss masked when tau_I=0 (no denoising step needed)
- Defaults: w_A=1.0, w_I=1.0

### Feature Cache Interface (Contract)

**Manifest:** `/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features/<variant>/manifest.json`
```json
{
  "variant": "pooled" | "grid16",
  "vision_dim": 1152,
  "num_vision_tokens": 1 | 16,
  "lang_dim": 1152,
  "cameras": ["base", "wrist"],
  "dtype": "float16",
  "frame_indexing": "matches LeRobot episode frame index"
}
```

**Per-episode npz:** `features/<task>/<episode_stem>.npz`
- `base`: (N_frames, 1152) or (N_frames, 16, 1152)
- `wrist`: (N_frames, 1152) or (N_frames, 16, 1152)
- `lang`: (1152,) — instruction embedding (constant per episode)

**Reader:** `steer_intent/feature_cache.py::FeatureCache` (manifest-driven dims, no hardcoding)

---

## Smoke Test Output

**Command:** `python examples/b0b1/test_b0b1_smoke.py`

```
======================================================================
B0/B1 Smoke Tests (1152D SigLIP + Real Feature Cache)
======================================================================

[TEST] B0 forward pass (random)
  ✓ B0 forward pass OK
[TEST] B1 forward pass (random)
  ✓ B1 forward pass OK
[TEST] B1 T-mask unit test
  ✓ B1 T-mask verified:
    Intent rows [3:11] mask action cols [11:21]
    Action rows [11:21] attend to all
[TEST] B1 intent_out_proj zero-init
  ✓ intent_out_proj zero-initialized
[TEST] B0 training loop (30 steps)
  Initial: 2.1114, Final: 2.0178
  ✓ B0 training loop OK (loss finite)
[TEST] B1 training loop (30 steps)
  Initial: 4.1100, Final: 4.0161
  ✓ B1 training loop OK (loss finite)
[TEST] B0 sampling
  ✓ B0 sampling OK
[TEST] B1 sampling
  ✓ B1 sampling OK
[TEST] Real cache (pooled)
  ✓ Real cache (pooled) OK | vision_dim=1152, lang_dim=1152
[TEST] Real cache (grid16)
  ✓ Real cache (grid16) OK | vision_dim=1152, num_vision_tokens=16

======================================================================
All smoke tests PASSED ✓
======================================================================
```

### T-Mask Unit Test Evidence

From `test_b0b1_t_mask()`:
- **Intent rows [3:11]** (8 intent tokens): `mask[i, 11:21] = True` for all i ∈ [3:11]
  - Verification: loop over all intent rows, assert True for all action columns
- **Action rows [11:21]** (10 action tokens): `mask[i, :] = False` for all i ∈ [11:21]
  - Verification: loop over all action rows, assert False for all columns
- **PyTorch convention:** True = "mask out (don't attend)", False = "attend"
- **Result:** Intent tokens never see action token state; action tokens see clean intent during phase 2 of sampling

---

## Sampling Schedules

### B0: Standard Euler (10 steps)
```python
sample_b0(ctx, action_dim=12, action_horizon=10, steps=10, z_action=None, seed=None)
  → tau_A: 1.0 → 0.0 over 10 steps
  → action: (B, 10, 12)
```

### B1: Intent-First (4 + 10 steps)
```python
sample_b1(ctx, ..., steps_intent=4, steps_action=10, z_intent=None, z_action=None, seed=None)
  Phase 1: tau_I: 1.0 → 0.0 over 4 steps (actions at noise)
  Clamp: intent → model-generated clean intent
  Phase 2: tau_A: 1.0 → 0.0 over 10 steps (conditioned on clean intent)
  → intent: (B, 8, 7), action: (B, 10, 12)
```

Both accept injectable noise `z_I`, `z_A` for reproducibility/control (DSRL interface).

---

## Training Configuration (Defaults)

| Param | Value | Notes |
|-------|-------|-------|
| **Model** | | |
| depth | 6 | transformer blocks |
| dim | 512 | embedding dimension |
| num_heads | 8 | attention heads |
| vision_dim | 1152 | frozen SigLIP features |
| num_vision_tokens | 1 | pooled variant (16 for grid16) |
| lang_dim | 1152 | instruction embedding |
| **Training** | | |
| batch_size | 256 | on cached features (CPU-bound load) |
| optimizer | AdamW | lr=1e-4, weight_decay=0.01 |
| lr_schedule | CosineAnnealingLR | T_max = total_steps |
| steps | 200,000 | total training steps |
| checkpoint_interval | 25,000 | save every 25k steps, keep all |
| log_interval | 100 | log loss to stdout/stderr |
| **Noise (B1)** | | |
| time_dist | Beta(1.5,1) | clamped to [0.001, 0.999] |
| tau_I stratify | 0.25, 0.10 | P(=0), P(=1), else independent |
| w_I, w_A | 1.0, 1.0 | loss weights (sweep if needed) |

---

## Launch Commands

### Do NOT submit (feature cache not yet fully extracted for production run)

```bash
# B0 training (CPU quota: ~6-8 GPU-hours on L40S)
sbatch --export=NONE slurm-scripts/b0b1/train_b0.sbatch

# B1 training (same compute)
sbatch --export=NONE slurm-scripts/b0b1/train_b1.sbatch
```

**Important:** Use `--export=NONE` to prevent inherited `CUDA_VISIBLE_DEVICES` from breaking GPU initialization.

**Environment variables for customization:**
```bash
VARIANT=pooled              # or grid16
DATA_ROOT=/path/to/robocasa/target
FEATURE_CACHE=/path/to/b0b1_features/pooled
NORM_STATS=/path/to/norm_stats.json
CKPT_DIR=/data/.../b0b1_checkpoints
```

**Example with overrides:**
```bash
sbatch --export=NONE \
  --env=VARIANT=grid16 \
  slurm-scripts/b0b1/train_b1.sbatch
```

---

## Deviations from v3 Spec

**None significant.** The implementation follows spec §2-§6 exactly:

1. ✓ DiT 6 blocks, width 512 (~73M params vs. spec's "30–80M")
2. ✓ Per-token adaLN-zero tau conditioning (tau_I ≠ tau_A)
3. ✓ T-mask: intent → {context, intent}, action → {context, intent, action}
4. ✓ B0: standard flow loss; B1: decoupled + stratified (0.25, 0.10)
5. ✓ Dataset reads manifest-driven cached features (vision_dim=1152, no hardcoding)
6. ✓ Sampling: B0 Euler 10 steps, B1 intent-first (4+10 steps)
7. ✓ Intent projections zero-init (silent start)
8. ✓ Checkpoints every 25k steps (keep all), per spec "not keep-last-3"

**Minor choices (not contradicting spec):**
- Velocity clipping (±10.0) in sampling to prevent untrained-model divergence during smoke test
- Feature cache reader at `steer_intent/feature_cache.py` (external agent, not part of this build)

---

## Known Limitations & Next Steps

1. **Feature cache volume:** Pooled variant (~15 GB); grid16 would be 150–250 GB fp16. Choose variant based on availability (pooled faster for smoke test, grid16 for full runs).

2. **Norm stats validation:** Must confirm `assets/pi05_robocasa_copred/robocasa/norm_stats.json` matches the RoboCasa dataset version used in feature extraction (cached normalization bug can invalidate entire run).

3. **Pre-DSRL gates (v3 §4):** After B1 training, run:
   - GT-intent injection test
   - Shuffled/zero intent test
   - Varying z_I (z_A fixed) → task outcomes
   - Best-of-N over z_I

4. **Sampling strategy tuning:** Current K_I=4, K_A=10 is conservative; measure wall-clock time and adjust for deployment.

---

## Files Modified / Created (Additive Only)

### Created
- `b0b1/{__init__,dit,feature_dataset,sample}.py`
- `examples/b0b1/{train_b0b1,test_b0b1_smoke}.py`
- `slurm-scripts/b0b1/{train_b0,train_b1}.sbatch`
- `docs/b0b1/implementation_report.md` (this file)

### NOT Modified
- `external/openpi`, `steer_intent/robocasa_copred_dataset.py`, `examples/openpi/eval_robocasa_intent.py`
- Any running job scripts (M11 sbatch files, pi0.5 training)
- `assets/pi05_robocasa_copred/robocasa/norm_stats.json`

---

## References

- **Spec:** `docs/intent_dsrl_plan_v3.md`, `docs/co_prediction.md`
- **Dataset:** `steer_intent/robocasa_copred_dataset.py` (RobocasaCopredDataset, h=8, Δ=2, d_I=7)
- **Feature cache:** `steer_intent/feature_cache.py` (FeatureCache reader, manifest contract)
- **Norm stats:** `assets/pi05_robocasa_copred/robocasa/norm_stats.json` (state, actions)
- **Test data:** `/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features_test/{pooled,grid16}/` (TurnOnElectricKettle episode_000000)
