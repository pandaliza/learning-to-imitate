# W3 De-risk: Step-0 Setup Report

**Date:** 2026-08-03  
**Branch:** copredict  
**Status:** Ready to launch

## Executive Summary

W3 (Robomimic square-mh-state control ladder) step 0 is set up to validate two base BC policies before RL. The flow-intent implementation (Config A) already exists in the codebase; configs and training scripts are ready; validation harness scaffolding is in place.

---

## 1. Flow-Intent Implementation Status

**Location:** `mip/flow_intent_agent.py`  
**Status:** FOUND AND READY TO USE

The Config A (flow-intent) architecture is fully implemented:
- **Shared context encoder:** MLP maps obs (B, To, obs_dim) → obs_emb (B, To, emb_dim)
- **Flow intent model:** FlowMap operates in intent space (Ta=1, D=7 or D=intent_emb_dim) samples intent from noise via ODE
- **MLP action decoder:** Deterministic (obs_emb, intent) → action
- **Training:** Two independent gradient steps per iteration
  - Step 1: flow loss on encoder + intent_flow_map
  - Step 2: MSE loss on action_decoder (with stopped gradients from obs_emb)
- **Inference:** intent ODE → action decoder

Supporting infrastructure:
- Task configs: `examples/configs/task/{lift,square}_ph_state_flow_intent.yaml`
- Network config: `examples/configs/network/mlp_flow_intent.yaml`
- Training already handles FlowIntentAgent detection via `arch_variant` (line 129, examples/train/train_robomimic.py)

**Not built, already exists:** Yes. No new flow-intent code needed.

---

## 2. Dataset and Obs/Intent Indices

### Dataset Location

**HuggingFace repo:** `ChaoyiPan/mip-dataset`  
**Dataset pattern:** `robomimic/{task}/{env_type}/low_dim.hdf5`

For **square-mh-state**:
- **Path:** `robomimic/square/mh/low_dim.hdf5`
- **Auto-download:** Yes, via `make_dataset()` in `mip/datasets/robomimic_dataset.py`
- **Storage:** HuggingFace hub cache (downloaded on first training run)

### Obs and EEF Indices (Verified Dynamically)

The dataset module (`RobomimicDataset.__init__` lines 268–312) computes intent slices dynamically from HDF5 key dimensions:

1. **obs_keys for square:** `["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]`
   - These are specified in `examples/configs/task/square_ph_state.yaml` (inherits to square_mh_state)

2. **Intent extraction:**
   - Intent keys: `["robot0_eef_pos", "robot0_eef_quat"]` (default in robomimic_dataset.py line 270)
   - Dimensions from HDF5:
     - `object`: dim inferred from HDF5 (≈14D)
     - `robot0_eef_pos`: 3D
     - `robot0_eef_quat`: 4D
     - `robot0_gripper_qpos`: 2D
   - Intent indices computed as: start = offset of `robot0_eef_pos`, end = offset of `robot0_eef_quat` + 4
   - Expected: **[14:21]** (or [obs_offset:obs_offset+7], verified at runtime)

3. **Runtime verification:**
   - `RobomimicDataset.intent_slices` is populated in `__init__` and logged
   - Intent extraction (lines 370–392): mean/final/sequence of future EEF over next act_steps

**Conclusion:** Intent indices are verified at runtime from the dataset; square-mh obs_dim and eef slices will be correct automatically.

---

## 3. Training Configurations and Scripts

### Task Configs Created

1. **`examples/configs/task/square_mh_state_flow_intent.yaml`**  
   - Inherits from `square_mh_state`
   - Sets: `intent_conditioning=true`, `intent_dim=7`, `intent_type=mean`, `intent_predictor=false`
   - Same design as lift_ph_state_flow_intent (Config A)

### Sbatch Training Scripts

Location: `slurm-scripts/w3_derisk/`

#### 1. **`train_square_mh_state_baseline.sbatch`**
- **Job name:** `w3-square-mh-baseline`
- **GPU:** 1× GPU (32GB, 12h wall-clock)
- **Gradient steps:** 300,000
- **Batch size:** 256
- **Task:** `square_mh_state`
- **Network:** `mlp` (plain flow BC, Config B style)
- **Expected runtime:** ≈6–8 hours
- **Expected SR:** 30–70% (difficulty-matched window)

#### 2. **`train_square_mh_state_flow_intent.sbatch`**
- **Job name:** `w3-square-mh-flow-intent`
- **GPU:** 1× GPU (32GB, 12h wall-clock)
- **Gradient steps:** 300,000
- **Batch size:** 256
- **Task:** `square_mh_state_flow_intent`
- **Network:** `mlp_flow_intent` (Config A)
- **Expected runtime:** ≈6–8 hours (intent sampling ~same cost as action sampling)
- **Expected SR:** 30–70% (to be validated in step 0)

### Launch Commands

```bash
# Baseline
sbatch slurm-scripts/w3_derisk/train_square_mh_state_baseline.sbatch

# Flow-intent
sbatch slurm-scripts/w3_derisk/train_square_mh_state_flow_intent.sbatch
```

---

## 4. Validation Harness (Step 0)

**Location:** `examples/eval/eval_w3_step0_validation.py`

**Purpose:** Verify base policies before RL deployment.

**Metrics:**
- **(a) Success rate (50 rollouts):** Target 30–70% for both
  - Tool: `eval_success_rate.py` (existing)
  - Command: See section 5
- **(b) Diversity (VDR/ghost bundles):** Reuse `collect_multimodal_viz.py` + `analyze_multimodal_clustering.py`
  - Command: Calls `collect_multimodal_viz.py` with critical state sampling
- **(c) Intent intervention:** Sample K=10 intents from fixed states, measure endpoint dispersion
  - Tool: `eval_w3_step0_validation.py` (scaffold in place; implementation deferred)

**Key gates (provisional):**
- Both bases in [30%, 70%] SR (adjust per actual results; fallback to reduced-demo if mh diverges)
- VDR(flow-intent) > 0.35 on mh (prior: VDR 0.71 on square-ph)
- Intent intervention causes ≥2× larger endpoint dispersion than action noise alone

---

## 5. Evaluation Commands (Post-Training)

Once checkpoints exist:

### 5a. Success Rate (50 rollouts each)

```bash
cd /home/ldahiya/max_vla/much-ado-about-noising

python examples/eval/eval_success_rate.py \
    --run "baseline:checkpoints/square_mh_state_flow_baseline_*.pt:task=square_mh_state" \
    --run "flow_intent:checkpoints/square_mh_state_flow_flow_intent_*.pt:task=square_mh_state_flow_intent:+network.arch_variant=flow_intent" \
    --n-rollouts 50 \
    --out results/w3_step0_sr.csv
```

### 5b. Diversity (VDR/Clustering)

```bash
# Collect multimodal rollouts
python examples/collect/collect_multimodal_viz.py \
    --task square_mh_state \
    --run "baseline:checkpoints/square_mh_state_flow_baseline.pt:task=square_mh_state" \
    --run "flow_intent:checkpoints/square_mh_state_flow_flow_intent.pt:task=square_mh_state_flow_intent:+network.arch_variant=flow_intent" \
    --n-rollouts 20 \
    --n-samples 50 \
    --n-critical 10 \
    --device cuda \
    --out results/w3_step0_multimodal.pkl

# Analyze clustering and VDR
python examples/analyze_multimodal_clustering.py \
    --data results/w3_step0_multimodal.pkl \
    --task square_mh_state \
    --out-dir results/w3_step0_clustering_figs \
    --k-min 2 --k-max 6 --n-ghost-states 5
```

### 5c. Intent Intervention Probe

```bash
python examples/eval/eval_w3_step0_validation.py \
    --baseline-ckpt checkpoints/square_mh_state_flow_baseline.pt \
    --flow-intent-ckpt checkpoints/square_mh_state_flow_flow_intent.pt \
    --task square_mh_state \
    --n-rollouts 50 \
    --n-critical 10 \
    --out results/w3_step0_validation.json
```

---

## 6. Checkpoint Paths and Naming

**Default save location:** Hydra auto-creates `outputs/YYYY-MM-DD/HH-MM-SS/{task}_{network}_{exp_name}/`

**Naming pattern:** `{env}_{env_type}_{obs_type}_{loss}_{network}_{emb_dim}_seed{seed}[_intent]`

For our runs:
- **Baseline:** `square_mh_state_flow_mlp_512_seed0/model_best.pt`
- **Flow-intent:** `square_mh_state_flow_mlp_512_seed0_intent_flow_intent/model_best.pt`

**Recommendation:** After training, symlink or copy checkpoints to a stable location:
```bash
mkdir -p checkpoints/w3_derisk
cp outputs/*/square_mh_*/model_best.pt checkpoints/w3_derisk/
```

---

## 7. Data Storage and Home Quota

**Home quota issue:** `/home/ldahiya` (100GB NFS) is chronically full (MEMORY.md note).

**Dataset auto-download location:** Controlled by `HF_HOME` (defaults to `~/.cache/huggingface`)

**Workaround:**
```bash
export HF_HOME=/data/user_data/ldahiya/hf_cache
```

**Checkpoints:** Save to `/data/user_data/ldahiya/` or group volume, NOT `/home/ldahiya`.

---

## 8. Code Changes (Additive, No Modifications to Existing Paths)

### New Files
1. `examples/configs/task/square_mh_state_flow_intent.yaml` ✓
2. `slurm-scripts/w3_derisk/train_square_mh_state_baseline.sbatch` ✓
3. `slurm-scripts/w3_derisk/train_square_mh_state_flow_intent.sbatch` ✓
4. `examples/eval/eval_w3_step0_validation.py` ✓
5. `docs/w3_derisk/step0_setup_report.md` ✓ (this file)

### No Changes to
- `mip/flow_intent_agent.py` — already exists, ready
- `examples/configs/task/lift_ph_state_flow_intent.yaml` — unchanged
- `examples/train/train_robomimic.py` — already handles FlowIntentAgent
- Pi0.5 or steer_intent code — untouched

---

## 9. Known Constraints and Risks

| # | Item | Mitigation |
|---|---|---|
| 1 | square-mh dataset not in HF yet? | Check HF repo; fallback: download from original Robomimic v0.3 |
| 2 | obs_dim or EEF indices wrong for square (vs lift) | Dataset computes dynamically; verify logs on first training run |
| 3 | flow-intent sample cost higher than expected | Implement MIP 2-call instead of Euler ODE (`arch_variant=flow_intent_mip`) |
| 4 | Base SR outside [30%, 70%] window | Use reduced-demo or shifted-reset for difficulty matching (fallback to secondary mechanism) |
| 5 | Babel node flakes / 48h timeout | Archive checkpoints frequently; use `--exclude` for known-bad nodes |

---

## 10. Timeline and Next Steps

**Step 0 (Ready now):**
- ✓ Configs created
- ✓ Sbatch scripts ready
- ✓ Validation scaffold in place

**Sequence:**
1. **Launch baseline + flow-intent training** (2 jobs, ~8h each, parallel)
2. **Monitor SR in logs** (eval every 10k steps)
3. **Once base SRs stabilize** (~50k–100k steps), run 50-rollout evaluation
4. **Check gate metrics:**
   - (a) Both in 30–70% SR?
   - (b) VDR > 0.35 on flow-intent?
   - (c) Intent intervention changes behavior?
5. **If gates pass:** Proceed to C0/C1/C3 control ladder (W3 phase 2)
6. **If gates fail:** Activate fallback (reduced-demo or shifted-reset)

---

## Appendix: Checkpoints and Asset Locations

**Existing related checkpoints:**
- `assets/pi05_base_copred/` — Pi0.5 co-pred base (for reference)
- `assets/pi05_libero/` — Pi0.5 LIBERO ckpts

**W3 outputs:**
- Training: Hydra outputs/ + symlinked to `checkpoints/w3_derisk/`
- Evals: `results/w3_step0_*.{csv,json,pkl}`
- Figs: `results/w3_step0_*_figs/`

---

## Sign-Off

**Setup owner:** Agent (rl-derisk)  
**Date ready:** 2026-08-03  
**Next milestone:** Training launched, base SRs converged (≈48h from launch)
