#!/bin/bash
#SBATCH --job-name=pi05-base-perturb
#SBATCH --output=logs/pi05_base_perturb_%j.out
#SBATCH --error=logs/pi05_base_perturb_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Radial XY-perturbation sweep on the pi05_base arms (LoRA-VL + full action head).
# Frozen-intent vs no-intent control, both at the matched final checkpoint (29999).
# Mirrors eval_pi05_libero_perturbed.sbatch but on the pi05_base_* runs.

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export JAX_COMPILATION_CACHE_DIR=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/jax_cache_eval
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

CKPT_BASE=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints
NOINTENT_CKPT=$CKPT_BASE/pi05_base_nointent/base_nointent_goal_run1/29999
INTENT_CKPT=$CKPT_BASE/pi05_base_intent/base_intent_goal_run1/29999
FLOW_MAP=/data/user_data/ldahiya/libero-train/libero-goal-suite-image/slot_intent/models/model_best.pt

DELTAS="0.0,0.02,0.05,0.10,0.15,0.20"
TRIALS=10   # 8 tasks x 6 deltas x 10 trials = 480 episodes each run

# 1. No-intent control.
python examples/openpi/eval_libero_perturbed.py \
    --config-name pi05_base_nointent \
    --checkpoint-dir "$NOINTENT_CKPT" \
    --deltas "$DELTAS" \
    --num-trials-per-delta $TRIALS \
    --out logs/base_perturb_nointent.json

# 2. Frozen-intent.
python examples/openpi/eval_libero_perturbed.py \
    --config-name pi05_base_intent \
    --checkpoint-dir "$INTENT_CKPT" \
    --intent \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt "$FLOW_MAP" \
    --deltas "$DELTAS" \
    --num-trials-per-delta $TRIALS \
    --out logs/base_perturb_intent.json

echo "===================== pi05_base SR-vs-δ COMPARISON ====================="
python - <<'PY'
import json
ni = json.load(open("logs/base_perturb_nointent.json"))
it = json.load(open("logs/base_perturb_intent.json"))
deltas = ni["deltas"]
print(f"{'delta':>6}  {'nointent':>9}  {'intent':>8}  {'Δ_SR':>8}  "
      f"{'ni|valid':>9}  {'it|valid':>9}  {'Δ|valid':>8}  {'valid%':>7}")
print("-" * 80)
for d in deltas:
    r_ni, r_it = ni["results"][str(d)], it["results"][str(d)]
    sr_ni  = r_ni.get("_mean_sr", float("nan"));        sr_it  = r_it.get("_mean_sr", float("nan"))
    srv_ni = r_ni.get("_mean_sr_valid", float("nan"));  srv_it = r_it.get("_mean_sr_valid", float("nan"))
    vf     = r_ni.get("_mean_valid_frac", float("nan"))  # identical init states -> same valid frac both arms
    print(f"{d:6.2f}  {sr_ni:9.3f}  {sr_it:8.3f}  {sr_it - sr_ni:+8.3f}  "
          f"{srv_ni:9.3f}  {srv_it:9.3f}  {srv_it - srv_ni:+8.3f}  {100*vf:6.1f}%")
PY
