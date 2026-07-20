#!/bin/bash
#SBATCH --job-name=pi05-base-intent-diag
#SBATCH --output=logs/pi05_base_intent_diag_%j.out
#SBATCH --error=logs/pi05_base_intent_diag_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Two mechanistic diagnostics for the negative intent result:
#   #1 diag_flow_collapse   — is the flow-map prior p(z|s) collapsed / object-position blind?
#   #2 diag_intent_steering — does the adaRMS intent channel have causal authority over reach?
# Together they convert "intent didn't help" into "intent didn't help BECAUSE ...".

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
INTENT_CKPT=$CKPT_BASE/pi05_base_intent/base_intent_goal_run1/29999
FLOW_MAP=/data/user_data/ldahiya/libero-train/libero-goal-suite-image/slot_intent/models/model_best.pt

# #1 — flow-map collapse (no rollouts; fast)
python examples/openpi/diag_flow_collapse.py \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt "$FLOW_MAP" \
    --inits-per-task 5 \
    --deltas 0.0,0.02,0.05,0.10,0.15 \
    --out logs/diag_flow_collapse.json

# #2 — intent-steering causal authority (rollouts)
python examples/openpi/diag_intent_steering.py \
    --config-name pi05_base_intent \
    --checkpoint-dir "$INTENT_CKPT" \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt "$FLOW_MAP" \
    --tasks 1,8 \
    --magnitude 0.10 \
    --n-dirs 4 \
    --inits-per-dir 5 \
    --out logs/diag_intent_steering.json
