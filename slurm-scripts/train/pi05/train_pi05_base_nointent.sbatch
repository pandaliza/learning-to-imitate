#!/bin/bash
#SBATCH --job-name=pi05-base-nointent
#SBATCH --output=logs/pi05_base_nointent_%j.out
#SBATCH --error=logs/pi05_base_nointent_%j.err
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-n5-32,babel-n9-32

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

# Dataset (same intent-augmented libero_goal; intent column ignored when include_intent=False).
export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_autotune_level=0"
# user_data is quota-blocked -> put the JAX cache on group_data (separate from the intent run).
export JAX_COMPILATION_CACHE_DIR=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/jax_cache_base_nointent
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# 1. Norm stats (idempotent; writes ./assets/pi05_base_nointent).
python external/openpi/scripts/compute_norm_stats.py --config-name pi05_base_nointent

# 2. No-intent control: SAME LoRA-VL + full-action-head finetune from pi05_base, intent_dim=0.
# 30k steps, batch 32 across 2 L40S. Checkpoints -> group_data. If OOM, drop --batch-size to 16.
python external/openpi/scripts/train.py pi05_base_nointent \
    --exp-name=base_nointent_goal_run1 \
    --batch-size=32 \
    --num-train-steps=30000 \
    --checkpoint-base-dir=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints \
    --resume
