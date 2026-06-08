#!/bin/bash
#SBATCH --job-name=pi05-base-intent
#SBATCH --output=logs/pi05_base_intent_%j.out
#SBATCH --error=logs/pi05_base_intent_%j.err
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-n5-32,babel-n9-32

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

# Dataset (intent-augmented libero_goal, read-only) + JAX GPU memory.
export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_autotune_level=0"
# user_data is quota-blocked -> put the JAX cache on group_data.
export JAX_COMPILATION_CACHE_DIR=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/jax_cache_base_intent
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# 1. Norm stats for the intent dataset (idempotent; writes ./assets/pi05_base_intent).
python external/openpi/scripts/compute_norm_stats.py --config-name pi05_base_intent

# 2. Finetune pi05_base with frozen slot-intent: LoRA on PaliGemma LLM (2B base frozen) +
# FULL finetune of action expert + SigLIP + intent_proj. 30k steps, batch 32 across 2 L40S.
# Checkpoints -> group_data (user_data is quota-blocked). If OOM, drop --batch-size to 16.
python external/openpi/scripts/train.py pi05_base_intent \
    --exp-name=base_intent_goal_run1 \
    --batch-size=32 \
    --num-train-steps=30000 \
    --checkpoint-base-dir=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints \
    --resume
