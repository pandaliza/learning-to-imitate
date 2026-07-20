#!/bin/bash
#SBATCH --job-name=pi05-base-intent-m3
#SBATCH --output=logs/pi05_base_intent_m3_%j.out
#SBATCH --error=logs/pi05_base_intent_m3_%j.err
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-n5-32,babel-n9-32

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

# M3 intent-augmented libero_goal dataset (VL-grounded intent) + JAX GPU memory.
export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export JAX_COMPILATION_CACHE_DIR=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/jax_cache_base_intent_m3
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# M3 = M1's exact finetune recipe (pi05_base + LoRA-VL + full action head + frozen intent),
# the ONLY difference being the VL-grounded intent baked into ldahiya/libero_goal_slot_intent_m3.
# 1. Norm stats (idempotent; writes ./assets/pi05_base_intent_m3).
python external/openpi/scripts/compute_norm_stats.py --config-name pi05_base_intent_m3

# 2. Finetune. 30k steps, batch 32 across 4 L40S. Checkpoints -> group_data.
python external/openpi/scripts/train.py pi05_base_intent_m3 \
    --exp-name=base_intent_m3_goal_run1 \
    --batch-size=32 \
    --num-train-steps=30000 \
    --checkpoint-base-dir=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints \
    --resume
