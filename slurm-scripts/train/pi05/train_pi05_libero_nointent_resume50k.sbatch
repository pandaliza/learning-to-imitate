#!/bin/bash
#SBATCH --job-name=pi05-nointent-50k
#SBATCH --output=logs/pi05_libero_nointent_50k_%j.out
#SBATCH --error=logs/pi05_libero_nointent_50k_%j.err
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-n5-32

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export JAX_COMPILATION_CACHE_DIR=/data/user_data/ldahiya/jax_cache_nointent
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# RESUME the no-intent control (same exp_name/checkpoint dir) from step 30k -> 50k,
# matching the intent arm's 50k. --resume loads the latest checkpoint; LR flat 5e-5.
python external/openpi/scripts/train.py pi05_libero_nointent \
    --exp-name=nointent_goal_run1 \
    --batch-size=32 \
    --num-train-steps=50000 \
    --checkpoint-base-dir=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints \
    --resume
