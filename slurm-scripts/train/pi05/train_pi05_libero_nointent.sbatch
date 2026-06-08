#!/bin/bash
#SBATCH --job-name=pi05-libero-nointent
#SBATCH --output=logs/pi05_libero_nointent_%j.out
#SBATCH --error=logs/pi05_libero_nointent_%j.err
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail

mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_autotune_level=0"
# Separate cache dir from the intent run — sharing one dir across concurrent jobs
# can lock/stall on compile-cache access.
export JAX_COMPILATION_CACHE_DIR=/data/user_data/ldahiya/jax_cache_nointent
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-manual}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-manual}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# Norm stats for the control config (same dataset, written under ./assets/pi05_libero_nointent).
python external/openpi/scripts/compute_norm_stats.py --config-name pi05_libero_nointent

# No-intent control: identical action-head finetune, intent_dim=0 (clean A/B vs intent).
python external/openpi/scripts/train.py pi05_libero_nointent \
    --exp-name=nointent_goal_run1 \
    --batch-size=32 \
    --checkpoint-base-dir=/data/user_data/ldahiya/openpi-checkpoints \
    --overwrite
