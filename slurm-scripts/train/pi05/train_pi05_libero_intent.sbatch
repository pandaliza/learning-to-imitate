#!/bin/bash
#SBATCH --job-name=pi05-libero-intent
#SBATCH --output=logs/pi05_libero_intent_%j.out
#SBATCH --error=logs/pi05_libero_intent_%j.err
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail

mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

# LeRobot dataset location (intent-augmented libero_goal) + JAX GPU memory.
export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# L40S has no cached XLA autotune configs -> first-step autotuning pegs the GPU for
# ~1h. Disable autotuning (algorithm choice only, not math) for fast multi-GPU compile,
# and persist compiled kernels so reruns/resumes skip compilation entirely.
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export JAX_COMPILATION_CACHE_DIR=/data/user_data/ldahiya/jax_cache
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-manual}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-manual}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-manual}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# 1. Normalization stats for the intent dataset (idempotent; writes to ./assets/pi05_libero_intent).
python external/openpi/scripts/compute_norm_stats.py --config-name pi05_libero_intent

# 2. Action-head-only finetune with slot-intent conditioning (data-parallel across 8 GPUs).
# Batch 32 across 2 GPUs (16/GPU) — fits L40S 48GB; JAX Gemma already remats every
# layer (gemma.py: nn.remat, nothing_saveable), so memory is batch-bound, not ckpt-bound.
python external/openpi/scripts/train.py pi05_libero_intent \
    --exp-name=slot_intent_goal_run1 \
    --batch-size=32 \
    --checkpoint-base-dir=/data/user_data/ldahiya/openpi-checkpoints \
    --overwrite
