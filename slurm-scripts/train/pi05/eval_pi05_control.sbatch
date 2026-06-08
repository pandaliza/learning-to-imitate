#!/bin/bash
#SBATCH --job-name=pi05-control-eval
#SBATCH --output=logs/pi05_control_eval_%j.out
#SBATCH --error=logs/pi05_control_eval_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=babel-n5-32

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export JAX_COMPILATION_CACHE_DIR=/data/user_data/ldahiya/jax_cache
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# No-intent control @30k, our convention (no --intent, no --rotate-images), libero_goal.
python examples/openpi/eval_libero_intent.py \
    --config-name pi05_libero_nointent \
    --checkpoint-dir /data/user_data/ldahiya/openpi-checkpoints/pi05_libero_nointent/nointent_goal_run1/29999 \
    --task-suite libero_goal \
    --num-trials-per-task 20 \
    --out logs/eval_control_goal.json
