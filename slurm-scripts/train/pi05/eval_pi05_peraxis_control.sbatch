#!/bin/bash
#SBATCH --job-name=pi05-peraxis-ctrl
#SBATCH --output=logs/pi05_peraxis_ctrl_%j.out
#SBATCH --error=logs/pi05_peraxis_ctrl_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00

# Per-axis perturbation, CONTROL (no intent). x-only (immediately OOD) vs y-only (in-dist to ~3σ).
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

B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
CK=$B/ldahiya_checkpoints/pi05_base_nointent/base_nointent_goal_run1/29999
DELTAS="0.0,0.02,0.03,0.05,0.10"

for AX in x y; do
  python examples/openpi/eval_libero_perturbed.py \
      --config-name pi05_base_nointent --checkpoint-dir "$CK" \
      --axis $AX --deltas "$DELTAS" --num-trials-per-delta 10 \
      --out logs/peraxis_control_${AX}.json
done
