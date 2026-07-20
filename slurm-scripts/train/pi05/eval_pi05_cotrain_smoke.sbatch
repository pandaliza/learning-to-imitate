#!/bin/bash
#SBATCH --job-name=pi05-cotrain-smoke
#SBATCH --output=logs/pi05_cotrain_smoke_%j.out
#SBATCH --error=logs/pi05_cotrain_smoke_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00

# SMOKE TEST: verify the VL co-train inference path in eval_libero_perturbed.py produces
# sane actions (non-zero SR at δ=0) before launching the full suite. 1 task, few trials.

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu                 # policy is PyTorch; keep JAX off the GPU
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

CK=${CK:-/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_vl_cotrain_v2/24000}

python examples/openpi/eval_libero_perturbed.py \
    --config-name pi05_base_intent \
    --checkpoint-dir "$CK" \
    --norm-stats-from-config \
    --fp32 \
    --vl-cotrain \
    --intent-stack "$CK/intent_stack.pt" \
    --tasks 1 \
    --deltas 0.0,0.10 \
    --num-trials-per-delta 3 \
    --out logs/cotrain_smoke.json
