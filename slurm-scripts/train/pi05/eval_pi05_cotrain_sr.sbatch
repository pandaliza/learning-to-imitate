#!/bin/bash
#SBATCH --job-name=pi05-cotrain-sr
#SBATCH --output=logs/pi05_cotrain_sr_%j.out
#SBATCH --error=logs/pi05_cotrain_sr_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00

# Co-train in-distribution success rate: all 10 libero_goal tasks, 20 trials each.
# VL-tapped intent (intent_stack.pt). Comparable to the base_intent / base_nointent SR sweeps.

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

CK=${CK:-/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_vl_cotrain_v2/24000}

python examples/openpi/eval_libero_intent.py \
    --config-name pi05_base_intent --checkpoint-dir "$CK" \
    --norm-stats-from-config --fp32 \
    --vl-cotrain --intent-stack "$CK/intent_stack.pt" \
    --task-suite libero_goal --num-trials-per-task 20 \
    --out logs/cotrain_sr_goal.json
