#!/bin/bash
#SBATCH --job-name=pi05-m4-cond-sr
#SBATCH --output=logs/pi05_m4_cond_sr_%j.out
#SBATCH --error=logs/pi05_m4_cond_sr_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00

# FIXED-M4 in-distribution SR. pi05_base_intent PyTorch ckpt (norm stats from config, fp32).
# Generator = aux MLP: z_hat = aux_head(pi05 vl_mean of current obs) -> fed to the action head.
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

STEP=${STEP:-30000}
TRIALS=${TRIALS:-20}
CK=${CK:-/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_m4_cond/${STEP}}
OUT=${OUT:-logs/m4_cond_sr_goal_${STEP}.json}

python examples/openpi/eval_libero_intent.py \
    --config-name pi05_base_intent --checkpoint-dir "$CK" \
    --norm-stats-from-config --fp32 \
    --vl-cotrain \
    --intent-task-config libero_goal_suite_image_slot_intent_vl \
    --aux-head "$CK/aux_head.pt" \
    --task-suite libero_goal --num-trials-per-task "$TRIALS" \
    --out "$OUT"
