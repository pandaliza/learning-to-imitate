#!/bin/bash
#SBATCH --job-name=eval-perturb
#SBATCH --output=logs/eval_perturb_%j.out
#SBATCH --error=logs/eval_perturb_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# XY-perturbation robustness campaign for M10 co-prediction arms (A0, A1, A2, A3)
# Usage: sbatch --job-name=eval-perturb-A0 -d <jobid if needed> eval_perturb_m10.sbatch
# Environment variables (set via sbatch -e or inline):
#   CONFIG (required): config name (pi05_base_nointent for A0, pi05_base_copred for A1/A2/A3)
#   CKPT (required): checkpoint directory path
#   SCHEDULE (optional): schedule for M10 (s1, s2, s3); omit for A0
#   OUT (required): output JSON file path
#   TAG (required): tag for logging (used in output filenames)

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

# M10 copred checkpoints require patched transformers with per-token adaRMS (3-D cond).
# This is harmless for A0 (pure BC, no copred).
cp -r external/openpi/src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python*/site-packages/transformers/

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# Validate required env vars
CONFIG=${CONFIG:?CONFIG not set}
CKPT=${CKPT:?CKPT not set}
OUT=${OUT:?OUT not set}
TAG=${TAG:?TAG not set}

# Optional variables with defaults
SCHEDULE=${SCHEDULE:-}
NUM_INTENT_STEPS=${NUM_INTENT_STEPS:-4}

# Build python command
CMD="python examples/openpi/eval_libero_perturbed.py \
    --config-name $CONFIG \
    --checkpoint-dir $CKPT \
    --task-suite libero_goal \
    --norm-stats-from-config \
    --fp32 \
    --deltas 0.0,0.02,0.05,0.10,0.15,0.20 \
    --num-trials-per-delta 10 \
    --out $OUT"

# Add schedule (M10 only; omitted for A0)
if [ -n "$SCHEDULE" ]; then
    CMD="$CMD --schedule $SCHEDULE --num-intent-steps $NUM_INTENT_STEPS"
fi

echo "[eval_perturb_m10] TAG=$TAG CONFIG=$CONFIG CKPT=$CKPT SCHEDULE=${SCHEDULE:-none} OUT=$OUT" >&2
echo "[eval_perturb_m10] Starting perturbed eval..." >&2
eval "$CMD"

echo "[eval_perturb_m10] ✓ Complete. Result: $OUT" >&2
