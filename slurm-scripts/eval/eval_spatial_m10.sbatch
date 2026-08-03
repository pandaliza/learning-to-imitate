#!/bin/bash
#SBATCH --job-name=eval-spatial
#SBATCH --output=logs/eval_spatial_%j.out
#SBATCH --error=logs/eval_spatial_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

# Generic launcher for suite transfer eval (default libero_spatial).
# Usage:
#   sbatch --export=CONFIG=pi05_base_copred,CKPT=/path/to/ckpt,OUT=logs/eval.json,TAG=mytag[,SCHEDULE=s2][,SUITE=libero_10] eval_spatial_m10.sbatch
#
# Environment variables (required):
#   CONFIG:   Config name (e.g., pi05_base_nointent, pi05_base_copred)
#   CKPT:     Checkpoint directory path
#   OUT:      Output JSON file path
#   TAG:      Tag for job naming and logs
#   SCHEDULE: (optional) Schedule variant for copred (s1, s2, s3); omit for non-copred
#   SUITE:    (optional) LIBERO task suite (libero_spatial [default], libero_10, libero_object, ...)
#
# Outputs JSON to OUT; sbatch logs to logs/eval_spatial_<TAG>_{job-id}.out/.err

: "${CONFIG:?set CONFIG}"
: "${CKPT:?set CKPT}"
: "${OUT:?set OUT}"
: "${TAG:?set TAG}"
SUITE=${SUITE:-libero_spatial}

# For copred checkpoints, synce the transformers module with the custom per-token adaRMS (3-D cond)
cp -r external/openpi/src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python*/site-packages/transformers/

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# Build command
CMD="python examples/openpi/eval_libero_intent.py \
    --config-name $CONFIG \
    --checkpoint-dir $CKPT \
    --task-suite $SUITE \
    --num-trials-per-task 20 \
    --norm-stats-from-config \
    --fp32 \
    --out $OUT"

# Add optional schedule (for copred checkpoints)
if [ -n "${SCHEDULE:-}" ]; then
    CMD="$CMD --schedule $SCHEDULE"
fi

# Optional extra args appended verbatim (e.g. "--tasks 3 --video-dir results/rollouts")
if [ -n "${EXTRA_ARGS:-}" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

eval "$CMD"
echo "=== Spatial transfer eval ($TAG) ===" >&2
cat "$OUT" >&2
