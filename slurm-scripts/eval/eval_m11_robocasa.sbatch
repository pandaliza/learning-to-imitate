#!/bin/bash
#SBATCH --job-name=eval-m11
#SBATCH --output=logs/eval_m11_%j.out
#SBATCH --error=logs/eval_m11_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising

# M11 eval runs entirely in the main .venv: openpi + transformers(adaRMS) + robocasa
# (external/robocasa + external/robosuite fork installed editable, 2026-08-01).
source .venv/bin/activate

# M11 deploys WITH intent tokens: config must be pi05_base_copred, and venv transformers must
# carry per-token adaRMS (3-D cond) -- re-sync it.
cp -r external/openpi/src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python*/site-packages/transformers/ 2>/dev/null || true

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# SR eval of an M11 co-prediction checkpoint on RoboCasa kitchen.
# Inference schedules (docs/co_prediction.md section 4):
#   s1 = intent-first (A3)
#   s2 = joint (A2/A1)
#   s3 = action-first (A4)
#
# Usage:
#  CKPT=/path/to/checkpoint/26000 SCHEDULE=s1 TASKSET=atomic_seen TRIALS=5 \
#    OUT=logs/eval_m11_a3_atomic_s1.json sbatch eval_m11_robocasa.sbatch
#
#  CKPT=... SCHEDULE=s2 TASKSET=all TRIALS=20 \
#    OUT=logs/eval_m11_a2_all_s2.json sbatch eval_m11_robocasa.sbatch

CKPT=${CKPT:?set CKPT (checkpoint dir)}
SCHEDULE=${SCHEDULE:-s2}
TASKSET=${TASKSET:-all}
TRIALS=${TRIALS:-5}
OUT=${OUT:?set OUT (output JSON file)}

python examples/openpi/eval_robocasa_intent.py \
    --config-name pi05_robocasa_copred \
    --checkpoint-dir "$CKPT" \
    --task-set "$TASKSET" \
    ${TASKS:+--tasks "$TASKS"} \
    ${VIDDIR:+--save-video-dir "$VIDDIR"} \
    --num-trials-per-task $TRIALS \
    $([ "$SCHEDULE" != standard ] && echo --schedule "$SCHEDULE") \
    --num-intent-steps ${KI:-4} \
    --out "$OUT"

echo "=== M11 RoboCasa Eval ($SCHEDULE, $TASKSET) ===" && cat "$OUT"
