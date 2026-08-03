#!/bin/bash
#SBATCH --job-name=eval-m10
#SBATCH --output=logs/eval_m10_%j.out
#SBATCH --error=logs/eval_m10_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

# M10 deploys WITH the intent tokens (unlike M8/M9): config must be pi05_base_copred and the venv
# transformers copy must carry the per-token adaRMS (3-D cond) -- re-sync it here like training does.
cp -r external/openpi/src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python*/site-packages/transformers/

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# SR eval of an M10 co-prediction checkpoint under one inference schedule
# (docs/co_prediction.md section 4: s1 intent-first = A3, s2 joint = A2/A1, s3 action-first = A4).
B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
CKPT=${CKPT:?set CKPT}
SCHEDULE=${SCHEDULE:?set SCHEDULE (s1|s2|s3)}
TRIALS=${TRIALS:-20}
OUT=${OUT:?set OUT}

python examples/openpi/eval_libero_intent.py \
    --config-name pi05_base_copred \
    --checkpoint-dir "$CKPT" \
    --task-suite libero_goal \
    --num-trials-per-task $TRIALS \
    --norm-stats-from-config \
    --fp32 \
    --schedule "$SCHEDULE" \
    --num-intent-steps ${KI:-4} \
    --out "$OUT"

echo "=== SR result ($SCHEDULE) ==="; cat "$OUT"
