#!/bin/bash
#SBATCH --job-name=ev-zs
#SBATCH --output=logs/eval_zs_%j.out
#SBATCH --error=logs/eval_zs_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

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

# Generic zero-shot / baseline eval: a non-LoRA pi0.5 checkpoint deployed as vanilla pi0.5 (no
# intent, no schedule). CONFIG supplies the architecture + norm stats; CKPT the weights.
CONFIG=${CONFIG:-pi05_libero_nointent}
CKPT=${CKPT:?set CKPT}
TRIALS=${TRIALS:-20}
OUT=${OUT:?set OUT}

# Stock PI checkpoints (pi05_base / pi05_libero) trained on 180-rotated LIBERO images; our
# finetunes did not. ROTATE=1 adds --rotate-images (required for the PI-convention models).
python examples/openpi/eval_libero_intent.py \
    --config-name "$CONFIG" \
    --checkpoint-dir "$CKPT" \
    --task-suite libero_goal \
    --num-trials-per-task $TRIALS \
    --norm-stats-from-config \
    --fp32 \
    ${ROTATE:+--rotate-images} \
    --out "$OUT"

echo "=== SR result ($CKPT) ==="; cat "$OUT"
