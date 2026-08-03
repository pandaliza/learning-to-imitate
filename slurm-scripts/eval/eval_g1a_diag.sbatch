#!/bin/bash
#SBATCH --job-name=g1a-diag
#SBATCH --output=logs/g1a_diag_%j.out
#SBATCH --error=logs/g1a_diag_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

set -euo pipefail
mkdir -p logs docs/robocasa_m11
cd /home/ldahiya/max_vla/much-ado-about-noising

# M11 diagnostic runs in the main .venv: openpi + transformers(adaRMS) + robocasa
source .venv/bin/activate

# M11 copred: inject per-token adaRMS from transformers_replace
cp -r external/openpi/src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python*/site-packages/transformers/ 2>/dev/null || true

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# G1a offline diagnostic: GT-intent vs model-intent action MSE on held-out demo states
# Usage:
#   CKPT=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/m11_checkpoints/m11_a1_tied_robocasa/30000 \
#     sbatch slurm-scripts/eval/eval_g1a_diag.sbatch

CKPT=${CKPT:-/data/group_data/maxlab/common_datasets/pandaliza/maxvla/m11_checkpoints/m11_a1_tied_robocasa/30000}
SAMPLES=${SAMPLES:-128}
OUT=${OUT:-logs/g1a_diag_results.json}

echo "=== G1a Diagnostic (intent channel probe) ==="
echo "Checkpoint: $CKPT"
echo "Held-out samples: $SAMPLES"
echo "Output: $OUT"

python examples/openpi/diag_intent_causal.py \
    --pi05-config pi05_robocasa_copred \
    --checkpoint-dir "$CKPT" \
    --num-samples "$SAMPLES" \
    --fp32 \
    --out "$OUT"

echo ""
echo "=== Results ==="
cat "$OUT"

# Copy results to report location
cp "$OUT" docs/robocasa_m11/g1a_diag_results.json 2>/dev/null || true

echo ""
echo "[done] G1a diagnostic complete"
