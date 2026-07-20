#!/bin/bash
#SBATCH --job-name=eval-m8-6k
#SBATCH --output=logs/eval_m8_6k_%j.out
#SBATCH --error=logs/eval_m8_6k_%j.err
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

# SR eval of the M8-WSM stage-2 checkpoint. M8 deploys as BASELINE pi0.5 -- w and the intent head are
# training-only (never in the inference graph) -- so this is a plain pi05_base_nointent rollout on the
# finetuned model.safetensors. --norm-stats-from-config: PyTorch cotrain ckpts save no assets/ dir.
B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
CKPT=${CKPT:-$B/ldahiya_checkpoints/pi05_m8_wsm_goal_run1/6000}
TRIALS=${TRIALS:-20}
OUT=${OUT:-logs/eval_m8_wsm_6000.json}

python examples/openpi/eval_libero_intent.py \
    --config-name pi05_base_nointent \
    --checkpoint-dir "$CKPT" \
    --task-suite libero_goal \
    --num-trials-per-task $TRIALS \
    --norm-stats-from-config \
    --fp32 \
    --out "$OUT"

echo "=== SR result ==="; cat "$OUT"
