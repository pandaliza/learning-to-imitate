#!/bin/bash
#SBATCH --job-name=pi05-cotrain-eval
#SBATCH --output=logs/pi05_cotrain_eval_%j.out
#SBATCH --error=logs/pi05_cotrain_eval_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=babel-n5-32

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
# Policy is PyTorch (model.safetensors) -> keep JAX off the GPU.
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# Co-train @10k SR eval, libero_goal, our finetune convention (no --rotate-images).
# CK = co-train checkpoint (PyTorch model.safetensors, no assets/) -> --norm-stats-from-config.
# INTENT_CKPT = Stage-1 slot ckpt used to warm-start the co-train. NOTE: this old 10k
# checkpoint did NOT save the co-trained flow map (intent_stack bug, since fixed), so the
# intent here is the Stage-1 flow map's -- an APPROXIMATE co-train eval. The clean run
# (group_data) saves the co-trained flow map; swap --intent-ckpt to it once it hits 10k.
CK=${CK:-/data/user_data/ldahiya/openpi-checkpoints/pi05_cotrain/10000}
INTENT_CKPT=${INTENT_CKPT:-/data/user_data/ldahiya/libero-train/libero-goal-suite-image/slot_intent/models/model_best.pt}
OUT=${OUT:-logs/eval_cotrain10k.json}

python examples/openpi/eval_libero_intent.py \
    --config-name pi05_libero_intent \
    --checkpoint-dir "$CK" \
    --norm-stats-from-config \
    --intent \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt "$INTENT_CKPT" \
    --task-suite libero_goal \
    --num-trials-per-task 20 \
    --out "$OUT"
