#!/bin/bash
#SBATCH --job-name=slot-dinosaur
#SBATCH --output=logs/slot_dinosaur_%j.out
#SBATCH --error=logs/slot_dinosaur_%j.err
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate
export JAX_PLATFORMS=cpu
export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/nb_${SLURM_JOB_ID:-m}_XXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"' EXIT
CFG=${CFG:-libero_goal_suite_image_slot_intent_dino_dinosaur_strong}
OUT=${OUT:?set OUT}
STEPS=${STEPS:-30000}
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    examples/openpi/train_slot_intent_vl.py \
    --slot-task-config "$CFG" --steps "$STEPS" --batch-size 16 --lr 1e-4 --save-every 2000 ${WARM:+--warmstart "$WARM"} --out "$OUT"
