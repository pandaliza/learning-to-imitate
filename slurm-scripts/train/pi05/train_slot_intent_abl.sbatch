#!/bin/bash
#SBATCH --job-name=slot-abl
#SBATCH --output=logs/slot_abl_%j.out
#SBATCH --error=logs/slot_abl_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00

# Tier-1 slot-encoder ablation: single-GPU Stage-1 (small model, cached VL grids). CFG selects the
# ablation config (recon weight / iters / selector), OUT the checkpoint dir. 6k steps is enough to
# read the collapse diagnostic (selector + spatial-attn entropy).
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

CFG=${CFG:?set CFG=libero_goal_suite_image_slot_intent_vl_<variant>}
OUT=${OUT:?set OUT=.../slot_intent_<variant>}
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    examples/openpi/train_slot_intent_vl.py \
    --slot-task-config "$CFG" \
    --steps 6000 --batch-size 16 --lr 1e-4 --save-every 2000 \
    --out "$OUT"
