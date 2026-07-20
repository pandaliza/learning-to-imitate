#!/bin/bash
#SBATCH --job-name=slot-intent-dino
#SBATCH --output=logs/slot_intent_dino_%j.out
#SBATCH --error=logs/slot_intent_dino_%j.err
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00

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

# Stage-1 decoupled slot-intent, slot encoder reads DINOv2 patch grids (256x768). 2x L40S DDP,
# effective batch = 2 x 16 = 32. Identical pipeline to the VL run, only the cached grids differ.
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    examples/openpi/train_slot_intent_vl.py \
    --slot-task-config libero_goal_suite_image_slot_intent_dino \
    --steps 40000 \
    --batch-size 16 \
    --lr 1e-4 \
    --save-every 2000 \
    --out /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/slot_intent_dino
