#!/bin/bash
#SBATCH --job-name=pi05-cotrain
#SBATCH --output=logs/pi05_cotrain_%j.out
#SBATCH --error=logs/pi05_cotrain_%j.err
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-n5-32

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
# Force JAX (imported via openpi config) onto CPU so it doesn't grab the GPUs that
# torch/DDP needs — otherwise torchrun's ranks hit "CUDA device busy or unavailable".
export JAX_PLATFORMS=cpu
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reclaim fragmented mem -> fit BS16 on 48GB L40S
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# End-to-end co-train: Pi0.5 action head + MIP slot encoder + flow map.
# 4x L40S data-parallel (DDP): BS8/GPU x 4 = effective batch 32, no accumulation.
# Warmstart the intent stack from the Stage-1 slot checkpoint (grounded init +
# comparability with the decoupled arm). Action-head-only freeze, grad checkpointing.
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    examples/openpi/train_pi05_cotrain.py \
    --pi05-config pi05_libero_intent \
    --slot-task-config libero_goal_suite_image_slot_intent \
    --warmstart /data/user_data/ldahiya/libero-train/libero-goal-suite-image/slot_intent/models/model_best.pt \
    --pi05-weights /data/user_data/ldahiya/pi05_libero_pytorch \
    --steps 30000 \
    --batch-size 8 \
    --accum-steps 1 \
    --save-every 2000 \
    --out /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_cotrain
