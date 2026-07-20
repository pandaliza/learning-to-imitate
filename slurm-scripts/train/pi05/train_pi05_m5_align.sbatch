#!/bin/bash
#SBATCH --job-name=m5-repa-align
#SBATCH --output=logs/m5_align_%j.out
#SBATCH --error=logs/m5_align_%j.err
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-n5-32,babel-n9-32,babel-o5-16,babel-o5-28

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu                                # torch/DDP own the GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FORCE_FP32=1                                     # pi05_base overflows bf16 on LIBERO
export NCCL_P2P_DISABLE=1
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# M5 / REPA-align: NO conditioning (pi05_base_nointent, deploy == M0). A projector aligns the
# action expert's penultimate hidden to the FROZEN DINOSAUR slot target z* via a cosine loss
# (Yu et al. 2024). Bounded + localized to the action expert -> avoids the aux-M4 backbone
# corruption. Framed as a SAMPLE-EFFICIENCY study: save every 2k, eval SR-vs-steps vs M0.
# Batch 2/GPU (pi05_base fp32 action forward w/ VL grad); effective batch 32 (2 x 2 GPU x accum 8).
WEIGHT=${WEIGHT:-0.5}
B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
STACK=${STACK:-$B/ldahiya_checkpoints/slot_intent_dino_dinosaur_60k/intent_stack_98000.pt}
OUT=${OUT:?set OUT}

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    examples/openpi/train_pi05_cotrain.py \
    --pi05-config pi05_base_nointent \
    --slot-task-config libero_goal_suite_image_slot_intent_dino_dinosaur \
    --pi05-weights $B/pi05_base_pytorch \
    --align-intent \
    --align-intent-stack "$STACK" \
    --align-weight "$WEIGHT" \
    --steps 30000 \
    --batch-size 2 \
    --accum-steps 8 \
    --lr 2e-5 \
    --save-every 2000 \
    --out "$OUT"
