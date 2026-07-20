#!/bin/bash
#SBATCH --job-name=m6-align
#SBATCH --output=logs/m6_align_%j.out
#SBATCH --error=logs/m6_align_%j.err
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

# M6-align: same REPA mechanism as M5 (deploy == M0, no conditioning) but the align TARGET is the
# frozen WORKSPACE token (MolmoPoint-8B saliency-supervised, task-selective) instead of the slot z*.
# The key readout is the align-loss curve: if it does NOT saturate instantly (as M5's did on the
# redundant z*), the selective saliency carries signal the action expert doesn't already encode.
# Runs in the main .venv (workspace encoder is plain torch; no transformers/Molmo overlay needed).
WEIGHT=${WEIGHT:-0.5}
B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
WSSTACK=${WSSTACK:-$B/ldahiya_checkpoints/workspace_stack_goal/workspace_stack_final.pt}
OUT=${OUT:?set OUT}

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC:-2} \
    examples/openpi/train_pi05_cotrain.py \
    --pi05-config pi05_base_nointent \
    --slot-task-config libero_goal_suite_image_slot_intent_dino_dinosaur \
    --pi05-weights $B/pi05_base_pytorch \
    --align-intent \
    --align-workspace-stack "$WSSTACK" \
    --align-weight "$WEIGHT" \
    --steps 30000 \
    --batch-size 2 \
    --accum-steps ${ACCUM:-8} \
    --lr 2e-5 \
    --save-every 2000 \
    --out "$OUT"
