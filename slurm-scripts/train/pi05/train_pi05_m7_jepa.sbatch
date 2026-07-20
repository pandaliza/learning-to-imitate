#!/bin/bash
#SBATCH --job-name=m7-jepa
#SBATCH --output=logs/m7_jepa_%j.out
#SBATCH --error=logs/m7_jepa_%j.err
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
export JAX_PLATFORMS=cpu
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FORCE_FP32=1
export NCCL_P2P_DISABLE=1
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# M7-JEPA (Sarvesh): SYMMETRIC joint-embedding alignment (deploy == M0). A trainable predictor (action
# side) and a trainable target encoder (frozen workspace token w + target EE pose = object_states[-1])
# meet in a joint latent. Collapse is prevented the SimSiam way -- a bottleneck predictor + stop-grad
# on each side's target (batch-independent, unlike SigReg which degenerates at batch=2). Unlike
# M5/M6-align's frozen cosine target, both heads train -> the joint space can discard nuisance and keep
# shared action-relevant factors. Watch: align (neg cosine) drops WITHOUT zstd collapsing to 0.
# Set RESUME=<ckpt_dir> (e.g. .../pi05_m7_jepa_confirm/14000) to continue a stopped run from that step.
WEIGHT=${WEIGHT:-0.5}
B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
WSSTACK=${WSSTACK:-$B/ldahiya_checkpoints/workspace_stack_goal/workspace_stack_final.pt}
OUT=${OUT:?set OUT}

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC:-2} \
    examples/openpi/train_pi05_cotrain.py \
    --pi05-config pi05_base_nointent \
    --slot-task-config libero_goal_suite_image_slot_intent_dino_dinosaur \
    --pi05-weights $B/pi05_base_pytorch \
    --align-jepa \
    --align-workspace-stack "$WSSTACK" \
    --align-weight "$WEIGHT" \
    --jepa-dim 256 \
    --steps 30000 \
    --batch-size 2 \
    --accum-steps ${ACCUM:-8} \
    --lr 2e-5 \
    --save-every 2000 \
    ${RESUME:+--resume-from "$RESUME"} \
    --out "$OUT"
