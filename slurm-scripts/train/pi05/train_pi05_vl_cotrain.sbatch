#!/bin/bash
#SBATCH --job-name=pi05-vl-cotrain
#SBATCH --output=logs/pi05_vl_cotrain_%j.out
#SBATCH --error=logs/pi05_vl_cotrain_%j.err
#SBATCH --gres=gpu:L40S:6
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-n5-32,babel-n9-32

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu                               # keep JAX off the GPUs (torch/DDP own them)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# pi05_base (un-finetuned generalist) overflows bf16 on LIBERO input -> NaN. Run fp32.
export FORCE_FP32=1
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# VL-grounded intent co-train (V1): pi05_base + LoRA-VL, action head full finetune.
# The flow map conditions on the LIVE LoRA-VL mean image-token feature (tapped from the
# Pi0.5 prefix), trained to match the ResNet18 slot target z*; the DEPLOYABLE z_hat is
# fed to the action head (detached for V1). LoRA freeze = gemma LLM base frozen, train
# LoRA + SigLIP + action expert + intent_proj + flow map + vl_proj. Warmstart slot from Stage 1.
# 6x L40S DDP, batch 4/GPU x6 x accum 1 = eff batch 24. Per-task prompts + un-normalize fix.
torchrun --standalone --nnodes=1 --nproc_per_node=6 \
    examples/openpi/train_pi05_cotrain.py \
    --pi05-config pi05_base_intent \
    --slot-task-config libero_goal_suite_image_slot_intent \
    --warmstart /data/user_data/ldahiya/libero-train/libero-goal-suite-image/slot_intent/models/model_best.pt \
    --pi05-weights /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/pi05_base_pytorch \
    --vl-intent \
    --steps 30000 \
    --batch-size 4 \
    --accum-steps 1 \
    --save-every 2000 \
    --out /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_vl_cotrain_v2
