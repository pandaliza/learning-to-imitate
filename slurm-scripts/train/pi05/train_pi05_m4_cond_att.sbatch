#!/bin/bash
#SBATCH --job-name=pi05-m4-cond-att
#SBATCH --output=logs/pi05_m4_cond_att_%j.out
#SBATCH --error=logs/pi05_m4_cond_att_%j.err
#SBATCH --gres=gpu:L40S:4
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

# FIXED-M4 (aux head -> deployable conditioning). pi05_base_intent (action head reads intent).
# Per step: stop-grad tap vl_mean -> aux_head(vl_mean)=z_hat -> CONDITION the action head on a
# DETACHED z_hat; aux loss = mse(z_hat, frozen z*). The stop-grad tap means the aux loss never
# touches the LoRA-VL backbone -> none of aux-only M4's corruption/divergence. + cosine LR decay.
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    examples/openpi/train_pi05_cotrain.py \
    --pi05-config pi05_base_intent \
    --slot-task-config libero_goal_suite_image_slot_intent_vl \
    --pi05-weights /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/pi05_base_pytorch \
    --aux-intent --aux-condition --intent-attach \
    --aux-intent-stack /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/slot_intent_vl/intent_stack_32000.pt \
    --aux-intent-weight 1.0 \
    --steps 30000 \
    --batch-size 2 \
    --accum-steps 4 \
    --lr 5e-5 \
    --save-every 2000 \
    --out /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_m4_cond_att
