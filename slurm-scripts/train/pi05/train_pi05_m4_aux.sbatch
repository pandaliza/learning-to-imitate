#!/bin/bash
#SBATCH --job-name=pi05-m4-aux
#SBATCH --output=logs/pi05_m4_aux_%j.out
#SBATCH --error=logs/pi05_m4_aux_%j.err
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=48:00:00
# o5-16/o5-28 hung NCCL's first intra-node collective (broken GPU P2P); avoid them.
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
export NCCL_P2P_DISABLE=1                               # some nodes hang on intra-node GPU P2P; use SHM transport
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# M4 (aux-loss intent): NO conditioning (pi05_base_nointent). An aux head on the live LoRA-VL
# mean is trained to PREDICT the FROZEN VL slot target z* (from intent_stack_32000.pt, on the
# cached future-frame VL grids) -> shapes LoRA-VL to be anticipatory. The action path never sees
# an intent, so deploy == M0 (no generator).
# Batch 2/GPU (NOT 4): M4 runs the VL prefix with grad for the action forward (the single-forward
# tap), so batch 4 OOMs. Effective batch 32 (2/GPU x 4 GPU x accum 4) -- the first run at eff 16
# / lr 5e-5 diverged (action loss ran away ~step 2300); bigger batch cuts the gradient noise that
# drove it, lower lr takes smaller steps, and aux-weight 0.1 keeps LoRA-VL shaping gentle.
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    examples/openpi/train_pi05_cotrain.py \
    --pi05-config pi05_base_nointent \
    --slot-task-config libero_goal_suite_image_slot_intent_vl \
    --pi05-weights /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/pi05_base_pytorch \
    --aux-intent \
    --aux-intent-stack /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/slot_intent_vl/intent_stack_32000.pt \
    --aux-intent-weight 0.1 \
    --steps 30000 \
    --batch-size 2 \
    --accum-steps 4 \
    --lr 2e-5 \
    --save-every 2000 \
    --out /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_m4_aux
