#!/bin/bash
#SBATCH --job-name=pi05-m3-sr
#SBATCH --output=logs/pi05_m3_sr_%j.out
#SBATCH --error=logs/pi05_m3_sr_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00

# M3 (VL-grounded frozen intent) in-distribution SR, all 10 libero_goal tasks x 20 trials.
# JAX M3 policy + frozen pi05_base PyTorch (VL tap) share the GPU -> cap XLA at 0.45.

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JAX_COMPILATION_CACHE_DIR=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/jax_cache_eval
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
CK=${CK:-$B/ldahiya_checkpoints/pi05_base_intent_m3/base_intent_m3_goal_run1/29999}
VLSTACK=${VLSTACK:-$B/ldahiya_checkpoints/slot_intent_vl/intent_stack_32000.pt}
FROZEN=${FROZEN:-$B/pi05_base_pytorch}

python examples/openpi/eval_libero_intent.py \
    --config-name pi05_base_intent_m3 --checkpoint-dir "$CK" \
    --task-suite libero_goal --num-trials-per-task 20 \
    --vl-cotrain \
    --intent-task-config libero_goal_suite_image_slot_intent_vl \
    --intent-stack "$VLSTACK" \
    --frozen-base-weights "$FROZEN" --frozen-base-config pi05_base_nointent \
    --out logs/m3_sr_goal.json
