#!/bin/bash
#SBATCH --job-name=pi05-base-videos
#SBATCH --output=logs/pi05_base_videos_%j.out
#SBATCH --error=logs/pi05_base_videos_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# Qualitative rollout videos (agentview + wrist side-by-side) of the failure modes.
# Captures the contrast: δ=0.00 (in-distribution, reaches & succeeds) vs δ=0.05/0.10
# (OOD, where the cascade shows ~100% no_reach). Filenames tag the failure stage
# (success / no_reach / no_grasp / no_place / invalid_collision) so failures are findable.
# num-trials == videos-per-cell, so every rolled episode is saved (fast).

set -euo pipefail
mkdir -p logs/rollout_videos
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export JAX_COMPILATION_CACHE_DIR=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/jax_cache_eval
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

CKPT_BASE=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints
NOINTENT_CKPT=$CKPT_BASE/pi05_base_nointent/base_nointent_goal_run1/29999
INTENT_CKPT=$CKPT_BASE/pi05_base_intent/base_intent_goal_run1/29999
FLOW_MAP=/data/user_data/ldahiya/libero-train/libero-goal-suite-image/slot_intent/models/model_best.pt

TASKS="1,6,8"
DELTAS="0.0,0.05,0.10"
N=4   # trials per cell == videos saved per cell (4 evenly-spaced directions)

COMMON="--task-suite libero_goal --tasks $TASKS --deltas $DELTAS \
        --num-trials-per-delta $N --videos-per-cell $N --video-dir logs/rollout_videos"

# control
python examples/openpi/eval_libero_perturbed.py $COMMON \
    --config-name pi05_base_nointent \
    --checkpoint-dir "$NOINTENT_CKPT" \
    --out logs/videos_nointent.json

# frozen-intent
python examples/openpi/eval_libero_perturbed.py $COMMON \
    --config-name pi05_base_intent \
    --checkpoint-dir "$INTENT_CKPT" \
    --intent \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt "$FLOW_MAP" \
    --out logs/videos_intent.json

echo "===================== SAVED VIDEOS ====================="
ls -1 logs/rollout_videos/*.mp4 | sort
echo "total: $(ls -1 logs/rollout_videos/*.mp4 | wc -l) clips"
