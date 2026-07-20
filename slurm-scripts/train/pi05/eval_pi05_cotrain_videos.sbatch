#!/bin/bash
#SBATCH --job-name=pi05-cotrain-videos
#SBATCH --output=logs/pi05_cotrain_videos_%j.out
#SBATCH --error=logs/pi05_cotrain_videos_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# Co-train rollout videos (agentview+wrist) for the failure-mode contrast at δ=0/0.05/0.10.

set -euo pipefail
mkdir -p logs/rollout_videos_cotrain
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

CK=${CK:-/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_vl_cotrain_v2/24000}

python examples/openpi/eval_libero_perturbed.py \
    --config-name pi05_base_intent --checkpoint-dir "$CK" \
    --norm-stats-from-config --fp32 \
    --vl-cotrain --intent-stack "$CK/intent_stack.pt" \
    --tasks 1,6,8 --deltas 0.0,0.05,0.10 \
    --num-trials-per-delta 4 --videos-per-cell 4 --video-dir logs/rollout_videos_cotrain \
    --out logs/cotrain_videos.json

echo "=== saved cotrain clips ===" && ls -1 logs/rollout_videos_cotrain/pi05_base_intent_*.mp4 2>/dev/null | wc -l
