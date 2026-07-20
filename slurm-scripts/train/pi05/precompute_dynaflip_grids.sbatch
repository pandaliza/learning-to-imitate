#!/bin/bash
#SBATCH --job-name=precompute-dynaflip
#SBATCH --output=logs/precompute_dynaflip_%j.out
#SBATCH --error=logs/precompute_dynaflip_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/nb_${SLURM_JOB_ID:-m}_XXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"' EXIT

# DynaFLIP patch grids (256x768) per agentview frame for the DynaFLIP slot-intent variant.
# vision_outputs(pixel_values).last_hidden_state -> (B,256,768). ~25GB fp16. Resumable.
python examples/openpi/precompute_encoder_grids.py \
    --encoder dynaflip \
    --out /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/vl_cache_goal_dynaflip \
    --batch 64
