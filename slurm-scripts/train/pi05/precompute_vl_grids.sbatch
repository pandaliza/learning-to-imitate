#!/bin/bash
#SBATCH --job-name=precompute-vl-grids
#SBATCH --output=logs/precompute_vl_grids_%j.out
#SBATCH --error=logs/precompute_vl_grids_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export JAX_PLATFORMS=cpu                                # torch owns the GPU
export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/nb_${SLURM_JOB_ID:-m}_XXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"' EXIT

# Frozen pi05_base VL agentview token grids for the VL-grounded slot-intent (decoupled) variant.
# ~63.7k frames -> ~67GB fp16, one .npy per (task,demo). Resumable (skips existing).
python examples/openpi/precompute_vl_grids.py \
    --pi05-weights /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/pi05_base_pytorch \
    --out /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/vl_cache_goal_agentview \
    --batch 16
