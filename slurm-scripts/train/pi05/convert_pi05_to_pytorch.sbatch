#!/bin/bash
#SBATCH --job-name=pi05-jax2torch
#SBATCH --output=logs/pi05_jax2torch_%j.out
#SBATCH --error=logs/pi05_jax2torch_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=3:00:00

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# Convert a Pi0.5 JAX/orbax checkpoint -> PyTorch safetensors for the co-train PyTorch path.
# Parameterized (override via: sbatch --export=ALL,CKPT=...,CONFIG=...,OUT=...). config_name
# picks a non-LoRA/non-intent architecture (e.g. pi05_libero) compatible with the base weights;
# the co-train loads it strict=False so LoRA/intent_proj are backfilled fresh.
CKPT=${CKPT:-/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/pi05_base}
CONFIG=${CONFIG:-pi05_libero}
OUT=${OUT:-/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/pi05_base_pytorch}
python external/openpi/examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir "$CKPT" \
    --config_name "$CONFIG" \
    --output_path "$OUT"
