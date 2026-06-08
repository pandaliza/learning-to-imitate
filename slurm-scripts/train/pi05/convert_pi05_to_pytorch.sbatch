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

# Convert pi05_libero (JAX/orbax) -> PyTorch safetensors for the co-train PyTorch path.
# config_name=pi05_libero -> output has no intent_proj; co-train loads strict=False.
python external/openpi/examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/pi05_libero \
    --config_name pi05_libero \
    --output_path /data/user_data/ldahiya/pi05_libero_pytorch
