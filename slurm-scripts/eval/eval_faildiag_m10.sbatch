#!/bin/bash
#SBATCH --job-name=faildiag-m10
#SBATCH --output=logs/eval_faildiag_%x_%j.out
#SBATCH --error=logs/eval_faildiag_%x_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Failure-mode diagnostic: canonically-protocoded cascade (reach -> grasp -> place)
# for M10 co-prediction arms. Env knobs: CONFIG, CKPT, SCHEDULE (optional), OUT, TAG

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

# Environment variable defaults
CONFIG=${CONFIG:-pi05_base_copred}
CKPT=${CKPT:-/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_m10_copred_archive/44000}
SCHEDULE=${SCHEDULE:-s2}
OUT=${OUT:-logs/eval_faildiag_m10.json}
TAG=${TAG:-m10}

# Standard exports
export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export JAX_COMPILATION_CACHE_DIR=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/jax_cache_eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Temp dirs with trap
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# Copred transformers sync (required for copred deploys)
cp -r external/openpi/src/openpi/models_pytorch/transformers_replace/* .venv/lib/python*/site-packages/transformers/

# Canonical protocol: tasks 1,6,8, deltas 0.0/0.05/0.10, 20 trials, norm-stats, fp32
# Optional --schedule for copred arms
CMD_BASE="--task-suite libero_goal --tasks 1,6,8 --deltas 0.0,0.05,0.10 --num-trials-per-delta 20 --norm-stats-from-config --fp32"

# EXTRA_ARGS: optional args appended last (argparse: later flags override CMD_BASE)
if [ -z "$SCHEDULE" ] || [ "$SCHEDULE" == "none" ]; then
    python examples/openpi/eval_libero_perturbed.py \
        --config-name "$CONFIG" \
        --checkpoint-dir "$CKPT" \
        $CMD_BASE \
        --out "$OUT" ${EXTRA_ARGS:-}
else
    python examples/openpi/eval_libero_perturbed.py \
        --config-name "$CONFIG" \
        --checkpoint-dir "$CKPT" \
        --schedule "$SCHEDULE" \
        $CMD_BASE \
        --out "$OUT" ${EXTRA_ARGS:-}
fi
