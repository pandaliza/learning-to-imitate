#!/bin/bash
#SBATCH --job-name=pi05-m4-sr
#SBATCH --output=logs/pi05_m4_sr_%j.out
#SBATCH --error=logs/pi05_m4_sr_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# M4 (aux-loss) in-distribution SR. M4 deploys like M0: pi05_base_nointent, NO intent / NO aux
# head at inference (aux head was training-only). PyTorch ckpt -> norm stats from config, fp32.
# Parametric: CFG/CK/OUT/TRIALS overridable for A/B diagnostics (e.g. M2-no-intent control).
STEP=${STEP:-14000}
CFG=${CFG:-pi05_base_nointent}
CK=${CK:-/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_m4_aux/${STEP}}
OUT=${OUT:-logs/m4_sr_goal_${STEP}.json}
TRIALS=${TRIALS:-20}

python examples/openpi/eval_libero_intent.py \
    --config-name "$CFG" --checkpoint-dir "$CK" \
    --norm-stats-from-config --fp32 \
    --task-suite libero_goal --num-trials-per-task "$TRIALS" \
    --out "$OUT"
