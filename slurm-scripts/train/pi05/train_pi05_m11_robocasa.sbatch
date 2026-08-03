#!/bin/bash
#SBATCH --job-name=m11-robocasa
#SBATCH --output=logs/m11_robocasa_%j.out
#SBATCH --error=logs/m11_robocasa_%j.err
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-n5-32,babel-n9-32,babel-o5-16,babel-o5-28

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

# M11 uses the same per-token adaRMS changes as M10 -- sync transformers_replace tree
cp -r external/openpi/src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python*/site-packages/transformers/

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FORCE_FP32=1
export NCCL_P2P_DISABLE=1
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# M11 CO-PREDICTION on RoboCasa (docs/co_prediction.md, ported to RoboCasa kitchen):
# h=8 intent waypoint tokens denoised INSIDE the action expert with independent noise level.
# State 16D (base+eef_rel), action 12D, intent 7D (eef_rel pos3 + quat4).
#
# Arms: A0 (pure BC, intent=0), A1 (tied tau_I=tau_A), A2/A3/A4 (decoupled, one checkpoint).
# KNOBS: ARM={a0,a1,a2a3} -> intent-weight and train-intent-only flags per the arm.
#
# Checkpoints saved under /data/group_data/maxlab/common_datasets/pandaliza/maxvla/m11_checkpoints/<run_name>
# following user directive (2026-07-30): store under common_datasets/pandaliza.
#
# Usage:
#  OUT=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/m11_checkpoints/m11_a0_pure_bc \
#    sbatch train_pi05_m11_robocasa.sbatch
#
#  OUT=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/m11_checkpoints/m11_a1_tied \
#    ARM=a1 sbatch train_pi05_m11_robocasa.sbatch
#
#  OUT=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/m11_checkpoints/m11_a2a3_decoupled \
#    ARM=a2a3 sbatch train_pi05_m11_robocasa.sbatch

B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
OUT=${OUT:?set OUT}
ARM=${ARM:-a2a3}

# Knobs per ARM
case "$ARM" in
  a0)
    # A0: pure behavior cloning, no intent gradient
    INTENT_W=0.0
    TRAIN_INTENT=False
    ;;
  a1)
    # A1: tied tau_I = tau_A (falsification test)
    INTENT_W=1.0
    TRAIN_INTENT=True
    TIED=1
    ;;
  a2a3)
    # A2/A3/A4: decoupled (one checkpoint, three eval schedules)
    INTENT_W=1.0
    TRAIN_INTENT=True
    unset TIED
    ;;
  *)
    echo "ERROR: ARM must be one of {a0,a1,a2a3}, got '$ARM'"
    exit 1
    ;;
esac

mkdir -p "$OUT"

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC:-4} \
    examples/openpi/train_pi05_m11.py \
    --pi05-config pi05_robocasa_copred \
    --pi05-weights $B/pi05_base_pytorch \
    --intent-weight ${INTENT_W} \
    --train-intent-only ${TRAIN_INTENT} \
    --lookahead-stride ${DELTA:-2} \
    ${TIED:+--tied} \
    ${MASK:+--copred-mask "$MASK"} \
    --steps ${STEPS:-30000} \
    --batch-size ${BS:-4} \
    --accum-steps ${ACCUM:-2} \
    --lr ${LR:-5e-5} \
    --save-every ${SAVE_EVERY:-5000} \
    ${RESUME:+--resume-from "$RESUME"} \
    --out "$OUT"

echo "[M11 Training] Completed. Checkpoints in: $OUT"
