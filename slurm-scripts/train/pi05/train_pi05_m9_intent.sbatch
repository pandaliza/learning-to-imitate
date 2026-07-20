#!/bin/bash
#SBATCH --job-name=m9-intent
#SBATCH --output=logs/m9_intent_%j.out
#SBATCH --error=logs/m9_intent_%j.err
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-n5-32,babel-n9-32,babel-o5-16,babel-o5-28

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

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

# M9-INTENT (steer_intent): M8 MINUS the workspace token. A 3-layer head on the action-expert
# PENULTIMATE predicts the h=8 future-EEF trajectory; its gradient flows back into the shared trunk.
# No w, no CFG conditioning, no JEPA target -> and therefore no stage-1 encoder, no MolmoPoint labels
# and no w precompute (this arm needs NOTHING but the hdf5, which is why it ports to robocasa /
# libero-long unchanged). Train-time only: deploy == M0, so eval with eval_libero_intent.py WITHOUT
# --intent, exactly like M8.
#
# Tests the claim that the aux SUPERVISION is what shapes the trunk and the workspace machinery is
# incidental in a memory-free setting. Read against: M8 w=0.25 (job 9308258) at MATCHED steps, and the
# pure-BC control. Note this is a REMOVAL from M8, not a replacement -- nothing fills w's seat -- so any
# SR delta between M9 and M8 w=0.25 is attributable to w alone.
#
# Watch: intent_fm starts ~1.24 (untrained velocity field on a ~unit-variance target) and should fall;
# action should track the other arms. DECODER=mse + TARGET=mean reproduces M8's exact objective minus w
# (use that to separate "removed w" from "changed the decoder/target").
B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
OUT=${OUT:?set OUT}

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC:-4} \
    examples/openpi/train_pi05_m9.py \
    --pi05-config pi05_base_nointent \
    --slot-task-config libero_goal_suite_image_slot_intent_vl \
    --pi05-weights $B/pi05_base_pytorch \
    --intent-flow \
    --intent-flow-decoder ${DECODER:-flow} \
    --intent-flow-target ${TARGET:-concat} \
    --intent-flow-weight ${INTENT_W:-0.25} \
    --steps ${STEPS:-30000} \
    --batch-size ${BS:-2} \
    --accum-steps ${ACCUM:-4} \
    --lr ${LR:-5e-5} \
    --save-every 2000 \
    ${RESUME:+--resume-from "$RESUME"} \
    --out "$OUT"
