#!/bin/bash
#SBATCH --job-name=m10-copred
#SBATCH --output=logs/m10_copred_%j.out
#SBATCH --error=logs/m10_copred_%j.err
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-n5-32,babel-n9-32,babel-o5-16,babel-o5-28

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

# M10 changes transformers_replace (per-token adaRMS in modeling_gemma.py) -- the venv carries a
# COPY of that tree, so re-sync it or the run silently uses the old broadcast-only RMSNorm.
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

# M10 CO-PREDICTION (docs/co_prediction.md): h=8 intent waypoint tokens denoised INSIDE the action
# expert with an independent noise level (diffusion-forcing style). No aux head, no w, no stage-1 --
# the intent projections live in model.safetensors. Deploy graph INCLUDES the intent tokens: eval
# with eval_libero_intent.py --schedule {s1,s2,s3} (s1 = intent-first, the method).
#
# Arms: TIED=1 -> A1 (tau_I = tau_A, tests co-training alone). Default decoupled+stratified ->
# one checkpoint serving A2 (s2) / A3 (s1) / A4 (s3). MASK=t -> C1 two-stream; MASK=b1 -> B1
# pooled-adaLN conditioning (no token attention -- pair with --schedule s1 at eval); MASK=xattn ->
# gated cross-attention on action tokens instead of adaLN (the "cross-attn vs adaLN0" comparison
# against b1; schedule-agnostic, no s1 pairing needed). Read against A0 = M9-penult 79.0 @ 12k and
# the A0b pure-BC control.
#
# Watch: intent loss ~1.2 at start (untrained velocity on ~unit-variance target) and falling;
# action tracking the M9 curve. REMEMBER (M8/M9 lesson): converged losses do NOT predict SR.
B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
OUT=${OUT:?set OUT}

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC:-4} \
    examples/openpi/train_pi05_m10.py \
    --pi05-config pi05_base_copred \
    --slot-task-config libero_goal_suite_image_slot_intent_vl \
    --pi05-weights $B/pi05_base_pytorch \
    --intent-weight ${INTENT_W:-1.0} \
    --lookahead-stride ${DELTA:-2} \
    ${TIED:+--tied} \
    ${MASK:+--copred-mask "$MASK"} \
    --steps ${STEPS:-30000} \
    --batch-size ${BS:-2} \
    --accum-steps ${ACCUM:-4} \
    --lr ${LR:-5e-5} \
    --save-every 2000 \
    ${RESUME:+--resume-from "$RESUME"} \
    --out "$OUT"
