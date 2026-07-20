#!/bin/bash
#SBATCH --job-name=m8-wsm
#SBATCH --output=logs/m8_wsm_%j.out
#SBATCH --error=logs/m8_wsm_%j.err
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

# M8-WSM (steer_intent): a 3-layer MLP on the action-expert PENULTIMATE predicts INTENT (future EEF pose).
# w_t (frozen causal workspace latent, past-only) CFG-conditions MLP layer 1; w_{t+1} is a JEPA alignment
# target for MLP layer 2. Both losses are TRAIN-TIME ONLY -> deploy == M0 (w never in the inference graph).
# The workspace encoder (stage-1) is frozen and PRECOMPUTED into per-demo w.npy (precompute_w.py); this run
# never loads the encoder. Watch: wsm intent (MSE) + jepa (1-cos) both drop; cos rises toward 1.
B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
WCACHE=${WCACHE:-$B/w_cache_goal}
OUT=${OUT:?set OUT}

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC:-4} \
    examples/openpi/train_pi05_cotrain.py \
    --pi05-config pi05_base_nointent \
    --slot-task-config libero_goal_suite_image_slot_intent_vl \
    --pi05-weights $B/pi05_base_pytorch \
    --wsm-intent \
    --wsm-w-cache-dir "$WCACHE" \
    --wsm-jepa-weight ${JEPA_W:-1.0} \
    --wsm-intent-weight ${INTENT_W:-1.0} \
    --wsm-sigreg-weight ${SIGREG_W:-0.0} \
    --steps ${STEPS:-30000} \
    --batch-size ${BS:-2} \
    --accum-steps ${ACCUM:-4} \
    --lr ${LR:-5e-5} \
    --save-every 2000 \
    ${RESUME:+--resume-from "$RESUME"} \
    --out "$OUT"
