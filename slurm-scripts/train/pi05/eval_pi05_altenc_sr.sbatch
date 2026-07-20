#!/bin/bash
#SBATCH --job-name=pi05-altenc-sr
#SBATCH --output=logs/pi05_altenc_sr_%j.out
#SBATCH --error=logs/pi05_altenc_sr_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00

# DINOv2/DynaFLIP-grounded intent SR. JAX policy + alt-encoder PyTorch tap share the GPU -> cap XLA.
# Mirrors the M3 eval; --encoder-tap replaces --frozen-base-weights (generator was trained on the
# alt encoder's agentview grid, not pi05 VL). No --fp32 (JAX policy handles its own precision).
set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JAX_COMPILATION_CACHE_DIR=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/jax_cache_eval
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

VARIANT=${VARIANT:?set VARIANT=dino or dynaflip}
STEP=${STEP:?set STEP}
TRIALS=${TRIALS:-20}
ENC=${ENC:-$([ "$VARIANT" = dino ] && echo dinov2 || echo dynaflip)}
EXP=${EXP:-base_intent_${VARIANT}_goal_run1}
B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
# Overridable for the DINOSAUR (feature-recon) arm: CFG/CK/STACK/TASKCFG/OUT.
CFG=${CFG:-pi05_base_intent_${VARIANT}}
CK=${CK:-$B/ldahiya_checkpoints/pi05_base_intent_${VARIANT}/${EXP}/${STEP}}
STACK=${STACK:-$B/ldahiya_checkpoints/slot_intent_${VARIANT}/intent_stack_32000.pt}
TASKCFG=${TASKCFG:-libero_goal_suite_image_slot_intent_${VARIANT}}
OUT=${OUT:-logs/${VARIANT}_sr_goal_${STEP}.json}

SUITE=${SUITE:-libero_goal}
python examples/openpi/eval_libero_intent.py \
    --config-name "$CFG" --checkpoint-dir "$CK" \
    --task-suite "$SUITE" --num-trials-per-task "$TRIALS" \
    --vl-cotrain \
    --intent-task-config "$TASKCFG" \
    --intent-stack "$STACK" \
    --encoder-tap "$ENC" \
    --out "$OUT"
