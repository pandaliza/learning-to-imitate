#!/bin/bash
#SBATCH --job-name=pi05-altenc-suite
#SBATCH --output=logs/pi05_altenc_suite_%j.out
#SBATCH --error=logs/pi05_altenc_suite_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00

# Full OOD suite for a DINOv2/DynaFLIP-grounded frozen-intent arm (mirrors M3 but --encoder-tap).
# Runs: perturb(+validity+cascade) | spatial transfer | failure cascade | override | videos.
# Usage: VARIANT=dino STEP=10000 sbatch [-p debug] eval_pi05_altenc_suite.sbatch
#        EXPS="perturb spatial faildiag override videos" to subset.

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
STEP=${STEP:-10000}
EXPS=${EXPS:-"perturb spatial faildiag override videos"}
ENC=$([ "$VARIANT" = dino ] && echo dinov2 || echo dynaflip)
B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
CK=$B/ldahiya_checkpoints/pi05_base_intent_${VARIANT}/base_intent_${VARIANT}_goal_run1/${STEP}
STACK=$B/ldahiya_checkpoints/slot_intent_${VARIANT}/intent_stack_32000.pt
TAG=${VARIANT}_${STEP}
FLAGS="--config-name pi05_base_intent_${VARIANT} --checkpoint-dir $CK --vl-cotrain \
  --intent-task-config libero_goal_suite_image_slot_intent_${VARIANT} --intent-stack $STACK \
  --encoder-tap $ENC"
EVAL="python examples/openpi/eval_libero_perturbed.py $FLAGS"

for E in $EXPS; do
  echo "########## ${TAG}: ${E} ##########"
  case $E in
    perturb)
      $EVAL --deltas 0.0,0.02,0.05,0.10,0.15,0.20 --num-trials-per-delta 10 \
            --out logs/${TAG}_perturb_intent.json ;;
    spatial)
      $EVAL --task-suite libero_spatial --tasks 0,1,2,3,4,5,6,7,8,9 --task-target-qpos 9 \
            --deltas 0.0,0.02,0.05,0.10,0.15,0.20 --num-trials-per-delta 10 \
            --out logs/${TAG}_spatial_intent.json ;;
    faildiag)
      $EVAL --tasks 1,6,8 --deltas 0.0,0.05,0.10 --num-trials-per-delta 20 \
            --out logs/${TAG}_faildiag.json ;;
    override)
      for M in flow zero random; do
        $EVAL --tasks 1,5,6 --deltas 0.0,0.05,0.10 --num-trials-per-delta 10 \
              --intent-override $M --out logs/${TAG}_override_${M}.json
      done ;;
    videos)
      mkdir -p logs/rollout_videos_${TAG}
      $EVAL --tasks 1,6,8 --deltas 0.0,0.05,0.10 --num-trials-per-delta 4 \
            --videos-per-cell 4 --video-dir logs/rollout_videos_${TAG} \
            --out logs/${TAG}_videos.json ;;
  esac
done
echo "DONE ${TAG}: ${EXPS}"
