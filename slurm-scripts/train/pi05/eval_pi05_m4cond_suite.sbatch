#!/bin/bash
#SBATCH --job-name=pi05-m4cond-suite
#SBATCH --output=logs/pi05_m4cond_suite_%j.out
#SBATCH --error=logs/pi05_m4cond_suite_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00

# FIXED-M4 (cond) OOD suite. PyTorch policy (model.safetensors) + aux MLP generator
# (z_hat = aux_head(pi05 vl_mean)). One EXPS subset per job for parallelism.
# Usage: EXPS="perturb" CK=.../pi05_m4_cond/6000 sbatch eval_pi05_m4cond_suite.sbatch
#   EXPS choices: perturb | spatial | faildiag | peraxis

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu                 # policy is PyTorch; keep JAX off the GPU
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

STEP=${STEP:-6000}
EXPS=${EXPS:-"perturb spatial faildiag peraxis"}
CK=${CK:-/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_m4_cond/${STEP}}
TAG=m4cond_${STEP}
FLAGS="--config-name pi05_base_intent --checkpoint-dir $CK --norm-stats-from-config --fp32 \
  --vl-cotrain --intent-task-config libero_goal_suite_image_slot_intent_vl --aux-head $CK/aux_head.pt"
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
    peraxis)
      for AX in x y; do
        $EVAL --axis $AX --deltas 0.0,0.02,0.03,0.05,0.10 --num-trials-per-delta 10 \
              --out logs/${TAG}_peraxis_${AX}.json
      done ;;
  esac
done
echo "DONE ${TAG}: ${EXPS}"
