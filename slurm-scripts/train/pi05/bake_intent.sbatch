#!/bin/bash
#SBATCH --job-name=bake-intent
#SBATCH --output=logs/bake_intent_%j.out
#SBATCH --error=logs/bake_intent_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export JAX_PLATFORMS=cpu
export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/nb_${SLURM_JOB_ID:-m}_XXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"' EXIT

# Bake the alt-encoder slot intent into a LeRobot dataset (Stage-2 part 1).
# VARIANT=dino|dynaflip ; STEP=stage-1 checkpoint step (default 32000, matches the VL arm).
VARIANT=${VARIANT:?set VARIANT=dino or dynaflip}
STEP=${STEP:-32000}
CACHE=$([ "$VARIANT" = dino ] && echo vl_cache_goal_dinov2 || echo vl_cache_goal_dynaflip)
B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
# Overridable for the DINOSAUR (feature-recon) stacks: CFG/STACK/REPO.
CFG=${CFG:-libero_goal_suite_image_slot_intent_${VARIANT}}
STACK=${STACK:-${B}/ldahiya_checkpoints/slot_intent_${VARIANT}/intent_stack_${STEP}.pt}
REPO=${REPO:-ldahiya/libero_goal_slot_intent_${VARIANT}}

python examples/openpi/convert_libero_to_lerobot_intent.py \
    --task-config "$CFG" \
    --m3-vl-stack "$STACK" \
    --vl-cache-dir ${B}/${CACHE} \
    --repo-id "$REPO"
