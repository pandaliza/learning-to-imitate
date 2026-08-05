#!/bin/bash
#SBATCH --job-name=b0b1-b1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/b0b1_b1_%j.log

# IMPORTANT: Submit with --export=NONE to avoid inherited CUDA_VISIBLE_DEVICES
# sbatch --export=NONE slurm-scripts/b0b1/train_b1.sbatch
#
# Feature cache must exist before training:
# /data/group_data/maxlab/common_datasets/pandaliza/b0b1_features/<VARIANT>/
#
# If not pre-extracted, run feature extraction first:
# python steer_intent/vl_feature_cache.py --variant pooled

set -e

# Configuration
ARM="b1"
VARIANT="${VARIANT:-pooled}"
DATA_ROOT="${DATA_ROOT:-/data/group_data/maxlab/common_datasets/robocasa-test/target}"
FEATURE_CACHE="/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features/${VARIANT}"
NORM_STATS="/home/ldahiya/max_vla/much-ado-about-noising/assets/pi05_robocasa_copred/robocasa/norm_stats.json"
CKPT_DIR="/data/group_data/maxlab/common_datasets/pandaliza/b0b1_checkpoints"

# Model config
DEPTH=6
DIM=512
NUM_HEADS=8
VISION_DIM=768
NUM_VISION_TOKENS=1
LANG_DIM=768

# Training config
BATCH_SIZE=256
LR=1e-4
STEPS=200000
CKPT_INTERVAL=25000
LOG_INTERVAL=100
SEED=0

# Check feature cache exists
if [ ! -d "$FEATURE_CACHE" ]; then
    echo "[ERROR] Feature cache not found: $FEATURE_CACHE"
    echo "Run feature extraction first: python steer_intent/vl_feature_cache.py --variant $VARIANT"
    exit 1
fi

# Setup environment
cd /home/ldahiya/max_vla/much-ado-about-noising
export PYTHONPATH="/home/ldahiya/max_vla/much-ado-about-noising:$PYTHONPATH"

# Log environment
echo "[B1] Training starts"
echo "  ARM: $ARM"
echo "  VARIANT: $VARIANT"
echo "  DATA_ROOT: $DATA_ROOT"
echo "  FEATURE_CACHE: $FEATURE_CACHE"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  STEPS: $STEPS"
echo "  DEPTH: $DEPTH"
echo "  DIM: $DIM"
echo "  SEED: $SEED"

# Train
python examples/b0b1/train_b0b1.py \
    --arm "$ARM" \
    --data-root "$DATA_ROOT" \
    --feature-cache "$FEATURE_CACHE" \
    --norm-stats "$NORM_STATS" \
    --depth "$DEPTH" \
    --dim "$DIM" \
    --num-heads "$NUM_HEADS" \
    --vision-dim "$VISION_DIM" \
    --num-vision-tokens "$NUM_VISION_TOKENS" \
    --lang-dim "$LANG_DIM" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --steps "$STEPS" \
    --ckpt-dir "$CKPT_DIR" \
    --ckpt-interval "$CKPT_INTERVAL" \
    --log-interval "$LOG_INTERVAL" \
    --device cuda \
    --seed "$SEED"

echo "[B1] Training complete"
