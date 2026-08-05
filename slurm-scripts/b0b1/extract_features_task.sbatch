#!/bin/bash
#SBATCH --job-name=b0b1-feat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/b0b1_feat_%j.out
#SBATCH --error=logs/b0b1_feat_%j.err
#SBATCH --export=NONE

# Task-sharded feature extraction: one job per RoboCasa task.
# Usage: TASK=TurnOnElectricKettle VARIANT=pooled sbatch --export=NONE,TASK=...,VARIANT=pooled \
#            slurm-scripts/b0b1/extract_features_task.sbatch
set -euo pipefail

ROOT_DIR="/data/group_data/maxlab/common_datasets/amagnuso/robocasa/v1.0/target"
CACHE_ROOT="/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features"
export HF_HOME="/data/user_data/ldahiya/hf_cache"

VARIANT="${VARIANT:-pooled}"
TASK="${TASK:?set TASK (RoboCasa task name)}"

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate
mkdir -p "$CACHE_ROOT/$VARIANT" logs

echo "=== B0/B1 feature extraction: task=$TASK variant=$VARIANT job=$SLURM_JOB_ID ==="

python - "$VARIANT" "$ROOT_DIR" "$CACHE_ROOT" "$TASK" << 'EOF_PYTHON'
import sys, os, time
from pathlib import Path

variant, root_dir, cache_root, task = sys.argv[1:5]
sys.path.insert(0, "/home/ldahiya/max_vla/much-ado-about-noising")
from steer_intent.vl_feature_cache import extract_all_features

t0 = time.time()
extract_all_features(
    root_dir=root_dir,
    output_dir=Path(cache_root) / variant,
    variant=variant,
    task_names=[task],
    device="cuda",
    hf_home=os.environ["HF_HOME"],
    skip_existing=True,
)
print(f"[done] {task} ({variant}) in {(time.time()-t0)/3600:.2f} h")
EOF_PYTHON
