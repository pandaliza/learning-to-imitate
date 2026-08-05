#!/bin/bash
#SBATCH --job-name=b0b1_features_shard
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/b0b1_features_shard_%j.log
#SBATCH --export=NONE

set -e

# ===== Configuration =====
ROOT_DIR="/data/group_data/maxlab/common_datasets/amagnuso/robocasa/v1.0/target"
CACHE_ROOT="/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features"
HF_HOME="/data/user_data/ldahiya/hf_cache"
REPO_ROOT="/home/ldahiya/max_vla/much-ado-about-noising"

# Shard test: TurnOnElectricKettle only, both variants
TASK="TurnOnElectricKettle"

echo "=========================================="
echo "B0/B1 Feature Cache Extraction — Shard Test"
echo "=========================================="
echo "Task: $TASK"
echo "Root dir: $ROOT_DIR"
echo "Cache root: $CACHE_ROOT"
echo "HF home: $HF_HOME"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Verify dataset exists
if [ ! -d "$ROOT_DIR" ]; then
    echo "ERROR: Dataset not found at $ROOT_DIR"
    exit 1
fi

# Create cache directory
mkdir -p "$CACHE_ROOT"

# Activate environment
cd "$REPO_ROOT"
source ~/.bashrc
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# ===== Python extraction script =====
python - "$TASK" "$ROOT_DIR" "$CACHE_ROOT" "$HF_HOME" << 'EOF_PYTHON'
import sys
import os
from pathlib import Path
import time
import traceback

task = sys.argv[1]
root_dir = sys.argv[2]
cache_root = sys.argv[3]
hf_home = sys.argv[4]

# Set HF_HOME
os.environ["HF_HOME"] = hf_home

# Import after env setup
sys.path.insert(0, "/home/ldahiya/max_vla/much-ado-about-noising")
from steer_intent.vl_feature_cache import extract_all_features

print(f"Starting feature extraction for task: {task}")
print(f"HF_HOME={os.environ.get('HF_HOME')}")
print("")

# Test both variants
for variant in ["pooled", "grid16"]:
    try:
        print(f"\n{'='*60}")
        print(f"Variant: {variant}")
        print(f"{'='*60}")

        output_dir = Path(cache_root) / variant
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # Extract features
        extract_all_features(
            root_dir=root_dir,
            output_dir=output_dir,
            variant=variant,
            task_names=[task],
            device="cuda",
            hf_home=hf_home,
            skip_existing=True,
        )

        elapsed = time.time() - start_time
        print(f"\nVariant {variant} complete in {elapsed:.1f}s")

        # Compute projected full-dataset time and cache sizes
        # TurnOnElectricKettle has ~500 episodes, ~1M frames
        # Full dataset: 9 tasks, ~4543 episodes, ~1.87M frames
        # Rough estimate: 500 episodes = 1M frames, so full = 3.74x scale
        print(f"\n[Projection for full dataset (1.87M frames, 9 tasks)]")
        full_time_hours = elapsed * 3.74 / 3600
        print(f"  Estimated time: {full_time_hours:.1f} hours on 1 GPU")

        # Estimate cache size
        output_path = output_dir / "features" / task
        if output_path.exists():
            import subprocess
            size_mb = int(subprocess.check_output(["du", "-sm", str(output_path)]).split()[0])
            full_cache_gb = size_mb * 3.74 / 1024
            print(f"  {task} cache size: {size_mb} MB")
            print(f"  Estimated full cache size: {full_cache_gb:.1f} GB")

    except Exception as e:
        print(f"\nERROR in variant {variant}:")
        traceback.print_exc()
        sys.exit(1)

print("\n" + "="*60)
print("Shard test complete!")
print("="*60)
EOF_PYTHON

echo "Extraction complete!"
