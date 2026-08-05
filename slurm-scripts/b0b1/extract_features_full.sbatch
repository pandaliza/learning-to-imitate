#!/bin/bash
#SBATCH --job-name=b0b1_features_full
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/b0b1_features_full_%j.log
#SBATCH --export=NONE

set -e

# ===== Configuration =====
ROOT_DIR="/data/group_data/maxlab/common_datasets/amagnuso/robocasa/v1.0/target"
CACHE_ROOT="/data/group_data/maxlab/common_datasets/pandaliza/b0b1_features"
HF_HOME="/data/user_data/ldahiya/hf_cache"
REPO_ROOT="/home/ldahiya/max_vla/much-ado-about-noising"

# Parameterize variant
VARIANT="${1:-pooled}"  # pooled or grid16

if [ "$VARIANT" != "pooled" ] && [ "$VARIANT" != "grid16" ]; then
    echo "ERROR: VARIANT must be 'pooled' or 'grid16'"
    exit 1
fi

echo "=========================================="
echo "B0/B1 Feature Cache Extraction — Full Dataset"
echo "=========================================="
echo "Variant: $VARIANT"
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
mkdir -p "$CACHE_ROOT/$VARIANT"

# Activate environment
cd "$REPO_ROOT"
source ~/.bashrc
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# ===== Python extraction script =====
python - "$VARIANT" "$ROOT_DIR" "$CACHE_ROOT" "$HF_HOME" << 'EOF_PYTHON'
import sys
import os
from pathlib import Path
import time
import traceback

variant = sys.argv[1]
root_dir = sys.argv[2]
cache_root = sys.argv[3]
hf_home = sys.argv[4]

# Set HF_HOME
os.environ["HF_HOME"] = hf_home

# Import after env setup
sys.path.insert(0, "/home/ldahiya/max_vla/much-ado-about-noising")
from steer_intent.vl_feature_cache import extract_all_features

print(f"Starting feature extraction for variant: {variant}")
print(f"HF_HOME={os.environ.get('HF_HOME')}")
print("")

try:
    output_dir = Path(cache_root) / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Extract features for all 9 tasks
    # 9 tasks: TurnOnElectricKettle, PickPlaceCounterToCabinet, PickPlaceCounterToStove,
    #          SlideDishwasherRack, KettleBoiling, LoadDishwasher, PrepareCoffee,
    #          PreSoakPan, WashLettuce
    extract_all_features(
        root_dir=root_dir,
        output_dir=output_dir,
        variant=variant,
        task_names=None,  # Use all 9 tasks
        device="cuda",
        hf_home=hf_home,
        skip_existing=True,
    )

    elapsed = time.time() - start_time
    print(f"\nVariant {variant} complete in {elapsed/3600:.1f} hours")

    # Print cache statistics
    import subprocess

    cache_dir = output_dir / "features"
    if cache_dir.exists():
        try:
            size_mb = int(subprocess.check_output(["du", "-sm", str(cache_dir)]).split()[0])
            size_gb = size_mb / 1024
            print(f"Total cache size: {size_gb:.1f} GB")
        except Exception as e:
            print(f"Could not compute cache size: {e}")

except Exception as e:
    print(f"\nERROR in extraction:")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("Full extraction complete!")
print("="*60)
EOF_PYTHON

echo "Done!"
