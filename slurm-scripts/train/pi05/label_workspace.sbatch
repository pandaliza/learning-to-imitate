#!/bin/bash
#SBATCH --job-name=ws-label
#SBATCH --output=logs/ws_label_%j.out
#SBATCH --error=logs/ws_label_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Stage-0: VLM saliency labeling for workspace-model intent (current-frame).
# MolmoPoint-8B over LIBERO-Goal frames -> salient DINO patch indices, aligned to the
# vl_cache_goal_dinov2 grids. Runs in an ISOLATED overlay (molmo_pyenv: transformers==4.57.1)
# so the main env's openpi-pinned transformers 4.53 is untouched; tfstub satisfies the (spurious)
# TF import; expandable_segments avoids OOM on the 8B model.
set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
export HF_HUB_CACHE=$B/hf_cache
export PYTHONPATH="$B/molmo_pyenv:$B/tfstub:${PYTHONPATH:-}"   # transformers 4.57.1 overlay + TF stub
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MOLMO=${MOLMO:-allenai/MolmoPoint-8B}   # the paper's dedicated pointing model (Table 4)
OUT=${OUT:-$B/workspace_salient_goal}
DEMOS=${DEMOS:-8}   # demos/task subset (8B model is slow; ~8 demos x ~23 frames x 10 tasks is enough)
RATE=${RATE:-6}

python -m workspace_models.saliency.libero_adapter \
    --hdf5 /home/ldahiya/LIBERO/libero/datasets/libero_goal/*_demo.hdf5 \
    --vl-cache-dir $B/vl_cache_goal_dinov2 \
    --out-dir "$OUT" \
    --objects-json workspace_models/saliency/libero_goal_objects.json \
    --molmo "$MOLMO" \
    --sample-rate "$RATE" --max-patches 8 --max-demos-per-task "$DEMOS"
echo "[done] salient sets -> $OUT"
