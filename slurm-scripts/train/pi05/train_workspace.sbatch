#!/bin/bash
#SBATCH --job-name=ws-stage1
#SBATCH --output=logs/ws_stage1_%j.out
#SBATCH --error=logs/ws_stage1_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Stage-1: train the workspace-model intent encoder on DINOv2 grids + MolmoPoint-8B salient sets.
# Main env (torch + workspace_models only; no transformers/Molmo overlay needed here).
set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
OUT=${OUT:-$B/ldahiya_checkpoints/workspace_stack_goal}

python examples/openpi/train_workspace.py \
    --vl-cache-dir $B/vl_cache_goal_dinov2 \
    --salient-dir $B/workspace_salient_goal \
    --out "$OUT" \
    --num-patches 256 --feat-dim 768 --max-patches 8 \
    --steps 5000 --batch-size 64 --lr 1e-4 --save-every 1000
echo "[done] workspace stack -> $OUT"
