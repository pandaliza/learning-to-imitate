#!/bin/bash
#SBATCH --job-name=debug-gt-replay
#SBATCH --output=logs/debug_gt_replay_%j.out
#SBATCH --error=logs/debug_gt_replay_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising

source .venv/bin/activate

export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# GT-action replay diagnostic: test if raw demo actions produce success in RoboCasa env
# Usage:
#   sbatch --export=NONE debug_gt_action_replay.sbatch
#   -- or with override --
#   TASK=PickPlaceCounterToCabinet NUM_EPS=5 \
#     sbatch --export=NONE slurm-scripts/debug/debug_gt_action_replay.sbatch

TASK=${TASK:-TurnOnElectricKettle}
NUM_EPS=${NUM_EPS:-3}
DATA_ROOT=/data/group_data/maxlab/common_datasets/amagnuso/robocasa/v1.0/target
OUT=logs/debug_gt_replay_${TASK}.json

echo "=== GT-Action Replay Test ===" >&2
echo "Task: $TASK" >&2
echo "Episodes: $NUM_EPS" >&2
echo "Data root: $DATA_ROOT" >&2
echo "Output: $OUT" >&2

python examples/openpi/debug_gt_action_replay.py \
    --task "$TASK" \
    --data-root "$DATA_ROOT" \
    --num-episodes "$NUM_EPS" \
    --out "$OUT"

echo "=== GT-Action Replay Complete ===" >&2
cat "$OUT"
