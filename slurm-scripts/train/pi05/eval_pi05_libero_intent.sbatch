#!/bin/bash
#SBATCH --job-name=pi05-libero-eval
#SBATCH --output=logs/pi05_libero_eval_%j.out
#SBATCH --error=logs/pi05_libero_eval_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00

set -euo pipefail

mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export JAX_COMPILATION_CACHE_DIR=/data/user_data/ldahiya/jax_cache
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

PI05_LIBERO=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/pi05_libero
INTENT_RUN=/data/user_data/ldahiya/openpi-checkpoints/pi05_libero_intent/slot_intent_goal_run1
FLOW_MAP=/data/user_data/ldahiya/libero-train/libero-goal-suite-image/slot_intent/models/model_best.pt
TRIALS=20

# Intent run is COMPLETE -> use the true latest checkpoint (final 29999, persists).
# (During training, prefer a multiple of keep_period=5000 to avoid the max_to_keep race.)
STEP=$(ls "$INTENT_RUN" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
echo "Using intent checkpoint step: ${STEP}"

# 1. Baseline: off-the-shelf pi05_libero (PI convention -> rotate images, no intent).
#    NOTE: vs the intent model this conflates 'finetuned on our data' + 'intent'.
#    Skip if already computed (baseline is fixed; no need to re-run).
if [ ! -f logs/eval_baseline_goal.json ]; then
python examples/openpi/eval_libero_intent.py \
    --config-name pi05_libero \
    --checkpoint-dir "$PI05_LIBERO" \
    --task-suite libero_goal \
    --num-trials-per-task $TRIALS \
    --rotate-images \
    --out logs/eval_baseline_goal.json
else
    echo "Reusing cached baseline: logs/eval_baseline_goal.json"
fi

# 2. Intent: our action-head finetune + 64D slot intent (our convention -> no rotate).
python examples/openpi/eval_libero_intent.py \
    --config-name pi05_libero_intent \
    --checkpoint-dir "$INTENT_RUN/$STEP" \
    --task-suite libero_goal \
    --num-trials-per-task $TRIALS \
    --intent \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt "$FLOW_MAP" \
    --out logs/eval_intent_goal.json

echo "===================== SR COMPARISON ====================="
python - <<'PY'
import json
b = json.load(open("logs/eval_baseline_goal.json"))
i = json.load(open("logs/eval_intent_goal.json"))
print(f"  baseline  pi05_libero        : mean SR = {b['mean_sr']:.3f}  ({b['total']})")
print(f"  intent    pi05_libero_intent : mean SR = {i['mean_sr']:.3f}  ({i['total']})")
print(f"  delta (intent - baseline)    : {i['mean_sr']-b['mean_sr']:+.3f}")
PY
