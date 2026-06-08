#!/bin/bash
#SBATCH --job-name=pi05-perturb-intent
#SBATCH --output=logs/pi05_perturb_intent_%j.out
#SBATCH --error=logs/pi05_perturb_intent_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00

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

INTENT_RUN=/data/user_data/ldahiya/openpi-checkpoints/pi05_libero_intent/slot_intent_goal_run1
FLOW_MAP=/data/user_data/ldahiya/libero-train/libero-goal-suite-image/slot_intent/models/model_best.pt

DELTAS="0.0,0.02,0.05,0.10,0.15,0.20"
TRIALS=10

INTENT_STEP=33000
echo "Using intent checkpoint: $INTENT_RUN/$INTENT_STEP"

python examples/openpi/eval_libero_perturbed.py \
    --config-name pi05_libero_intent \
    --checkpoint-dir "$INTENT_RUN/$INTENT_STEP" \
    --intent \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt "$FLOW_MAP" \
    --deltas "$DELTAS" \
    --num-trials-per-delta $TRIALS \
    --out logs/perturb_intent.json

# Print comparison against the already-saved no-intent results.
if [ -f logs/perturb_nointent.json ]; then
  echo "===================== SR-vs-δ COMPARISON ====================="
  python - <<'PY'
import json

ni = json.load(open("logs/perturb_nointent.json"))
it = json.load(open("logs/perturb_intent.json"))

deltas = it["deltas"]
print(f"{'delta':>8}  {'nointent':>10}  {'intent':>10}  {'delta_SR':>10}")
print("-" * 46)
for d in deltas:
    sr_ni = ni["results"][str(d)].get("_mean_sr", float("nan"))
    sr_it = it["results"][str(d)].get("_mean_sr", float("nan"))
    print(f"{d:8.2f}  {sr_ni:10.3f}  {sr_it:10.3f}  {sr_it - sr_ni:+10.3f}")
PY
else
  echo "(logs/perturb_nointent.json not found — run eval_pi05_libero_perturbed.sbatch first)"
fi
