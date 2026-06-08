#!/bin/bash
#SBATCH --job-name=pi05-intent-override
#SBATCH --output=logs/pi05_intent_override_%j.out
#SBATCH --error=logs/pi05_intent_override_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Sanity-check experiment: does the intent model actually use the intent signal?
# Runs the same intent-conditioned policy under three intent modes:
#   flow   — normal flow-map sidecar (real intent)
#   zero   — all-zeros vector injected (no signal)
#   random — random Gaussian vector injected (noise)
# If zero ≈ flow SR, the policy ignores intent.
# If random < zero < flow, the policy reads it correctly.
# Runs on 3 representative tasks to keep wall-clock under ~3h.

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

INTENT_CKPT=/data/user_data/ldahiya/openpi-checkpoints/pi05_libero_intent/slot_intent_goal_run1/25000
FLOW_MAP=/data/user_data/ldahiya/libero-train/libero-goal-suite-image/slot_intent/models/model_best.pt

# Tasks 1, 5, 6: diverse (pick+place, push, pick+place with small target).
# δ=0.0 and δ=0.10 only — the two most informative points from the main experiment.
TASKS="1,5,6"
DELTAS="0.0,0.05,0.10"
TRIALS=10

COMMON="--config-name pi05_libero_intent \
        --checkpoint-dir $INTENT_CKPT \
        --intent \
        --intent-task-config libero_goal_suite_image_slot_intent \
        --intent-ckpt $FLOW_MAP \
        --tasks $TASKS \
        --deltas $DELTAS \
        --num-trials-per-delta $TRIALS"

python examples/openpi/eval_libero_perturbed.py $COMMON \
    --intent-override flow   --out logs/override_flow.json

python examples/openpi/eval_libero_perturbed.py $COMMON \
    --intent-override zero   --out logs/override_zero.json

python examples/openpi/eval_libero_perturbed.py $COMMON \
    --intent-override random --out logs/override_random.json

echo "===================== INTENT OVERRIDE COMPARISON ====================="
python - <<'PY'
import json, numpy as np

results = {}
for mode in ("flow", "zero", "random"):
    d = json.load(open(f"logs/override_{mode}.json"))
    results[mode] = d["results"]

deltas = json.load(open("logs/override_flow.json"))["deltas"]
print(f"{'delta':>8}  {'flow':>8}  {'zero':>8}  {'random':>8}  {'flow-zero':>10}  {'flow-random':>12}")
print("-" * 68)
for d in deltas:
    row = {}
    for mode in ("flow", "zero", "random"):
        srs = [v["sr"] for k, v in results[mode][str(d)].items() if not k.startswith("_")]
        row[mode] = float(np.mean(srs)) if srs else float("nan")
    print(f"{d:8.2f}  {row['flow']:8.3f}  {row['zero']:8.3f}  {row['random']:8.3f}"
          f"  {row['flow']-row['zero']:+10.3f}  {row['flow']-row['random']:+12.3f}")
PY
