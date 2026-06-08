#!/bin/bash
#SBATCH --job-name=pi05-spatial-transfer
#SBATCH --output=logs/pi05_spatial_transfer_%j.out
#SBATCH --error=logs/pi05_spatial_transfer_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00

# Transfer experiment: both models were fine-tuned on libero_goal; run the same
# radial-perturbation sweep on libero_spatial (10 tasks, all unseen at train time).
# libero_spatial: "pick up the black bowl [from X] and place it on the plate" —
# same pick+place action, same bowl object (qpos 9), different initial bowl locations.
# Tests whether intent conditioning helps generalise spatially on truly novel tasks.

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

NOINTENT_CKPT=/data/user_data/ldahiya/openpi-checkpoints/pi05_libero_nointent/nointent_goal_run1/35000
INTENT_CKPT=/data/user_data/ldahiya/openpi-checkpoints/pi05_libero_intent/slot_intent_goal_run1/35000
FLOW_MAP=/data/user_data/ldahiya/libero-train/libero-goal-suite-image/slot_intent/models/model_best.pt

DELTAS="0.0,0.02,0.05,0.10,0.15,0.20"
TRIALS=10

# Bowl is always qpos[9] in libero_spatial (same address as libero_goal).
# All 10 tasks target the same bowl object so --task-target-qpos 9 applies to all.
COMMON="--task-suite libero_spatial \
        --task-target-qpos 9 \
        --deltas $DELTAS \
        --num-trials-per-delta $TRIALS"

# 1. No-intent control.
python examples/openpi/eval_libero_perturbed.py $COMMON \
    --config-name pi05_libero_nointent \
    --checkpoint-dir $NOINTENT_CKPT \
    --out logs/spatial_transfer_nointent.json

# 2. Intent-conditioned.
python examples/openpi/eval_libero_perturbed.py $COMMON \
    --config-name pi05_libero_intent \
    --checkpoint-dir $INTENT_CKPT \
    --intent \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt "$FLOW_MAP" \
    --out logs/spatial_transfer_intent.json

echo "===================== LIBERO-SPATIAL TRANSFER SR-vs-δ ====================="
python - <<'PY'
import json, numpy as np

ni = json.load(open("logs/spatial_transfer_nointent.json"))
it = json.load(open("logs/spatial_transfer_intent.json"))
deltas = ni["deltas"]

print(f"{'delta':>8}  {'nointent':>10}  {'intent':>10}  {'delta_SR':>10}")
print("-" * 46)
for d in deltas:
    sr_ni = ni["results"][str(d)].get("_mean_sr", float("nan"))
    sr_it = it["results"][str(d)].get("_mean_sr", float("nan"))
    print(f"{d:8.2f}  {sr_ni:10.3f}  {sr_it:10.3f}  {sr_it - sr_ni:+10.3f}")

# Also print per-task breakdown at δ=0 to see which tasks transfer
print("\n--- per-task SR at δ=0.00 ---")
for tid, v in sorted(it["results"]["0.0"].items()):
    if tid.startswith("_"):
        continue
    ni_sr = ni["results"]["0.0"].get(tid, {}).get("sr", float("nan"))
    print(f"  task {tid}: nointent={ni_sr:.2f}  intent={v['sr']:.2f}  diff={v['sr']-ni_sr:+.2f}  :: {v['task']}")
PY
