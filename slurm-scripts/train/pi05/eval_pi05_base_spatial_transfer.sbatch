#!/bin/bash
#SBATCH --job-name=pi05-base-spatial
#SBATCH --output=logs/pi05_base_spatial_%j.out
#SBATCH --error=logs/pi05_base_spatial_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00

# Zero-shot transfer: pi05_base arms (trained on libero_goal) -> libero_spatial.
# All 10 libero_spatial tasks are "pick up the black bowl ... place on the plate"
# with the bowl at qpos[9] (same address as libero_goal), so --task-target-qpos 9
# applies uniformly. Pass --tasks 0..9 so tasks 0 and 7 (skipped by the libero_goal
# TASK_TARGET map) are still evaluated under the override.

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export JAX_COMPILATION_CACHE_DIR=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/jax_cache_eval
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

CKPT_BASE=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints
NOINTENT_CKPT=$CKPT_BASE/pi05_base_nointent/base_nointent_goal_run1/29999
INTENT_CKPT=$CKPT_BASE/pi05_base_intent/base_intent_goal_run1/29999
FLOW_MAP=/data/user_data/ldahiya/libero-train/libero-goal-suite-image/slot_intent/models/model_best.pt

DELTAS="0.0,0.02,0.05,0.10,0.15,0.20"
TRIALS=10

COMMON="--task-suite libero_spatial \
        --tasks 0,1,2,3,4,5,6,7,8,9 \
        --task-target-qpos 9 \
        --deltas $DELTAS \
        --num-trials-per-delta $TRIALS"

# 1. No-intent control.
python examples/openpi/eval_libero_perturbed.py $COMMON \
    --config-name pi05_base_nointent \
    --checkpoint-dir "$NOINTENT_CKPT" \
    --out logs/base_spatial_nointent.json

# 2. Frozen-intent.
python examples/openpi/eval_libero_perturbed.py $COMMON \
    --config-name pi05_base_intent \
    --checkpoint-dir "$INTENT_CKPT" \
    --intent \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt "$FLOW_MAP" \
    --out logs/base_spatial_intent.json

echo "===================== pi05_base LIBERO-SPATIAL TRANSFER ====================="
python - <<'PY'
import json
ni = json.load(open("logs/base_spatial_nointent.json"))
it = json.load(open("logs/base_spatial_intent.json"))
deltas = ni["deltas"]
print(f"{'delta':>6}  {'nointent':>9}  {'intent':>8}  {'Δ_SR':>8}  "
      f"{'ni|valid':>9}  {'it|valid':>9}  {'Δ|valid':>8}  {'valid%':>7}")
print("-" * 80)
for d in deltas:
    r_ni, r_it = ni["results"][str(d)], it["results"][str(d)]
    sr_ni  = r_ni.get("_mean_sr", float("nan"));        sr_it  = r_it.get("_mean_sr", float("nan"))
    srv_ni = r_ni.get("_mean_sr_valid", float("nan"));  srv_it = r_it.get("_mean_sr_valid", float("nan"))
    vf     = r_ni.get("_mean_valid_frac", float("nan"))
    print(f"{d:6.2f}  {sr_ni:9.3f}  {sr_it:8.3f}  {sr_it - sr_ni:+8.3f}  "
          f"{srv_ni:9.3f}  {srv_it:9.3f}  {srv_it - srv_ni:+8.3f}  {100*vf:6.1f}%")

print("\n--- per-task SR at δ=0.00 ---")
for tid, v in sorted(it["results"]["0.0"].items(), key=lambda kv: (kv[0].startswith("_"), kv[0])):
    if tid.startswith("_"):
        continue
    ni_sr = ni["results"]["0.0"].get(tid, {}).get("sr", float("nan"))
    print(f"  task {tid}: nointent={ni_sr:.2f}  intent={v['sr']:.2f}  diff={v['sr']-ni_sr:+.2f}  :: {v['task']}")
PY
