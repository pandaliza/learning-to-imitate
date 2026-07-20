#!/bin/bash
#SBATCH --job-name=pi05-base-eval
#SBATCH --output=logs/pi05_base_eval_%j.out
#SBATCH --error=logs/pi05_base_eval_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --exclude=babel-n5-32,babel-n9-32

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

B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints
INTENT=$B/pi05_base_intent/base_intent_goal_run1/29999
CONTROL=$B/pi05_base_nointent/base_nointent_goal_run1/29999
FLOW_MAP=/data/user_data/ldahiya/libero-train/libero-goal-suite-image/slot_intent/models/model_best.pt
TRIALS=20

# A/B from the GENERALIST pi05_base init (LoRA-VL + full action head, 30k). Our finetune
# convention -> no --rotate-images. JAX checkpoints carry their own assets/norm_stats.

# Intent arm: 64D slot intent injected into adaRMS (intent from the flow-map sidecar).
python examples/openpi/eval_libero_intent.py \
    --config-name pi05_base_intent \
    --checkpoint-dir "$INTENT" \
    --task-suite libero_goal \
    --num-trials-per-task $TRIALS \
    --intent \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt "$FLOW_MAP" \
    --out logs/eval_base_intent_30k.json

# Control arm: identical finetune, intent_dim=0.
python examples/openpi/eval_libero_intent.py \
    --config-name pi05_base_nointent \
    --checkpoint-dir "$CONTROL" \
    --task-suite libero_goal \
    --num-trials-per-task $TRIALS \
    --out logs/eval_base_control_30k.json

echo "================= pi05_base A/B SR (30k) ================="
python - <<'PY'
import json
i = json.load(open("logs/eval_base_intent_30k.json"))
c = json.load(open("logs/eval_base_control_30k.json"))
print(f"  intent   pi05_base_intent   : mean SR = {i['mean_sr']:.3f}  ({i['total']})")
print(f"  control  pi05_base_nointent : mean SR = {c['mean_sr']:.3f}  ({c['total']})")
print(f"  delta (intent - control)    : {i['mean_sr']-c['mean_sr']:+.3f}")
PY
