#!/bin/bash
#SBATCH --job-name=pi05-m3-faildiag
#SBATCH --output=logs/pi05_m3_faildiag_%j.out
#SBATCH --error=logs/pi05_m3_faildiag_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00

# M3 failure cascade (reach -> grasp -> place), tasks 1/6/8, 20 trials.
set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JAX_COMPILATION_CACHE_DIR=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/jax_cache_eval
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
CK=${CK:-$B/ldahiya_checkpoints/pi05_base_intent_m3/base_intent_m3_goal_run1/29999}
VLSTACK=${VLSTACK:-$B/ldahiya_checkpoints/slot_intent_vl/intent_stack_32000.pt}
FROZEN=${FROZEN:-$B/pi05_base_pytorch}
M3FLAGS="--config-name pi05_base_intent_m3 --checkpoint-dir $CK --vl-cotrain \
  --intent-task-config libero_goal_suite_image_slot_intent_vl --intent-stack $VLSTACK \
  --frozen-base-weights $FROZEN --frozen-base-config pi05_base_nointent"

python examples/openpi/eval_libero_perturbed.py $M3FLAGS \
    --tasks 1,6,8 --deltas 0.0,0.05,0.10 --num-trials-per-delta 20 \
    --out logs/m3_faildiag.json

echo "===================== M3 FAILURE CASCADE (valid trials) ====================="
python - <<'PY'
import json
from collections import Counter
d = json.load(open("logs/m3_faildiag.json"))
print(f"{'δ':>5} {'N':>4} {'reach':>7} {'grasp':>7} {'place':>7}   breakdown")
for dl in d["deltas"]:
    agg=Counter()
    for tid,v in d["results"][str(dl)].items():
        if tid.startswith("_"): continue
        agg.update(v.get("fail_breakdown",{}))
    N=sum(agg.values()) or 1
    reach=N-agg["no_reach"]; grasp=reach-agg["no_grasp"]; place=agg["success"]
    b="  ".join(f"{k}={agg.get(k,0)}" for k in ("success","no_reach","no_grasp","no_place"))
    print(f"{dl:5.2f} {N:4d} {reach/N:7.2f} {grasp/N:7.2f} {place/N:7.2f}   {b}")
PY
