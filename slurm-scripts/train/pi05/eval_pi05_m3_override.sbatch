#!/bin/bash
#SBATCH --job-name=pi05-m3-override
#SBATCH --output=logs/pi05_m3_override_%j.out
#SBATCH --error=logs/pi05_m3_override_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=5:00:00

# M3 intent override: does the VL-grounded frozen intent channel have authority?
# flow vs zero vs random on tasks 1,5,6. (M1-frozen showed flow==zero -> inert.)
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
  --frozen-base-weights $FROZEN --frozen-base-config pi05_base_nointent \
  --tasks 1,5,6 --deltas 0.0,0.05,0.10 --num-trials-per-delta 10"

python examples/openpi/eval_libero_perturbed.py $M3FLAGS --intent-override flow   --out logs/m3_override_flow.json
python examples/openpi/eval_libero_perturbed.py $M3FLAGS --intent-override zero   --out logs/m3_override_zero.json
python examples/openpi/eval_libero_perturbed.py $M3FLAGS --intent-override random --out logs/m3_override_random.json

echo "===================== M3 INTENT OVERRIDE ====================="
python - <<'PY'
import json, numpy as np
res={m:json.load(open(f"logs/m3_override_{m}.json"))["results"] for m in ("flow","zero","random")}
deltas=json.load(open("logs/m3_override_flow.json"))["deltas"]
print(f"{'delta':>8}  {'flow':>8}  {'zero':>8}  {'random':>8}  {'flow-zero':>10}")
for d in deltas:
    row={m:float(np.mean([v['sr'] for k,v in res[m][str(d)].items() if not k.startswith('_')])) for m in res}
    print(f"{d:8.2f}  {row['flow']:8.3f}  {row['zero']:8.3f}  {row['random']:8.3f}  {row['flow']-row['zero']:+10.3f}")
PY
