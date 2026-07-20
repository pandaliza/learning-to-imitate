#!/bin/bash
#SBATCH --job-name=pi05-m3-spatial
#SBATCH --output=logs/pi05_m3_spatial_%j.out
#SBATCH --error=logs/pi05_m3_spatial_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

# M3 zero-shot transfer to libero_spatial (10 tasks, bowl at qpos[9]).
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
    --task-suite libero_spatial --tasks 0,1,2,3,4,5,6,7,8,9 --task-target-qpos 9 \
    --deltas 0.0,0.02,0.05,0.10,0.15,0.20 --num-trials-per-delta 10 \
    --out logs/m3_spatial_intent.json

echo "===================== SPATIAL: control vs M1-frozen vs M3 ====================="
python - <<'PY'
import json, pathlib
def L(p): return json.load(open(p)) if pathlib.Path(p).exists() else None
ni,fr,m3=L("logs/base_spatial_nointent.json"),L("logs/base_spatial_intent.json"),L("logs/m3_spatial_intent.json")
def sr(d,k): return d["results"][str(k)].get("_mean_sr",float("nan")) if d else float("nan")
print(f"{'δ':>6} {'ctrl':>7} {'M1froz':>7} {'M3':>7}")
for d in m3["deltas"]:
    print(f"{d:6.2f} {sr(ni,d):7.3f} {sr(fr,d):7.3f} {sr(m3,d):7.3f}")
PY
