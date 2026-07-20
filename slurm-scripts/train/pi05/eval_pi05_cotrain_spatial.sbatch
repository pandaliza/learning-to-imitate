#!/bin/bash
#SBATCH --job-name=pi05-cotrain-spatial
#SBATCH --output=logs/pi05_cotrain_spatial_%j.out
#SBATCH --error=logs/pi05_cotrain_spatial_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00

# Co-train zero-shot transfer to libero_spatial (all 10 tasks, bowl at qpos[9]).
# Compared against saved control + frozen spatial-transfer results.

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

CK=${CK:-/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_vl_cotrain_v2/24000}

python examples/openpi/eval_libero_perturbed.py \
    --config-name pi05_base_intent --checkpoint-dir "$CK" \
    --norm-stats-from-config --fp32 \
    --vl-cotrain --intent-stack "$CK/intent_stack.pt" \
    --task-suite libero_spatial --tasks 0,1,2,3,4,5,6,7,8,9 --task-target-qpos 9 \
    --deltas 0.0,0.02,0.05,0.10,0.15,0.20 --num-trials-per-delta 10 \
    --out logs/cotrain_spatial_intent.json

echo "===================== SPATIAL TRANSFER: control vs frozen vs co-train ====================="
python - <<'PY'
import json, pathlib
def load(p): return json.load(open(p)) if pathlib.Path(p).exists() else None
ni, fr, ct = load("logs/base_spatial_nointent.json"), load("logs/base_spatial_intent.json"), load("logs/cotrain_spatial_intent.json")
def sr(d,k): return d["results"][str(k)].get("_mean_sr", float("nan")) if d else float("nan")
print(f"{'δ':>6} {'ctrl':>7} {'frozen':>7} {'cotr':>7}")
print("-"*34)
for d in ct["deltas"]:
    print(f"{d:6.2f} {sr(ni,d):7.3f} {sr(fr,d):7.3f} {sr(ct,d):7.3f}")
PY
