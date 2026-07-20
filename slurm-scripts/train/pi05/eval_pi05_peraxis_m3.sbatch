#!/bin/bash
#SBATCH --job-name=pi05-peraxis-m3
#SBATCH --output=logs/pi05_peraxis_m3_%j.out
#SBATCH --error=logs/pi05_peraxis_m3_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00

# Per-axis perturbation, M3 (frozen VL-grounded intent). x-only vs y-only.
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
CK=$B/ldahiya_checkpoints/pi05_base_intent_m3/base_intent_m3_goal_run1/29999
VLSTACK=$B/ldahiya_checkpoints/slot_intent_vl/intent_stack_32000.pt
FROZEN=$B/pi05_base_pytorch
DELTAS="0.0,0.02,0.03,0.05,0.10"

for AX in x y; do
  python examples/openpi/eval_libero_perturbed.py \
      --config-name pi05_base_intent_m3 --checkpoint-dir "$CK" --vl-cotrain \
      --intent-task-config libero_goal_suite_image_slot_intent_vl --intent-stack "$VLSTACK" \
      --frozen-base-weights "$FROZEN" --frozen-base-config pi05_base_nointent \
      --axis $AX --deltas "$DELTAS" --num-trials-per-delta 10 \
      --out logs/peraxis_m3_${AX}.json
done

echo "===================== PER-AXIS: control vs M3 (x and y) ====================="
python - <<'PY'
import json, pathlib
def L(p): return json.load(open(p)) if pathlib.Path(p).exists() else None
def sr(d,k):  return d["results"][str(k)].get("_mean_sr",float("nan")) if d else float("nan")
def srv(d,k): return d["results"][str(k)].get("_mean_sr_valid",float("nan")) if d else float("nan")
for ax in ("x","y"):
    c,m3=L(f"logs/peraxis_control_{ax}.json"),L(f"logs/peraxis_m3_{ax}.json")
    if not m3: continue
    print(f"\n--- {ax.upper()}-AXIS ({'pure OOD' if ax=='x' else 'in-dist to ~3σ'}) ---")
    print(f"{'δ':>6} {'ctrl':>7} {'M3':>7} {'M3-ctrl':>8}  {'ctrl|v':>7} {'M3|v':>7}")
    for d in m3["deltas"]:
        cc,a3=sr(c,d),sr(m3,d)
        print(f"{d:6.2f} {cc:7.3f} {a3:7.3f} {a3-cc:+8.3f}  {srv(c,d):7.3f} {srv(m3,d):7.3f}")
PY
