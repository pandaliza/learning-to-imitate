#!/bin/bash
#SBATCH --job-name=pi05-dinosaur-perturb
#SBATCH --output=logs/pi05_dinosaur_perturb_%j.out
#SBATCH --error=logs/pi05_dinosaur_perturb_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

# M3+DINOSAUR radial XY-perturbation sweep (libero_goal), same deltas/trials as the M0 control.
# Mirrors eval_pi05_m3_perturbed.sbatch but swaps the PaliGemma frozen-base tap for the LIVE
# DINOv2 tap (the DINOSAUR generator was trained on the DINOv2 agentview grid, not pi05 VL),
# so JAX policy + PyTorch DINOv2 share the GPU -> cap XLA mem fraction.
# INTENT={flow,zero} selects the intent source; zero gives the attribution control on the
# *same* checkpoint (M0 is a separately-trained no-intent policy).
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
INTENT=${INTENT:-flow}
CK=${CK:-$B/ldahiya_checkpoints/pi05_base_intent_dino_dinosaur/base_intent_dino_dinosaur_goal_run1/10000}
STACK=${STACK:-$B/ldahiya_checkpoints/slot_intent_dino_dinosaur_60k/intent_stack_98000.pt}
TASKCFG=${TASKCFG:-libero_goal_suite_image_slot_intent_dino_dinosaur}
OUT=${OUT:-logs/dino_dinosaur_perturb_${INTENT}.json}

python examples/openpi/eval_libero_perturbed.py \
    --config-name pi05_base_intent_dino_dinosaur --checkpoint-dir "$CK" \
    --vl-cotrain --intent-task-config "$TASKCFG" --intent-stack "$STACK" \
    --encoder-tap dinov2 --intent-override "$INTENT" \
    --deltas 0.0,0.02,0.05,0.10,0.15,0.20 --num-trials-per-delta 10 \
    --out "$OUT"

echo "===================== PERTURB: M0 (ctrl) vs M3 vs M3+DINOSAUR ====================="
python - "$OUT" <<'PY'
import json, pathlib, sys
def L(p): return json.load(open(p)) if pathlib.Path(p).exists() else None
ni  = L("logs/base_perturb_nointent.json")   # M0 (no-intent policy)
m3  = L("logs/m3_perturb_intent.json")        # M3 (PaliGemma intent)
dn  = L(sys.argv[1])                          # M3+DINOSAUR (this run)
def sr(d,k): return d["results"][str(k)].get("_mean_sr",float("nan")) if d else float("nan")
print(f"{'d':>6} {'M0':>7} {'M3':>7} {'DINOSAUR':>9}")
for d in dn["deltas"]:
    print(f"{d:6.2f} {sr(ni,d):7.3f} {sr(m3,d):7.3f} {sr(dn,d):9.3f}")
PY
