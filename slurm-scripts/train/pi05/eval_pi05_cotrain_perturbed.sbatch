#!/bin/bash
#SBATCH --job-name=pi05-cotrain-perturb
#SBATCH --output=logs/pi05_cotrain_perturb_%j.out
#SBATCH --error=logs/pi05_cotrain_perturb_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Co-train arm: radial XY-perturbation sweep with validity + reach/grasp/place cascade.
# Intent = co-trained VL flow map (intent_stack.pt). Compared against the already-saved
# control (base_nointent) and frozen-intent (base_intent) perturbation results.

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu                 # policy is PyTorch; keep JAX off the GPU
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

CK=${CK:-/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_vl_cotrain_v2/24000}
DELTAS="0.0,0.02,0.05,0.10,0.15,0.20"
TRIALS=10

python examples/openpi/eval_libero_perturbed.py \
    --config-name pi05_base_intent \
    --checkpoint-dir "$CK" \
    --norm-stats-from-config --fp32 \
    --vl-cotrain --intent-stack "$CK/intent_stack.pt" \
    --deltas "$DELTAS" --num-trials-per-delta $TRIALS \
    --out logs/cotrain_perturb_intent.json

echo "===================== pi05 PERTURB: control vs frozen vs co-train ====================="
python - <<'PY'
import json, pathlib
def load(p):
    return json.load(open(p)) if pathlib.Path(p).exists() else None
ni = load("logs/base_perturb_nointent.json")
fr = load("logs/base_perturb_intent.json")
ct = load("logs/cotrain_perturb_intent.json")
deltas = ct["deltas"]
def sr(d, k):   return d["results"][str(k)].get("_mean_sr", float("nan")) if d else float("nan")
def srv(d, k):  return d["results"][str(k)].get("_mean_sr_valid", float("nan")) if d else float("nan")
print(f"{'δ':>6} {'ctrl':>7} {'frozen':>7} {'cotr':>7}   {'ctrl|v':>7} {'froz|v':>7} {'cotr|v':>7}  {'val%':>6}")
print("-"*72)
for d in deltas:
    vf = ct["results"][str(d)].get("_mean_valid_frac", float("nan"))
    print(f"{d:6.2f} {sr(ni,d):7.3f} {sr(fr,d):7.3f} {sr(ct,d):7.3f}   "
          f"{srv(ni,d):7.3f} {srv(fr,d):7.3f} {srv(ct,d):7.3f}  {100*vf:5.1f}%")
PY
