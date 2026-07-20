#!/bin/bash
#SBATCH --job-name=pi05-cotrain-override
#SBATCH --output=logs/pi05_cotrain_override_%j.out
#SBATCH --error=logs/pi05_cotrain_override_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# THE decisive co-train test: does the co-trained intent channel have causal authority?
# The frozen arm showed flow==zero (channel inert). If co-train flow > zero, joint training
# gave the adaRMS channel authority over the policy. flow/zero/random on tasks 1,5,6.

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
COMMON="--config-name pi05_base_intent --checkpoint-dir $CK --norm-stats-from-config --fp32 \
        --vl-cotrain --intent-stack $CK/intent_stack.pt \
        --tasks 1,5,6 --deltas 0.0,0.05,0.10 --num-trials-per-delta 10"

python examples/openpi/eval_libero_perturbed.py $COMMON --intent-override flow   --out logs/cotrain_override_flow.json
python examples/openpi/eval_libero_perturbed.py $COMMON --intent-override zero   --out logs/cotrain_override_zero.json
python examples/openpi/eval_libero_perturbed.py $COMMON --intent-override random --out logs/cotrain_override_random.json

echo "===================== CO-TRAIN INTENT OVERRIDE ====================="
python - <<'PY'
import json, numpy as np
res = {m: json.load(open(f"logs/cotrain_override_{m}.json"))["results"] for m in ("flow","zero","random")}
deltas = json.load(open("logs/cotrain_override_flow.json"))["deltas"]
print(f"{'delta':>8}  {'flow':>8}  {'zero':>8}  {'random':>8}  {'flow-zero':>10}  {'flow-random':>12}")
print("-"*68)
for d in deltas:
    row = {}
    for m in ("flow","zero","random"):
        srs = [v["sr"] for k,v in res[m][str(d)].items() if not k.startswith("_")]
        row[m] = float(np.mean(srs)) if srs else float("nan")
    print(f"{d:8.2f}  {row['flow']:8.3f}  {row['zero']:8.3f}  {row['random']:8.3f}"
          f"  {row['flow']-row['zero']:+10.3f}  {row['flow']-row['random']:+12.3f}")
PY
