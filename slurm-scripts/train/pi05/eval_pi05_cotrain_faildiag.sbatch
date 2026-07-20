#!/bin/bash
#SBATCH --job-name=pi05-cotrain-faildiag
#SBATCH --output=logs/pi05_cotrain_faildiag_%j.out
#SBATCH --error=logs/pi05_cotrain_faildiag_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00

# Co-train failure cascade (reach -> grasp -> place), 20 trials for power, tasks 1/6/8.
# Compared against the saved control + frozen faildiag results: does co-train reduce
# the no_reach rate (the entire OOD bottleneck) that frozen intent could not?

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
    --tasks 1,6,8 --deltas 0.0,0.05,0.10 --num-trials-per-delta 20 \
    --out logs/cotrain_faildiag.json

echo "===================== CO-TRAIN FAILURE CASCADE (valid trials) ====================="
python - <<'PY'
import json
from collections import Counter
d = json.load(open("logs/cotrain_faildiag.json"))
print(f"{'δ':>5} {'N':>4} {'reach':>7} {'grasp':>7} {'place':>7}   breakdown")
print("-"*64)
for dl in d["deltas"]:
    agg = Counter()
    for tid,v in d["results"][str(dl)].items():
        if tid.startswith("_"): continue
        agg.update(v.get("fail_breakdown",{}))
    N = sum(agg.values()) or 1
    reach = N - agg["no_reach"]; grasp = reach - agg["no_grasp"]; place = agg["success"]
    b = "  ".join(f"{k}={agg.get(k,0)}" for k in ("success","no_reach","no_grasp","no_place"))
    print(f"{dl:5.2f} {N:4d} {reach/N:7.2f} {grasp/N:7.2f} {place/N:7.2f}   {b}")
PY
