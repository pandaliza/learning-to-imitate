#!/bin/bash
#SBATCH --job-name=pi05-base-faildiag
#SBATCH --output=logs/pi05_base_faildiag_%j.out
#SBATCH --error=logs/pi05_base_faildiag_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00

# Failure-mode diagnostic: decompose each episode into reach -> grasp -> place and
# compare where the intent vs control policies break under perturbation.
# Clean-grasp pick&place tasks only (bowls + cream cheese; wine bottle rolls -> noisy lift).
# 20 trials/delta for statistical power in the OOD regime where the intent hint appeared.

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

CKPT_BASE=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints
NOINTENT_CKPT=$CKPT_BASE/pi05_base_nointent/base_nointent_goal_run1/29999
INTENT_CKPT=$CKPT_BASE/pi05_base_intent/base_intent_goal_run1/29999
FLOW_MAP=/data/user_data/ldahiya/libero-train/libero-goal-suite-image/slot_intent/models/model_best.pt

TASKS="1,6,8"            # put bowl on stove / cream cheese in bowl / bowl on plate
DELTAS="0.0,0.05,0.10"
TRIALS=20

COMMON="--task-suite libero_goal --tasks $TASKS --deltas $DELTAS --num-trials-per-delta $TRIALS"

python examples/openpi/eval_libero_perturbed.py $COMMON \
    --config-name pi05_base_nointent \
    --checkpoint-dir "$NOINTENT_CKPT" \
    --out logs/faildiag_nointent.json

python examples/openpi/eval_libero_perturbed.py $COMMON \
    --config-name pi05_base_intent \
    --checkpoint-dir "$INTENT_CKPT" \
    --intent \
    --intent-task-config libero_goal_suite_image_slot_intent \
    --intent-ckpt "$FLOW_MAP" \
    --out logs/faildiag_intent.json

echo "===================== FAILURE-STAGE CASCADE (over valid trials) ====================="
python - <<'PY'
import json
from collections import Counter

def cascade(path):
    d = json.load(open(path)); out = {}
    for dl in d["deltas"]:
        agg = Counter()
        for tid, v in d["results"][str(dl)].items():
            if tid.startswith("_"):
                continue
            agg.update(v.get("fail_breakdown", {}))
        valid = sum(agg.values()) or 1
        reached = valid - agg["no_reach"]
        grasped = reached - agg["no_grasp"]
        placed  = agg["success"]
        out[dl] = dict(valid=valid,
                       reach=reached/valid, grasp=grasped/valid, place=placed/valid,
                       raw=dict(agg))
    return out

ni = cascade("logs/faildiag_nointent.json")
it = cascade("logs/faildiag_intent.json")

print(f"{'δ':>5} {'arm':>9} {'N':>4} {'reach':>7} {'grasp':>7} {'place':>7}   breakdown")
print("-"*78)
for dl in json.load(open('logs/faildiag_nointent.json'))["deltas"]:
    for name, c in (("nointent", ni[dl]), ("intent", it[dl])):
        b = "  ".join(f"{k}={c['raw'].get(k,0)}" for k in ("success","no_reach","no_grasp","no_place"))
        print(f"{dl:5.2f} {name:>9} {c['valid']:4d} {c['reach']:7.2f} {c['grasp']:7.2f} {c['place']:7.2f}   {b}")
    a, b = ni[dl], it[dl]
    print(f"{'':5} {'Δ(it-ni)':>9} {'':4} {b['reach']-a['reach']:+7.2f} {b['grasp']-a['grasp']:+7.2f} {b['place']-a['place']:+7.2f}")
    print()
PY
