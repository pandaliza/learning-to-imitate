#!/bin/bash
#SBATCH --job-name=pi05-base-nointent-sweep
#SBATCH --output=logs/pi05_base_nointent_sweep_%j.out
#SBATCH --error=logs/pi05_base_nointent_sweep_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00

# Checkpoint-vs-SR sweep for the pi05_base_nointent run (10k, 20k, 29999).
# Outputs one JSON per checkpoint; prints a summary table at the end.

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

BASE=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/ldahiya_checkpoints/pi05_base_nointent/base_nointent_goal_run1
TRIALS=20

for STEP in 10000 20000 29999; do
    echo "=== nointent step ${STEP} ==="
    python examples/openpi/eval_libero_intent.py \
        --config-name pi05_base_nointent \
        --checkpoint-dir "$BASE/$STEP" \
        --task-suite libero_goal \
        --num-trials-per-task $TRIALS \
        --out "logs/eval_base_nointent_${STEP}.json"
done

echo "===================== NOINTENT CKPT-vs-SR ====================="
python - <<'PY'
import json, pathlib
print(f"{'step':>8}  {'mean_SR':>8}  {'succ/ep':>10}  per-task")
print("-" * 70)
for step in (10000, 20000, 29999):
    p = pathlib.Path(f"logs/eval_base_nointent_{step}.json")
    if not p.exists():
        print(f"{step:>8}  MISSING")
        continue
    d = json.load(open(p))
    per = "  ".join(f"{t}:{v:.2f}" for t, v in d["per_task"].items())
    print(f"{step:>8}  {d['mean_sr']:>8.3f}  {d['total']:>10}  {per}")
PY
