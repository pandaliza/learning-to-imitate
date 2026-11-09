#!/usr/bin/env bash
# Evaluate one tool_hang run's selected checkpoint on the shared fresh-reset pool (background job).
# Usage: scripts/tool_hang_fresh_eval.sh RUN ROLLOUT_SEED GPU [CKPT]
#   RUN: run dir name under $TOOL_HANG_RUN_ROOT, e.g. state_mip_s0 | state_ipath_s1
#   CKPT: best (default; models/model_best.pt = first checkpoint reaching the max training-eval success)
#         | latest (models/model_latest.pt, the final 300k checkpoint)
# Pool: reset seeds RESET_BASE .. RESET_BASE + N_RESETS - 1 (defaults 100000, 500), identical for every
# policy and disjoint from the training-time eval resets. ROLLOUT_SEED seeds torch/numpy, i.e. the
# intent-sampling noise (the MIP baseline samples from zeros and is deterministic per reset).
set -euo pipefail
RUN=$1; RS=$2; GPU=$3; CKPT=${4:-best}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${TOOL_HANG_RUN_ROOT:-/edrive4/ldahiya/mip_runs/tool_hang}
RESET_BASE=${RESET_BASE:-100000}; N_RESETS=${N_RESETS:-500}; NUM_ENVS=${NUM_ENVS:-25}
case $RUN in
  state_mip_s*)   TASK=tool_hang_ph_state;;
  state_ipath_s*) TASK=tool_hang_ph_state_ipath;;
  state_ipath_sampled_s*) TASK=tool_hang_ph_state_ipath_sampled;;
  *) echo "unsupported run: $RUN" >&2; exit 1;;
esac
MODEL=$ROOT/$RUN/models/model_$CKPT.pt; [ -f "$MODEL" ] || { echo "missing $MODEL" >&2; exit 1; }
OUT=$ROOT/fresh_eval; TAG=${RUN}_${CKPT}_r${RS}; mkdir -p "$OUT/$TAG"
export MUJOCO_GL=egl HF_HOME=/edrive4/ldahiya/.hf-cache CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
rm -f "$OUT/$TAG.jsonl"
cd "$OUT/$TAG"
setsid nohup "$REPO/.venv/bin/python" "$REPO/examples/train/train_robomimic.py" mode=eval task=$TASK network=mlp \
  optimization.loss_type=mip optimization.seed=$RS optimization.model_path=$MODEL optimization.auto_resume=false \
  log.eval_reset_seed_base=$RESET_BASE log.eval_episodes=$N_RESETS task.num_envs=$NUM_ENVS \
  log.eval_results_path=$OUT/$TAG.jsonl log.save_video=false log.wandb_mode=disabled \
  log.log_dir=$OUT/$TAG hydra.run.dir=$OUT/$TAG/hydra > "$OUT/$TAG/eval.out" 2>&1 &
echo "$TAG  pid $!  gpu $GPU  -> $OUT/$TAG.jsonl"
