#!/usr/bin/env bash
# Launch one robomimic tool_hang (ph, state) run in the background.
# Usage: scripts/tool_hang_run.sh ARM SEED GPU [extra hydra overrides...]
#   ARM: abs         -> flow + mlp, low_dim_abs.hdf5 + absolute OSC control (task.abs_control=true)
#        delta       -> flow + mlp, low_dim.hdf5 + delta OSC control (repo default; upstream "delta_legacy")
#        mip         -> MIP loss + mlp, delta_legacy (upstream release: 87-95% on tool_hang ph state)
#        abs_chiunet -> flow + chiunet, absolute (upstream release: 80-87%); checks the abs_control fix
#        ipath       -> mip arm + I-PATH intent (task tool_hang_ph_state_ipath); same obs/actions/budget/evals
#        ipath_sampled -> ipath, but the action policy trains on sampled intent after a 20k GT warm-up
# Upstream MIP checkpoints/logs: huggingface.co/ChaoyiPan/mip-checkpoints (robomimic/tool_hang_ph_state/).
set -euo pipefail
ARM=$1; SEED=${2:-0}; GPU=$3; shift 3
REPO=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${TOOL_HANG_RUN_ROOT:-/edrive4/ldahiya/mip_runs/tool_hang}
NET=mlp; LOSS=flow; STEPS=200000
case $ARM in
  abs)   TASK=tool_hang_ph_state_abs;;
  delta) TASK=tool_hang_ph_state;;
  mip)   TASK=tool_hang_ph_state; LOSS=mip; STEPS=300000;;
  abs_chiunet) TASK=tool_hang_ph_state_abs; NET=chiunet; STEPS=300000;;
  ipath) TASK=tool_hang_ph_state_ipath; LOSS=mip; STEPS=300000;;
  ipath_sampled) TASK=tool_hang_ph_state_ipath_sampled; LOSS=mip; STEPS=300000;;
  *) echo "unknown arm: $ARM" >&2; exit 1;;
esac
RUN=state_${ARM}_s${SEED}; DIR=$ROOT/$RUN; mkdir -p "$DIR"
export MUJOCO_GL=egl HF_HOME=/edrive4/ldahiya/.hf-cache WANDB_DIR=$DIR CUDA_VISIBLE_DEVICES=$GPU
export MUJOCO_EGL_DEVICE_ID=$GPU PYTHONUNBUFFERED=1 WANDB_CONSOLE=off OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# CWD = run dir: mip/logger.py writes "best" checkpoints to a CWD-relative checkpoints/ dir
# that auto_resume reads, so runs launched from one directory would resume from each other.
cd "$DIR"
setsid nohup "$REPO/.venv/bin/python" "$REPO/examples/train/train_robomimic.py" task=$TASK network=$NET \
  optimization.loss_type=$LOSS optimization.seed=$SEED optimization.batch_size=1024 optimization.gradient_steps=$STEPS \
  log.eval_episodes=50 log.eval_freq=20000 log.save_freq=50000 task.num_envs=10 \
  log.wandb_mode=offline log.project=tool_hang log.group=state_$ARM log.exp_name=$RUN \
  log.log_dir=$DIR hydra.run.dir=$DIR/hydra "$@" > "$DIR/train.out" 2>&1 &
echo "$RUN  pid $!  gpu $GPU  -> $DIR/train.out"
