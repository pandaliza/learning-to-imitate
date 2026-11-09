#!/usr/bin/env bash
# Launch one DexMimicGen TwoArmThreading run in the background.
# Usage: [DEXMG_TASK=drawer_cleanup] scripts/dexmg_threading_run.sh ARM SEED GPU [extra hydra overrides...]
#   ARM: baseline | mean8 | final_h2 | final_h4 | final_h6 | final_h8 | wp4 | wp4_sampled | hier_mean8 | hier_wp4
# Hyperparameters follow the repo's earlier image flow-intent runs
# (slurm-scripts/train/lift/train_lift_mh_image_flow_intent_h32.sbatch).
set -euo pipefail
ARM=$1; SEED=${2:-0}; GPU=$3; shift 3
REPO=$(cd "$(dirname "$0")/.." && pwd)
T=${DEXMG_TASK:-threading}   # dexmg task name: threading | drawer_cleanup | ...
ROOT=${DEXMG_RUN_ROOT:-/edrive4/ldahiya/mip_runs/dexmg_$T}
case $ARM in
  baseline) TASK=dexmg_${T}_image;                  NET=chiunet;;
  mean8)    TASK=dexmg_${T}_image_flow_intent;      NET=mlp_flow_intent_chiunet;;
  final_h*) TASK=dexmg_${T}_image_flow_intent_$ARM; NET=mlp_flow_intent_chiunet;;
  wp4)      TASK=dexmg_${T}_image_flow_intent_wp4;  NET=mlp_flow_intent_chiunet;;
  wp4_sampled) TASK=dexmg_${T}_image_flow_intent_wp4_sampled; NET=mlp_flow_intent_chiunet;;
  hier_*)   TASK=dexmg_${T}_image_$ARM;             NET=chiunet;;  # Config B: predictor + chiunet
  *) echo "unknown arm: $ARM" >&2; exit 1;;
esac
RUN=${ARM}_s${SEED}; DIR=$ROOT/$RUN; mkdir -p "$DIR"
export MUJOCO_GL=egl HF_HOME=/edrive4/ldahiya/.hf-cache WANDB_DIR=$DIR CUDA_VISIBLE_DEVICES=$GPU
# Render eval envs on the same physical GPU as training (EGL ignores CUDA_VISIBLE_DEVICES; the
# dataset pins render_gpu_device_id=0). Unbuffered logs and no wandb console capture so that
# train.out is complete if the process dies. setsid detaches from the launching session.
export MUJOCO_EGL_DEVICE_ID=$GPU PYTHONUNBUFFERED=1 WANDB_CONSOLE=off
# Cap CPU threads: many concurrent runs x 96 OMP threads each only adds contention.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# Run with CWD = the run dir: mip/logger.py writes "best" checkpoints to a CWD-relative
# checkpoints/ dir keyed by a name that ignores the intent arm, and auto_resume reads it.
# Running from the repo root makes arms (and smoke tests) resume from / overwrite each other.
cd "$DIR"
setsid nohup "$REPO/.venv/bin/python" "$REPO/examples/train/train_robomimic.py" task=$TASK network=$NET \
  optimization.seed=$SEED optimization.batch_size=64 optimization.gradient_steps=300000 \
  log.eval_episodes=50 log.eval_freq=20000 log.save_freq=50000 task.num_envs=5 \
  log.wandb_mode=offline log.project=dexmg_intent log.group=$T log.exp_name=$RUN \
  log.log_dir=$DIR hydra.run.dir=$DIR/hydra "$@" >> "$DIR/train.out" 2>&1 &
echo "$RUN  pid $!  gpu $GPU  -> $DIR/train.out"
