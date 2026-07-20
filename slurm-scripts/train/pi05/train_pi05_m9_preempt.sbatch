#!/bin/bash
#SBATCH --job-name=m9-pre
#SBATCH --output=logs/m9_pre_%j.out
#SBATCH --error=logs/m9_pre_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --exclude=babel-n5-32,babel-n9-32,babel-o5-16,babel-o5-28

# M9 on the PREEMPT partition (24-GPU cap, priority 0, KILLABLE at any moment).
#
# Preemption safety: --requeue tells Slurm to put the job back in the queue when it is killed, and the
# auto-resume below restarts from the newest checkpoint in $OUT. Together these make a preempted run
# self-healing; WITHOUT the auto-resume, a requeued job would silently restart from step 0 and quietly
# overwrite good checkpoints -- worse than not requeueing at all.
#
# What resume restores: model.safetensors + the head .pt + the step counter, and the LR schedule is
# fast-forwarded (`for _ in range(start_step)`).
# What it does NOT restore: OPTIMIZER STATE (Adam moments are not checkpointed). Each preemption resets
# them; they rebuild over a few hundred steps, so expect a small bump in the loss curve at each restart.
# Worst case per preemption = (steps since last save) lost, <= --save-every (2000, ~2.8h at ~5s/step).
#
# Use this for EXPENDABLE sweeps (target/decoder/lambda/h). Keep the arms that answer real questions on
# the non-preemptable `general`+normal partition.
#
#   OUT=.../pi05_m9_pre_mean ATTACH=penult TARGET=mean sbatch train_pi05_m9_preempt.sbatch

set -euo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FORCE_FP32=1
export NCCL_P2P_DISABLE=1
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
OUT=${OUT:?set OUT}

# --- auto-resume: pick the newest COMPLETE checkpoint in $OUT (numeric dirs holding model.safetensors).
# A checkpoint mid-write when the preemption landed would be truncated, so require the file to exist.
RESUME_ARG=""
if [ -d "$OUT" ]; then
  for d in $(ls -1 "$OUT" 2>/dev/null | grep -E '^[0-9]+$' | sort -rn); do
    if [ -f "$OUT/$d/model.safetensors" ]; then
      RESUME_ARG="--resume-from $OUT/$d"
      echo "[preempt] restart #${SLURM_RESTART_COUNT:-0}: resuming from step $d"
      break
    fi
  done
fi
[ -z "$RESUME_ARG" ] && echo "[preempt] no checkpoint found -> fresh start from step 0"

srun torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC:-4} \
    examples/openpi/train_pi05_m9_attach.py \
    --pi05-config pi05_base_nointent \
    --slot-task-config libero_goal_suite_image_slot_intent_vl \
    --pi05-weights $B/pi05_base_pytorch \
    --intent-flow \
    --intent-flow-attach ${ATTACH:-penult} \
    --intent-flow-decoder ${DECODER:-flow} \
    --intent-flow-target ${TARGET:-mean} \
    --intent-flow-weight ${INTENT_W:-0.25} \
    --steps ${STEPS:-30000} \
    --batch-size ${BS:-2} \
    --accum-steps ${ACCUM:-4} \
    --lr ${LR:-5e-5} \
    --save-every 2000 \
    $RESUME_ARG \
    --out "$OUT"
