#!/bin/bash
#SBATCH --job-name=m9-attach
#SBATCH --output=logs/m9_attach_%j.out
#SBATCH --error=logs/m9_attach_%j.err
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-n5-32,babel-n9-32,babel-o5-16,babel-o5-28
# Default partition (general) + normal QOS: 8-GPU cap, shared with the other pi05 arms.
# NOT maxlab_qos: that QOS is only accepted by --partition=maxlab, whose 2 nodes carry
# RTX_PRO_6000 (not L40S), so the gres line below would never match there. To use it:
#   #SBATCH --partition=maxlab --qos=maxlab_qos --gres=gpu:RTX_PRO_6000:4
# (16 GPUs total in that partition and contended, so check `squeue -p maxlab` first.)

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

# M9-ATTACH: the stripped intent head with a selectable tap.
#   ATTACH=vl     -> head reads the LIVE PaliGemma prefix (image-token mean, 2048). The aux gradient
#                    shapes ONLY the VLM (LoRA adapters + SigLIP vision tower); the action expert never
#                    sees it. Max's words taken literally; the no-sharing arm of Sarvesh's question.
#   ATTACH=penult -> identical to train_pi05_m9.py (shared trunk). Default.
#
# --qos=maxlab_qos: 16-GPU cap (vs normal's 8) and NOT preemptable. normal's 8 are already spoken for by
# the M8/M9 runs; these caps are per-QOS and stack, so this schedules alongside them rather than queueing.
#
# The VL tap is taken from the SAME DDP forward via return_vl_mean=True -- no second VL pass, so the
# per-step cost matches M9-penult. (A separate raw_model.vl_image_features() call instead crashes DDP
# with "Expected to mark a variable ready only once": the LoRA params would get two grad hooks per step.)
# Watch: intent_fm should fall from ~1.24 as in M9, and the step-0 [M9-ATTACH check] must report a
# non-zero count of VLM params receiving gradient -- it raises rather than train a silent no-op.
B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
OUT=${OUT:?set OUT}

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC:-4} \
    examples/openpi/train_pi05_m9_attach.py \
    --pi05-config pi05_base_nointent \
    --slot-task-config libero_goal_suite_image_slot_intent_vl \
    --pi05-weights $B/pi05_base_pytorch \
    --intent-flow \
    --intent-flow-attach ${ATTACH:-vl} \
    --intent-flow-decoder ${DECODER:-flow} \
    --intent-flow-target ${TARGET:-concat} \
    --intent-flow-weight ${INTENT_W:-0.25} \
    --steps ${STEPS:-30000} \
    --batch-size ${BS:-2} \
    --accum-steps ${ACCUM:-4} \
    --lr ${LR:-5e-5} \
    --save-every 2000 \
    ${RESUME:+--resume-from "$RESUME"} \
    --out "$OUT"
