#!/bin/bash
# Auto-chain to stage-2 (M8-WSM): wait for the w-cache precompute, run a 1-GPU smoke of the M8 arm as a
# safety gate (real w cache, ~20 steps), and ONLY sbatch the 4-GPU job if the smoke passes. Protects the
# 48h x 4-GPU allocation from any wiring bug that couldn't be tested until the w cache existed.
#
#   PRECOMPUTE_PID=<pid> bash examples/openpi/chain_stage2.sh   (run in background / nohup)
set -uo pipefail
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
WCACHE=$B/w_cache_goal
ART=.venv-label-artifacts
SMOKE_OUT=$ART/m8_smoke_out
SMOKE_LOG=$ART/m8_smoke.log
STAGE2_OUT=$B/ldahiya_checkpoints/pi05_m8_wsm_goal_run1

# 1) wait for precompute_w
if [ -n "${PRECOMPUTE_PID:-}" ]; then
  echo "[chain] waiting for precompute_w PID $PRECOMPUTE_PID ..."
  while kill -0 "$PRECOMPUTE_PID" 2>/dev/null; do sleep 15; done
fi
N=$(ls "$WCACHE"/*.npy 2>/dev/null | wc -l)
echo "[chain] w cache ready: $N demos"
if [ "$N" -lt 100 ]; then echo "[chain] ABORT: w cache too small ($N)"; exit 1; fi

# 2) 1-GPU smoke of the M8 arm (exercises dataset w loading + head + losses + ckpt on real data)
echo "[chain] running 1-GPU M8 smoke (20 steps) -> $SMOKE_LOG"
rm -rf "$SMOKE_OUT"
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu FORCE_FP32=1 PYTHONPATH=. \
  python examples/openpi/train_pi05_cotrain.py \
    --pi05-config pi05_base_nointent \
    --slot-task-config libero_goal_suite_image_slot_intent_vl \
    --pi05-weights $B/pi05_base_pytorch \
    --wsm-intent --wsm-w-cache-dir "$WCACHE" \
    --steps 20 --batch-size 2 --accum-steps 1 --save-every 10 \
    --device cuda --out "$SMOKE_OUT" > "$SMOKE_LOG" 2>&1
RC=$?
if [ $RC -ne 0 ] || ! grep -q "\[M8-WSM\]" "$SMOKE_LOG" || ! grep -qE "step 10:" "$SMOKE_LOG"; then
  echo "[chain] ABORT: smoke failed (rc=$RC). Not submitting the 4-GPU job. See $SMOKE_LOG"
  tail -20 "$SMOKE_LOG"
  exit 1
fi
echo "[chain] smoke PASSED:"; grep -E "\[M8-WSM\]|step (0|10):" "$SMOKE_LOG" | head

# 3) submit the real 4-GPU job
echo "[chain] submitting 4-GPU stage-2 -> $STAGE2_OUT"
OUT="$STAGE2_OUT" sbatch slurm-scripts/train/pi05/train_pi05_m8_wsm.sbatch
echo "[chain] done. squeue:"; squeue -u ldahiya --format="%.10i %.22j %.8T %.10M %R" | head
