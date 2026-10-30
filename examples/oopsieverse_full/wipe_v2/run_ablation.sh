#!/bin/bash
# Intent ablation (spec "Secondary analysis"): ipath s0 and W3 s0, P-devlayout, rollout seeds 0,1, modes shuffled + zero.
# At most two runs at once (ipath pair, then W3 pair); 16 workers each; rendering on parity GPUs under the shared
# per-GPU budget (>= 8 GB free); inference on GPU ${INFER:-0}. Resumable: re-run this script.
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
unset OV_TIMEOUT_OVERRIDE
cd /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/wipe_v2
LOG=/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/logs
INFER=${INFER:-0}
declare -A RD=([ipath_s0]=/edrive1/lizad/oopsieverse_full_study/runs/matrix/ipath_wipe_countertop_v1_s0
               [w3_s0]=/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/train/w3_wipe_countertop_v2_s0)
for arm in ipath_s0 w3_s0; do
  for M in shuffled zero; do
    nice -n 10 $PY eval_wipe_v2_ablate.py --intent-ablation $M --run-dir ${RD[$arm]} --ckpt step_30000.pth \
      --pools P-devlayout --rollout-seeds 0,1 --max-workers 16 --total-workers 36 --infer-gpu $INFER \
      --label abl_${arm}__$M >> $LOG/abl_${arm}__$M.log 2>&1 &
    sleep 30
  done
  wait
  echo "$(date) $arm pair done" >> $LOG/run_ablation.log
done
echo "$(date) ALL_DONE" >> $LOG/run_ablation.log
