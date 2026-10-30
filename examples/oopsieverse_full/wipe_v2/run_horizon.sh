#!/bin/bash
# Execution-horizon sweep (spec addendum): A-MIX s1 and I-PATH s0, P-demo + P-devlayout, rollout seeds 0,1; h=4 first,
# then h=1 (h=8 = existing records). One supervisor at a time, 32 workers, inference GPU ${INFER:-0}. Resumable.
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
unset OV_TIMEOUT_OVERRIDE WV2_HORIZON_ROOT_SMOKE_ONLY WV2_EVAL_ROOT_SMOKE_ONLY
cd /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/wipe_v2
LOG=/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/logs
M=/edrive1/lizad/oopsieverse_full_study/runs/matrix
for H in 4 1; do
  nice -n 10 $PY eval_wipe_v2_horizon.py --exec-horizon $H --run-dir $M/amix_wipe_countertop_v1_s1 \
    --run-dir $M/ipath_wipe_countertop_v1_s0 --pools P-demo,P-devlayout --rollout-seeds 0,1 --max-workers 32 \
    --total-workers 36 --infer-gpu ${INFER:-0} --label horizon_h$H >> $LOG/horizon_h$H.log 2>&1
  echo "$(date) h=$H done rc=$?" >> $LOG/run_horizon.log
done
echo "$(date) ALL_DONE" >> $LOG/run_horizon.log
