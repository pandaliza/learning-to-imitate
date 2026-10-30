#!/bin/bash
# wipe v2 calibration (spec): W0 = amix_wipe_countertop_v1_s{0,1,2} step_30000 on P-train, P-devlayout, P-random x
# rollout seeds {0,1}; the other 5 primary wipe arms (seed 0) on P-train, P-devlayout. Two supervisors share the
# per-GPU render budget ledger (runs/wipe_v2/_gpus); 18 + 18 workers (CPU cap 40). Resumable: re-run this script.
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
unset OV_TIMEOUT_OVERRIDE
cd /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/wipe_v2
M=/edrive1/lizad/oopsieverse_full_study/runs/matrix
LOG=/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/logs
W=${W:-18}
nohup nice -n 10 $PY eval_wipe_v2.py --run-dir $M/amix_wipe_countertop_v1_s0 --run-dir $M/amix_wipe_countertop_v1_s1 \
  --run-dir $M/amix_wipe_countertop_v1_s2 --pools P-train,P-devlayout,P-random --rollout-seeds 0,1 \
  --max-workers $W --total-workers 36 --label calib_w0 >> $LOG/calib_w0.log 2>&1 &
sleep 60
nohup nice -n 10 $PY eval_wipe_v2.py $(for a in asafe imean ipath ipathg auxpathg; do echo --run-dir $M/${a}_wipe_countertop_v1_s0; done) \
  --pools P-train,P-devlayout --rollout-seeds 0,1 --max-workers $W --total-workers 36 --label calib_other_s0 >> $LOG/calib_other_s0.log 2>&1 &
