#!/bin/bash
# all60 matched-condition eval queue (coordinator order): stage1 amix s0 + ipath s1; stage2 amix s1 + ipath s2;
# stage3 amix s2 + ipath s0, each started only once its DONE.json exists. Resumable: rerun this script; finished
# records are skipped by the evaluator. Status: runs/wipe_v2/logs/all60_queue.status
cd "$(dirname "$0")"; source ../baseline/envsetup.sh
T=/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/all60/train
LOGS=/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/logs; ST=$LOGS/all60_queue.status
run_stage() {  # label run...
  local label=$1; shift; local args=()
  for r in "$@"; do args+=(--run-dir $T/$r); done
  echo "$(date -Is) START $label $*" >> $ST
  nice -n 10 $PY eval_wipe_v2_all60.py "${args[@]}" --pools P-demo60,P-layout60,P-random --rollout-seeds 0,1,2 \
      --max-workers 32 --total-workers 36 --label $label > $LOGS/$label.log 2>&1
  echo "$(date -Is) EXIT $? $label" >> $ST
}
run_stage all60_stage1 amix_wipe_countertop_all60_s0 ipath_wipe_countertop_all60_s1
run_stage all60_stage2 amix_wipe_countertop_all60_s1 ipath_wipe_countertop_all60_s2
echo "$(date -Is) WAIT DONE.json amix_s2 ipath_s0" >> $ST
for r in amix_wipe_countertop_all60_s2 ipath_wipe_countertop_all60_s0; do
  until [ -f $T/$r/DONE.json ] && [ -f $T/$r/ckpt/step_30000.pth ]; do sleep 120; done
done
run_stage all60_stage3 amix_wipe_countertop_all60_s2 ipath_wipe_countertop_all60_s0
echo "$(date -Is) QUEUE_DONE" >> $ST
