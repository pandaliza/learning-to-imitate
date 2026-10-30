#!/bin/bash
# All-60 I-PATH rollout-time intent ablation (spec addendum 2026-10-29 23:31 UTC): modes shuffled + zero, P-layout60,
# three checkpoints (s0,s1,s2), rollout seeds 0,1,2. One supervisor per mode (the ablation mode is a process-wide env var),
# both on GPU 3 only (render and inference), total worker cap 24 shared through the GPU ledger. Resumable: rerun this
# script; finished records are skipped by the evaluator. Status: runs/wipe_v2/logs/all60_ablation_queue.status
cd "$(dirname "$0")"; source ../baseline/envsetup.sh
unset WV2_ALL60_ABL_ROOT_SMOKE_ONLY
T=/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/all60/train
LOGS=/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/logs; ST=$LOGS/all60_ablation_queue.status
RUNS=""; for s in 0 1 2; do RUNS="$RUNS --run-dir $T/ipath_wipe_countertop_all60_s$s"; done
for M in shuffled zero; do
  ( echo "$(date -Is) START $M" >> $ST
    nice -n 10 $PY eval_wipe_v2_ablate_all60.py --intent-ablation $M $RUNS --pools P-layout60 --rollout-seeds 0,1,2 \
      --max-workers 24 --total-workers 24 --render-gpus 3 --infer-gpu 3 --label all60_abl_$M \
      >> $LOGS/all60_abl_$M.log 2>&1
    echo "$(date -Is) EXIT $? $M" >> $ST ) &
  sleep 20
done
wait
echo "$(date -Is) QUEUE_DONE" >> $ST
