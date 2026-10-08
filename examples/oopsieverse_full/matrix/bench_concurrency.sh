#!/bin/bash
# Throughput benchmark: jobs per GPU x DataLoader workers per job. Each job: ipathg (heaviest arm) pick_egg,
# 150 s measurement window after 60 s warm-up (train_matrix --max-wall-s). Usage: bench_concurrency.sh ROUND "gpu:jobs:workers ..."
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
M=/scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/matrix
ROUND=$1; shift
R=$OVS/runs/matrix_dev/bench/$ROUND; mkdir -p $R
ARM=${ARM:-ipathg}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MUJOCO_EGL_DEVICE_ID=0 EGL_DEVICE_ID=0
pids=()
for spec in $1; do
  IFS=: read g n w <<< "$spec"
  for i in $(seq 1 $n); do
    CUDA_VISIBLE_DEVICES=$g $PY $M/train_matrix.py --arm $ARM --task pick_egg --seed $i --num-steps 30000 --label bench \
      --workers $w --act-ckpt --max-wall-s ${WIN:-150} --runs-root $R --run-id g${g}_n${n}_w${w}_j${i}_$ARM --no-ledger > $R/g${g}_n${n}_w${w}_j${i}_$ARM.log 2>&1 &
    pids+=($!)
  done
done
wait "${pids[@]}"
echo ROUND_DONE $ROUND
