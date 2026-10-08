#!/bin/bash
# Amendment 1: remaining 500-step smoke TRAININGS on GPU 0 (no rendering), <= 3 concurrent jobs x (1+4 workers).
# Memory rule: launch only if GPU 0 has >= 25 GB free; wait otherwise.
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
M=/scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/matrix
R=$OVS/runs/matrix_smoke; mkdir -p $R/logs
free0() { nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits; }
pids=()
for spec in "$@"; do
  IFS=: read task arm <<< "$spec"; id=${arm}_${task}_v1_s0__smoke500
  [ -f $R/$id/DONE.json ] && continue
  while [ $(jobs -rp | wc -l) -ge 3 ] || [ $(free0) -lt 25600 ]; do sleep 15; done
  echo "$(date) train $id"
  ( $PY $M/train_matrix.py --arm $arm --task $task --seed 0 --num-steps 500 --save-every 500 --label smoke --workers 4 \
      --act-ckpt --runs-root $R > $R/logs/$id.train_gpu0.log 2>&1 && echo "$(date) DONE $id" || echo "$(date) FAIL $id" ) &
  sleep 60   # let the job reach its memory peak before the next launch decision
done
wait; echo SMOKE_TRAIN_DONE
