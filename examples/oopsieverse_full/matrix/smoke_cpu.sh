#!/bin/bash
# CPU fallback for the 500-step smoke TRAININGS while GPUs 5-7 have < 25 GB free (other users' reservations).
# Same code path as GPU training except device; 3 concurrent jobs x 14 threads (<= 48 cores). Run ids match
# smoke_all.sh, which then performs the GPU rollouts when memory frees.
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
M=/scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/matrix
R=$OVS/runs/matrix_smoke
export OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=""
run() { IFS=: read task arm <<< "$1"; id=${arm}_${task}_v1_s0__smoke500
  [ -f $R/$id/DONE.json ] && return
  $PY $M/train_matrix.py --arm $arm --task $task --seed 0 --num-steps 500 --save-every 500 --label smoke --workers 0 \
     --device cpu --runs-root $R > $R/logs/$id.train_cpu.log 2>&1 && echo "$(date) CPU_DONE $id" || echo "$(date) CPU_FAIL $id"; }
for group in "${@}"; do
  for spec in $group; do run $spec & done; wait
done
echo SMOKE_CPU_DONE
