#!/bin/bash
# Phase D TEST branching (pick_egg, sel_test_parents, training seeds 0,1,2; protocol Amendment 7 item 5), multi-GPU
# rendering on exact-parity GPUs (Amendment 6). Resumable: rerun to resume. Per-seed stop:
#   touch /edrive1/lizad/oopsieverse_full_study/runs/selection/pick_egg/sel_test_parents/train_seed_<s>/STOP
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
cd /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/selection
mkdir -p $OVS/runs/selection/logs
export OV_ALLOW_TEST_PARENTS=1
for s in ${SEEDS:-0 1 2}; do
  nohup nice -n 10 $PY run_selection_mgpu.py --task pick_egg --pool sel_test_parents --train-seed $s \
    --max-workers ${MAXW:-18} --total-workers ${TOTALW:-52} --servers ${SERVERS:-2} "$@" \
    >> $OVS/runs/selection/logs/test_pick_egg_s$s.log 2>&1 &
  echo "seed $s supervisor pid $!"
  sleep 5
done
