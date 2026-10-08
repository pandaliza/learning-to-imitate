#!/bin/bash
# Phase D dev branching (pick_egg, sel_dev_parents, training seed 0). Resumable: rerun this script to resume.
# Stop gracefully: touch /edrive1/lizad/oopsieverse_full_study/runs/selection/pick_egg/sel_dev_parents/train_seed_0/STOP
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
cd /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/selection
mkdir -p $OVS/runs/selection/logs
nohup $PY run_selection.py --task pick_egg --pool sel_dev_parents --train-seed 0 --max-workers 8 "$@" \
  >> $OVS/runs/selection/logs/dev_pick_egg_s0.log 2>&1 &
echo $!
