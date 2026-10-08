#!/bin/bash
# usage: run_replay.sh TASK COND GPU "DEMO_IDS" TAG
source /edrive1/lizad/oopsieverse_full_study/env_b1k/b1k_env.sh
TASK=$1; COND=$2; GPU=$3; IDS=$4; TAG=$5; EXTRA=$6
D=$B1K_ROOT/data/oopsieverse-demos/paper_experiments/teleop_data/$TASK/$COND.hdf5
R=$B1K_ROOT/runs/b1k_replay/$TAG; mkdir -p $R
O=/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/b1k/replay; mkdir -p $O
export OMNIGIBSON_GPU_ID=$GPU
cd $TMPDIR
/usr/bin/time -v ${TASKSET:+taskset -c $TASKSET} python /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/b1k/replay_b1k.py \
  --task_name $TASK --source_hdf5_path $D --playback_hdf5_path $R/playback.hdf5 --demo_ids $IDS \
  --metrics_json $O/${TAG}_timing.json $EXTRA > $R/log.txt 2>&1
echo "EXIT $? $TAG" >> $R/log.txt
