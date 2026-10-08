#!/bin/bash
# Official playback (gripper + headless + path remap) for all pick_egg and wipe demos, one process per demo.
# Usage: bash run_playback_all.sh [NPROC]
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
NPROC=${1:-24}
D=$OVS/data/oopsieverse-demos/paper_experiments/teleop_data
OUT=$OVS/data/playback/parts
LOG=$OVS/runs/baseline/playback_logs
mkdir -p $OUT $LOG
GPUS=(6 7)   # EGL-correct devices (0,2 corrupt); at most 2 GPUs per user rule
jobs=()
k=0
for task in pick_egg wipe_countertop; do
  env=pick_egg; [ $task = wipe_countertop ] && env=wipe_counter
  for cond in with without; do
    for i in $(seq 0 29); do
      jobs+=("$task $env $cond $i ${GPUS[$((k % 2))]}"); k=$((k+1))
    done
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P $NPROC -L 1 bash -c '
task=$0; env=$1; cond=$2; i=$3; gpu=$4
o='$OUT'/${task}__${cond}__demo_${i}.hdf5
[ -f $o.done ] && exit 0
s=$(date +%s.%N)
MUJOCO_EGL_DEVICE_ID=$gpu '$PY' /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/playback_official.py --playback \
  --input '$D'/$task/${cond}_live_feedback.hdf5 --output $o.tmp --env $env --demo_ids $i > '$LOG'/${task}__${cond}__demo_${i}.log 2>&1
e=$(date +%s.%N)
if grep -q "Playback complete" '$LOG'/${task}__${cond}__demo_${i}.log; then mv $o.tmp $o; echo "$s $e" > $o.done; fi
'
echo ALL_DONE
