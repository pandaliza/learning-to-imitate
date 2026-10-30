#!/bin/bash
# Arms the W2/W3 pool evaluations: as soon as <run>/DONE.json and ckpt/step_30000.pth exist, launch eval_wipe_v2.py
# on P-train + P-devlayout x rollout seeds 0,1 (--max-workers 32), once per run. Log: runs/wipe_v2/logs/eval_chain.log
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
cd /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/wipe_v2
V=/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2
left="$*"
while [ -n "$left" ]; do
  rem=""
  for a in $left; do
    R=$V/train/${a}_wipe_countertop_v2_s0
    if [ -f $R/DONE.json ] && [ -f $R/ckpt/step_30000.pth ] && [ -z "$(find $R/ckpt/step_30000.pth -newermt '-60 sec')" ]; then
      nohup nice -n 10 $PY eval_wipe_v2.py --run-dir $R --pools P-train,P-devlayout --rollout-seeds 0,1 --ckpt step_30000.pth \
        --max-workers 32 --label ${a}_wipe_countertop_v2_s0 > $V/logs/${a}_wipe_countertop_v2_s0.eval.log 2>&1 &
      echo "$(date -Is) launched eval $a pid $!"
    else
      rem="$rem $a"
    fi
  done
  left=$(echo $rem)
  [ -n "$left" ] && sleep 60
done
echo "$(date -Is) chain finished"
