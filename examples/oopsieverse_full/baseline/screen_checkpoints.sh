#!/bin/bash
# Competence screen (frozen reset_banks_v1 eval_dev, first 30 in listed order, rollout seed 0) on every
# saved checkpoint of a run as it appears: step_5000 ... step_30000 (= final).
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
RUN=$1; NAME=$(basename $RUN); W=${W:-30}
B=/scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline
O=/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/baseline
for N in ${STEPS:-5000 10000 15000 20000 25000 30000}; do
  CK=step_$N.pth
  until [ -f $RUN/ckpt/$CK ]; do sleep 60; done
  sleep 20
  OUT=$OVS/runs/baseline/eval/${NAME}_${N}_dev30_r0
  $PY $B/rollout_eval.py --run-dir $RUN --ckpt $CK --resets $O/eval_dev_pick_egg_v1.json --limit 30 \
     --rollout-seeds 0 --workers $W --egl-devices 6 --torch-device cuda:6 --no-camera-obs --out-dir $OUT > $OUT.log 2>&1
  $PY $B/summarize_screen.py $OUT $O/screen_${NAME}_step${N}_dev30_r0.json "${NAME} step $N, eval_dev[0:30], rollout seed 0" > /dev/null
  echo "$N $(date) $(python3 -c "import json;d=json.load(open('$O/screen_${NAME}_step${N}_dev30_r0.json'));print(d['task_success'],d['primary_safe_success'],d['outcomes'])")"
done
