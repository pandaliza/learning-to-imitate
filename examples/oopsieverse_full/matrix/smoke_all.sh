#!/bin/bash
# Verification 4: 500-step smoke training of every arm (pick_egg; plus amix/ipathg on wipe) and a 5-reset
# development rollout each (frozen eval_dev pool, contract protocol, native stochastic sampling, rollout seed 0).
# Memory rule: launch only on GPU 5/6/7 with >= 25 GB free; wait (never fail) otherwise. Sequential.
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/matrix
R=$OVS/runs/matrix_smoke; mkdir -p $R/logs $R/eval
pick_gpu() {
  while true; do
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$1>=5 && $1<=7 && $2>=25600 {print $2, $1}' | sort -rn | head -1 | awk '{print $2}')
    [ -n "$best" ] && { echo $best; return; }
    sleep 60
  done
}
for spec in ${SPECS:-pick_egg:amix pick_egg:asafe pick_egg:ssmix pick_egg:imean pick_egg:ipath pick_egg:ipathg pick_egg:cmean pick_egg:auxpathg wipe_countertop:amix wipe_countertop:ipathg}; do
  IFS=: read task arm <<< "$spec"
  id=${arm}_${task}_v1_s0__smoke500
  if [ ! -f $R/$id/DONE.json ]; then
    g=$(pick_gpu); echo "$(date) train $id gpu $g"
    CUDA_VISIBLE_DEVICES=$g $PY $M/train_matrix.py --arm $arm --task $task --seed 0 --num-steps 500 --save-every 500 \
       --label smoke --workers 4 --act-ckpt --runs-root $R > $R/logs/$id.train.log 2>&1 || echo "TRAIN_FAIL $id"
  fi
  if [ -f $R/$id/DONE.json ] && [ $(ls $R/eval/$id/*__r0.json 2>/dev/null | wc -l) -lt 5 ]; then
    g=$(pick_gpu); echo "$(date) eval $id gpu $g"
    CUDA_VISIBLE_DEVICES=$g $PY $M/rollout_eval_matrix.py --run-dir $R/$id --task $task --resets dev --limit 5 \
       --rollout-seeds 0 --out-dir $R/eval/$id --workers 5 --egl-devices $g --torch-device cuda:0 > $R/logs/$id.eval.log 2>&1 || echo "EVAL_FAIL $id"
  fi
done
echo SMOKE_ALL_DONE
