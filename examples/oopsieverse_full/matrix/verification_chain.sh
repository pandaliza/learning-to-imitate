#!/bin/bash
# Armed chain (2026-10-06): finish verification items 2 and 4 when GPU memory allows (coordinator memory rules:
# >= 25 GB free before any launch, wait otherwise), then apply the predeclared gate (check_gate.py). Only if it
# passes: register the existing A-MIX pick_egg s0 run and start the resumable training queue.
# Stop at any time: touch /edrive1/lizad/oopsieverse_full_study/runs/matrix/_queue/STOP
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/matrix
LOG=$OVS/runs/matrix_dev/logs
S=$OVS/runs/matrix_smoke
Q=$OVS/runs/matrix/_queue; mkdir -p $Q
stopped() { [ -f $Q/STOP ]; }
pick_gpu() {
  while true; do
    stopped && { echo "STOP flag"; exit 0; }
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$1>=5 && $1<=7 && $2>=25600 {print $2, $1}' | sort -rn | head -1 | awk '{print $2}')
    [ -n "$best" ] && { echo $best; return; }
    sleep 60
  done
}
echo "$(date) chain start"
until grep -q SMOKE_ALL_DONE $LOG/smoke_all.log 2>/dev/null; do stopped && exit 0; sleep 60; done
# second pass covers any spec skipped by a lock race
$M/smoke_all.sh >> $LOG/smoke_all.log 2>&1
echo "$(date) smoke complete"
g=$(pick_gpu); echo "$(date) intervention on gpu $g"
CUDA_VISIBLE_DEVICES=$g $PY $M/test_intervention.py final.pth $S/imean_pick_egg_v1_s0__smoke500 $S/ipath_pick_egg_v1_s0__smoke500 \
   $S/ipathg_pick_egg_v1_s0__smoke500 $S/cmean_pick_egg_v1_s0__smoke500 $S/auxpathg_pick_egg_v1_s0__smoke500 > $LOG/intervention.log 2>&1
if $PY $M/check_gate.py > $LOG/gate.log 2>&1; then
  echo "$(date) GATE PASS"
  cd $M && CUDA_VISIBLE_DEVICES="" $PY train_matrix.py --arm amix --task pick_egg --seed 0 --label final \
     --adopt-existing $OVS/runs/baseline/train/amix_pick_egg_v1_s0 >> $LOG/adopt.log 2>&1
  stopped && exit 0
  nohup $PY $M/queue_runner.py --slots 5:3,6:3,7:3 --max-jobs 9 --workers 4 >> $Q/runner.log 2>&1 &
  echo "$(date) queue runner started pid $!"
else
  echo "$(date) GATE FAIL: queue NOT started; see verification/gate.json"
fi
