#!/bin/bash
# Wipe diagnosis rollouts: GPU 0 inference, GPU 3 EGL, 8 workers, nice 10. Dev resets / demo states only.
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
unset OV_TIMEOUT_OVERRIDE
cd /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/wipe_diag
O=/edrive1/lizad/oopsieverse_full_study/runs/wipe_diag
TRAIN=with/demo_10,with/demo_11,with/demo_12,with/demo_13,with/demo_14,without/demo_0,without/demo_1,without/demo_2,without/demo_3,without/demo_4
DEV=with/demo_0,with/demo_1,with/demo_2,with/demo_3,with/demo_4,with/demo_20,with/demo_21,with/demo_22,with/demo_23,with/demo_24
for s in 0 1; do
  R=/edrive1/lizad/oopsieverse_full_study/runs/matrix/amix_wipe_countertop_v1_s$s
  for c in bank_alias bank; do
    echo "=== $(date -u) amix s$s $c"
    nice -n 10 $PY diag_rollout.py --run-dir $R --cond $c --n 30 --workers 8 --out-dir $O/amix_s$s/$c --frames 2 2>&1 | grep -v -i "warning\|INFO\|mimicgen"
  done
  echo "=== $(date -u) amix s$s demo"
  nice -n 10 $PY diag_rollout.py --run-dir $R --cond demo --demos $TRAIN,$DEV --workers 8 --out-dir $O/amix_s$s/demo --frames 3 2>&1 | grep -v -i "warning\|INFO\|mimicgen"
done
echo "=== ALL_DONE $(date -u)"
