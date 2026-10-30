#!/bin/bash
# Run one single-env render job (build_pools / verify_alias) under selection/guard_gpu.py on the parity-passed GPU with
# the most free memory; on a headroom kill (exit 4) or no-slot (exit 3) wait and retry on the best GPU again.
#   guarded_single.sh <name> <python args...>
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
cd /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/wipe_v2
NAME=$1; shift
L=/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/logs
for i in $(seq 1 40); do
  G=$($PY -c "
import sys; sys.path.insert(0,'../selection'); import gpu_budget as GB
_, free = GB.gpu_table(); ok=[g for g in GB.parity_passed_gpus() if GB.single_slot_ok(g)[0]]
print(max(ok, key=lambda g: free[g]) if ok else -1)")
  if [ "$G" = "-1" ]; then echo "$(date) no GPU with headroom, waiting" >> $L/$NAME.log; sleep 120; continue; fi
  echo "$(date) attempt $i on GPU $G" >> $L/$NAME.log
  CUDA_VISIBLE_DEVICES="" MUJOCO_EGL_DEVICE_ID=$G EGL_DEVICE_ID=$G nice -n 10 $PY ../selection/guard_gpu.py $L/$NAME $G $PY "$@" >> $L/$NAME.log 2>&1
  rc=$?
  echo "$(date) exit $rc" >> $L/$NAME.log
  [ $rc -eq 0 ] && exit 0
  [ $rc -ne 3 ] && [ $rc -ne 4 ] && exit $rc
  sleep 60
done
