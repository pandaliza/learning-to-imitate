#!/bin/bash
# usage: guarded_probe_threshold.sh TAG  (env: ALLOWED_BUS regex, B1K_KIT_ARGS, OMNIGIBSON_GPU_ID; CUDA_VISIBLE_DEVICES unset)
# Runs probe_isaac.py. Isaac opens small CUDA probe contexts on every GPU at startup (user-approved 2026-10-22).
# Allowed GPUs (ALLOWED_BUS): unrestricted. Any other GPU: SIGKILL as soon as our usage there exceeds MAX_OTHER_MIB.
# Every sample is logged so the lifetime of the small contexts on other GPUs can be reported.
source /edrive1/lizad/oopsieverse_full_study/env_b1k/b1k_env.sh
TAG=$1; OUT=/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/b1k/gpu_confinement; mkdir -p $OUT
ALLOWED="${ALLOWED_BUS:?set ALLOWED_BUS}"
MAX_OTHER_MIB=${MAX_OTHER_MIB:-100}
cd $TMPDIR
PROBE_OUT=$OUT/${TAG}_probe.json taskset -c 104-115 python /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/b1k/probe_isaac.py > $OUT/${TAG}_kit.log 2>&1 &
P=$!
: > $OUT/${TAG}_nvsmi.log
verdict="confined (other GPUs <= ${MAX_OTHER_MIB} MiB)"
while kill -0 $P 2>/dev/null; do
  rows=$(nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader,nounits 2>/dev/null | grep "^$P,")
  [ -n "$rows" ] && echo "$(date +%T.%N | cut -c1-12) $(echo "$rows" | tr '\n' ';')" >> $OUT/${TAG}_nvsmi.log
  over=$(echo "$rows" | grep "^$P," | grep -v "$ALLOWED" | awk -F', ' -v m=$MAX_OTHER_MIB '$3+0 > m')
  if [ -n "$over" ]; then kill -9 $P; verdict="VIOLATION (>${MAX_OTHER_MIB} MiB off allowed GPUs): $(echo "$over" | tr '\n' ';')"; break; fi
  sleep 0.3
done
wait $P 2>/dev/null; rc=$?
echo "$TAG rc=$rc verdict=$verdict env: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset} OMNIGIBSON_GPU_ID=$OMNIGIBSON_GPU_ID B1K_KIT_ARGS='$B1K_KIT_ARGS'" | tee -a $OUT/summary.txt
