#!/bin/bash
# usage: guarded_probe.sh TAG  (env: CUDA_VISIBLE_DEVICES?, B1K_KIT_ARGS, OMNIGIBSON_GPU_ID)
# Runs probe_isaac.py; SIGKILLs it at the first nvidia-smi sample showing its PID on any GPU other than the allowed GPU (ALLOWED_BUS).
source /edrive1/lizad/oopsieverse_full_study/env_b1k/b1k_env.sh
TAG=$1; OUT=/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/b1k/gpu_confinement; mkdir -p $OUT
GPU0_BUS="${ALLOWED_BUS:?set ALLOWED_BUS}"
cd $TMPDIR
PROBE_OUT=$OUT/${TAG}_probe.json taskset -c 104-115 python /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/b1k/probe_isaac.py > $OUT/${TAG}_kit.log 2>&1 &
P=$!
: > $OUT/${TAG}_nvsmi.log
verdict="confined"
while kill -0 $P 2>/dev/null; do
  rows=$(nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader 2>/dev/null | grep "^$P,")
  [ -n "$rows" ] && echo "$(date +%T.%N | cut -c1-12) $rows" | tr '\n' ' ' >> $OUT/${TAG}_nvsmi.log && echo >> $OUT/${TAG}_nvsmi.log
  if echo "$rows" | grep -v "$GPU0_BUS" | grep -q "^$P,"; then kill -9 $P; verdict="VIOLATION: $(echo "$rows" | grep -v "$GPU0_BUS" | tr '\n' ';')"; break; fi
  sleep 0.3
done
wait $P 2>/dev/null; rc=$?
echo "$TAG rc=$rc verdict=$verdict env: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset} OMNIGIBSON_GPU_ID=$OMNIGIBSON_GPU_ID B1K_KIT_ARGS='$B1K_KIT_ARGS'" | tee -a $OUT/summary.txt
