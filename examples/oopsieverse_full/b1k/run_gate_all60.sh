#!/bin/bash
# usage: run_gate_all60.sh <parity_gpu> <seed> [seed ...]   Dev gate (30 dev resets, rollout seed 0, step_30000.pth)
# for the all-60 add_firewood runs, through eval_b1k_all60.py. Launch scheme of b1k_results.md section 1:
# CUDA_VISIBLE_DEVICES hides GPU 0 and keeps the active GPU at its own index (valid for g=3,4,6 only).
# Waits until the GPU has >= 14 GiB free before each launch. Status: runs/b1k/all60/status/gate_<seed>.status
G=$1; shift
O=/edrive1/lizad/oopsieverse_full_study
B=/scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/b1k
case $G in 3) CVD=1,2,4,3;; 4) CVD=1,2,3,5,4;; 6) CVD=1,2,3,4,5,7,6;; *) echo "bad GPU $G"; exit 2;; esac
source $O/env_b1k/b1k_env.sh
for S in "$@"; do
  RUN=amix_add_firewood_all60_s$S
  RD=$O/runs/b1k/all60/train/$RUN
  KEY=$(sha256sum $RD/ckpt/step_30000.pth | cut -c1-16)
  OUT=$O/runs/b1k/all60/eval/$RUN/dev_gate__$KEY
  ST=$O/runs/b1k/all60/status/gate_$S.status
  mkdir -p $OUT
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G)
    [ "$FREE" -ge 14000 ] && break
    echo "$(date -Is) WAIT gpu=$G free=${FREE}MiB" >> $ST; sleep 120
  done
  echo "$(date -Is) START seed=$S gpu=$G cvd=$CVD out=$OUT" >> $ST
  cd $B
  CUDA_VISIBLE_DEVICES=$CVD OMNIGIBSON_GPU_ID=$G nice -n 10 /edrive1/lizad/oopsieverse_full_study/env_b1k/venv/bin/python -u eval_b1k_all60.py \
    --run-dir $RD --ckpt step_30000.pth --task add_firewood --pool dev --ids 0-29 --rollout-seed 0 --out $OUT \
    >> $O/runs/b1k/all60/logs/gate_$RUN.log 2>&1
  echo "$(date -Is) EXIT rc=$? seed=$S" >> $ST
done
echo "$(date -Is) QUEUE_DONE" >> $ST
