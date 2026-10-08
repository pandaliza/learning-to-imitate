#!/bin/bash
# Rollout throughput / parallel scaling benchmark (untrained step_0 policy, 120-step capped episodes, one
# episode per worker). Labelled benchmark-only via OV_TIMEOUT_OVERRIDE (recorded in each record).
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
RUN=${RUN:-$OVS/runs/baseline/train/smoke_amix_pick_egg_draft_s0_omp1_aborted}
CK=${CK:-step_0.pth}
for W in ${WS:-4 16 32}; do
  OUT=$OVS/runs/baseline/eval/bench_scaling_W${W}
  rm -rf $OUT
  OV_TIMEOUT_OVERRIDE=120 $PY /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/rollout_eval.py \
    --run-dir $RUN --ckpt $CK --resets /scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/baseline/draft_dev_resets_pick_egg.json \
    --limit $W --workers $W --egl-devices 6 --torch-device cuda:6 --no-camera-obs --out-dir $OUT > $OUT.log 2>&1
  echo "W=$W $(tail -1 $OUT.log) load=$(cut -d' ' -f1 /proc/loadavg)"
done
