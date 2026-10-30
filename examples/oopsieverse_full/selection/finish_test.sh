#!/bin/bash
# Run after all three test supervisors report finished (status json "finished": true).
source /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/baseline/envsetup.sh
cd /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/selection
O=/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/selection
for s in 0 1 2; do
  $PY analyze_selection.py --root $OVS/runs/selection/pick_egg/sel_test_parents/train_seed_$s --task pick_egg \
    --pool sel_test_parents --train-seed $s --out $O/analysis_test_seed$s.json > /dev/null
done
$PY aggregate_selection.py --inputs $O/analysis_test_seed0.json $O/analysis_test_seed1.json $O/analysis_test_seed2.json \
  --out $O/analysis_test_aggregate.json
