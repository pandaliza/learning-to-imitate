#!/bin/bash
# Regenerate every paper number/figure from the manifest + episode records (seconds; login node is fine).
#   PROVISIONAL=1 bash slurm-scripts/oops/oops_30_analyze.sh   # while rollouts are still landing
source slurm-scripts/oops/_env.sh
OUT=${OUT:-docs/oops_workshop/results}
python examples/openpi/oops/analyze_robustness.py --manifest-dir "$MANIFEST" --episodes-dir "$EPISODES" \
    --baseline-arm M0 --intent-arm ${INTENT_ARM:-M1} --out-dir "$OUT" ${PROVISIONAL:+--provisional}
cp "$MANIFEST"/{task_manifest.csv,manifest_meta.json} "$OUT"/
