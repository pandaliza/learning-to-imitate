#!/bin/bash
#SBATCH --job-name=watch-m10
#SBATCH --output=logs/watch_m10_%j.out
#SBATCH --error=logs/watch_m10_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=72:00:00

# CPU-only watcher for the M10 A-row (docs/co_prediction.md section 6): polls the two training OUT
# dirs; when a 4k/8k/12k checkpoint lands (and model.safetensors has been stable >5 min), submits
# the matching SR evals -- decoupled ckpt x {s1,s2,s3} (A3/A2/A4), tied ckpt x s2 (A1) -- and
# archives the 12k weights (the M9 79% checkpoint was lost to the keep-last-3 pruner; not again).
set -uo pipefail
cd /home/ldahiya/max_vla/much-ado-about-noising
mkdir -p logs logs/m10_watch

B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
declare -A RUNS=( [copred]=$B/ldahiya_checkpoints/pi05_m10_copred [tied]=$B/ldahiya_checkpoints/pi05_m10_tied )
declare -A SCHEDS=( [copred]="s1 s2 s3" [tied]="s2" )
STEPS="4000 8000 12000"
MARK=logs/m10_watch

stable() {  # model.safetensors exists and unmodified for >5 min (not mid-write)
  local f="$1/model.safetensors"
  [ -f "$f" ] && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 300 ]
}

while true; do
  alldone=1
  for run in copred tied; do
    for step in $STEPS; do
      d="${RUNS[$run]}/$step"
      for sc in ${SCHEDS[$run]}; do
        m="$MARK/eval_${run}_${step}_${sc}.submitted"
        if [ ! -f "$m" ]; then
          alldone=0
          if stable "$d"; then
            CKPT="$d" SCHEDULE="$sc" OUT="logs/eval_m10_${run}_${sc}_${step}.json" \
              sbatch slurm-scripts/eval/eval_m10_copred.sbatch && touch "$m" \
              && echo "[watch] submitted eval ${run}/${step}/${sc}"
          fi
        fi
      done
    done
    # archive 12k the moment it is stable (pruner kills it at 18k)
    am="$MARK/archive_${run}_12000.done"
    if [ ! -f "$am" ]; then
      alldone=0
      if stable "${RUNS[$run]}/12000"; then
        mkdir -p "${RUNS[$run]}_archive"
        cp -r "${RUNS[$run]}/12000" "${RUNS[$run]}_archive/12000" && touch "$am" \
          && echo "[watch] archived ${run}/12000"
      fi
    fi
  done
  [ "$alldone" = 1 ] && { echo "[watch] all evals submitted + 12k archived; exiting"; exit 0; }
  sleep 900
done
