#!/usr/bin/env bash
# Stop each run of a DexMG sweep once it reaches STEP:
#   - STEP a multiple of log.save_freq=50000 (e.g. 100000, 250000): stop after model_latest.pt
#     at STEP is saved, so the run can resume from STEP later.
#   - otherwise (e.g. an eval step like 240000): stop after the eval logged at STEP finishes;
#     a later resume restarts from the last 50k checkpoint.
#   - REQUIRE_EVAL=1: always wait for the eval logged at STEP (use when STEP is both a checkpoint
#     and an eval step, e.g. 200000: the checkpoint is written just BEFORE that eval runs).
# Usage: scripts/dexmg_stop_at_step.sh TASK STEP [--dry-run]
#   TASK: dexmg task name, e.g. three_piece_assembly | drawer_cleanup | threading
# Detection: in the log segment after the last resume, a "model_latest saved" line must appear
# after the "[Step STEP-1]" line. Kills the main train process and its env/dataloader workers.
T=$1; STEP=$2; DRY=${3:-}
ROOT=/edrive4/ldahiya/mip_runs/dexmg_$T
# STOP_RUNS="run1 run2 ..." limits the watcher to a subset of runs (default: all nine).
RUNS=${STOP_RUNS:-"baseline_s0 baseline_s1 baseline_s2 hier_mean8_s0 hier_mean8_s1 hier_mean8_s2 hier_wp4_s0 hier_wp4_s1 hier_wp4_s2"}
MARK="[Step $((STEP-1))]"   # fixed string, matched with index() (no regex)
reached() {
  local f=$ROOT/$1/train.out; [ -f "$f" ] || return 1
  tac "$f" | sed '/Resuming training from step/q' | tac \
    | if [ $((STEP % 50000)) -eq 0 ] && [ -z "${REQUIRE_EVAL:-}" ]; then
        awk -v m="$MARK" 'index($0, m) {seen=1} seen && /model_latest saved/{ok=1} END{exit !ok}'
      else
        awk -v m="$MARK mean_step_9" 'index($0, m) {ok=1} END{exit !ok}'
      fi
}
pids_of() { pgrep -f "train_robomimic[.]py task=dexmg_${T}_image.* log.exp_name=$1( |$)"; }
stop_run() {
  local p; p=$(pids_of $1); [ -z "$p" ] && return
  echo "$(date '+%F %T') stopping $1 ($STEP checkpoint present) pids: $(echo $p | tr '\n' ' ')"
  kill $p 2>/dev/null; sleep 20
  p=$(pids_of $1); [ -n "$p" ] && { echo "  force-killing $1"; kill -9 $p 2>/dev/null; }
}
while true; do
  remaining=""
  for r in $RUNS; do
    [ -z "$(pids_of $r)" ] && continue
    if reached $r; then
      if [ -n "$DRY" ]; then echo "dry-run: would stop $r"; else sleep 30; stop_run $r; fi
    else remaining="$remaining $r"; fi
  done
  echo "$(date '+%F %T') still training (below $STEP):$remaining"
  [ -z "$remaining" ] && { echo "$(date '+%F %T') all $T runs stopped"; break; }
  [ -n "$DRY" ] && exit 0
  sleep 300
done
