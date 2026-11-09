#!/usr/bin/env bash
# Stop each DexMG Drawer Cleanup run as soon as it has written its 100k model_latest.pt
# (so it can be resumed from 100k later). Usage:
#   scripts/dexmg_drawer_stop_at_100k.sh [--dry-run]
# Detection: in the log segment after the last resume, a "model_latest saved" line must appear
# after the "[Step 99999]" line. Kills the main train process and its env/dataloader workers.
ROOT=/edrive4/ldahiya/mip_runs/dexmg_drawer_cleanup
RUNS="baseline_s0 baseline_s1 baseline_s2 hier_mean8_s0 hier_mean8_s1 hier_mean8_s2 hier_wp4_s0 hier_wp4_s1 hier_wp4_s2"
DRY=${1:-}
past100k() {  # $1 = run name
  local f=$ROOT/$1/train.out
  [ -f "$f" ] || return 1
  tac "$f" | sed '/Resuming training from step/q' | tac \
    | awk '/\[Step 99999\]/{seen=1} seen && /model_latest saved/{ok=1} END{exit !ok}'
}
pids_of() { pgrep -f "train_robomimic[.]py task=dexmg_drawer_cleanup_image.* log.exp_name=$1( |$)"; }
stop_run() {  # $1 = run name
  local p; p=$(pids_of $1); [ -z "$p" ] && return
  echo "$(date '+%F %T') stopping $1 (100k checkpoint present) pids: $(echo $p | tr '\n' ' ')"
  kill $p 2>/dev/null; sleep 20
  p=$(pids_of $1); [ -n "$p" ] && { echo "  force-killing $1"; kill -9 $p 2>/dev/null; }
}
while true; do
  remaining=""
  for r in $RUNS; do
    [ -z "$(pids_of $r)" ] && continue            # already gone
    if past100k $r; then
      if [ -n "$DRY" ]; then echo "dry-run: would stop $r"; else sleep 30; stop_run $r; fi
    else
      remaining="$remaining $r"
    fi
  done
  echo "$(date '+%F %T') still training (below 100k):$remaining"
  [ -z "$remaining" ] && { echo "$(date '+%F %T') all drawer runs stopped"; break; }
  [ -n "$DRY" ] && exit 0
  sleep 300
done
