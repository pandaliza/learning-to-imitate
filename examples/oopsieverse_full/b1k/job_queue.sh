#!/bin/bash
# Pinned job queue: each line of $1 = "TASK COND GPU IDS TAG EXTRA..." (IDS quoted with commas, e.g. 0,1,2).
# Runs at most MAXP Isaac processes in total (counting ones started elsewhere); each job pinned to a free 8-core set.
Q=$1; MAXP=${MAXP:-4}
S=/scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/b1k/run_replay.sh
read -ra SETS <<< "${CPU_SETS:-96-103 104-111 112-119 120-127}"
nproc_isaac() { ps -eo args | grep "^python .*replay_b1k.py --task\|^python .*bench_rollout.py" | grep -vc "^$"; }
while read -r TASK COND GPU IDS TAG EXTRA; do
  [ -z "$TASK" ] && continue
  CPUS=""
  while true; do
    L=$(cut -d' ' -f2 /proc/loadavg | cut -d. -f1); lim=$MAXP; [ -n "$LOWLOAD_MAX" ] && [ "$L" -lt 100 ] && lim=$LOWLOAD_MAX  # 5-min load avg; relaxation opt-in
    if [ "$(nproc_isaac)" -lt "$lim" ]; then
      used=$(for p in $(pgrep -f "^python .*replay_b1k.py|^python .*bench_rollout.py"); do taskset -cp $p 2>/dev/null | awk '{print $NF}'; done)
      for s in "${SETS[@]}"; do echo "$used" | grep -qx "$s" || { CPUS=$s; break; }; done
      break
    fi
    sleep 30
  done
  [ -z "$CPUS" ] && { echo "no free cpu set"; exit 1; }
  echo "$(date +%T) start $TAG cpus=$CPUS gpu=$GPU"
  TASKSET=$CPUS nohup $S $TASK $COND $GPU "${IDS//,/ }" $TAG "$EXTRA" >/dev/null 2>&1 &
  sleep 90
done < $Q
wait
echo "$(date +%T) queue done"
