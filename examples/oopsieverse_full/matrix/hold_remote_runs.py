"""Reserve runs assigned to another host (Amendment 5: wipe_countertop seeds 1-2 train on genos).

The queue runners on this host treat a run whose <run>/LOCK is flocked as running elsewhere and never launch it.
This process takes and holds those locks (waiting for any local holder to exit first). Kill it to release them.

  nohup python hold_remote_runs.py > runs/matrix/_queue/hold_remote_runs.log 2>&1 &
"""

import fcntl
import json
import os
import time

RUNS = "/edrive1/lizad/oopsieverse_full_study/runs/matrix"
ASSIGN = os.path.join(RUNS, "_queue", "remote_assignment.json")
REMOTE = [f"{a}_wipe_countertop_v1_s{s}" for s in (1, 2)
          for a in ("amix", "asafe", "imean", "ipath", "ipathg", "auxpathg", "ssmix", "cmean")]


def main():
    json.dump({"host": "genos", "amendment": 5, "runs": REMOTE, "holder_pid": os.getpid(), "time": time.time()},
              open(ASSIGN, "w"), indent=1)
    held = {}
    while True:
        for r in REMOTE:
            if r in held:
                continue
            os.makedirs(os.path.join(RUNS, r), exist_ok=True)
            f = open(os.path.join(RUNS, r, "LOCK"), "a")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held[r] = f
                print("held", r, time.strftime("%c"), flush=True)
            except OSError:
                f.close()
        time.sleep(2 if len(held) < len(REMOTE) else 60)


if __name__ == "__main__":
    main()
