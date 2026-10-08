"""Run a single GPU-3 rendering test process under the shared budget: start only if a slot is granted, kill it
(and its children) at once if the eval daemon becomes ACTIVE or our total exceeds the cap. Logs to <out>.guard."""
import json, os, signal, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu3_budget as GB

log = open(sys.argv[1] + ".guard", "a")
cmd = sys.argv[2:]
d = GB.assess(set())
if d["granted_workers"] < 1 or d["daemon"]["active"]:
    log.write(json.dumps({"event": "not_started", **d}) + "\n"); sys.exit(3)
p = subprocess.Popen(cmd, start_new_session=True)
while p.poll() is None:
    time.sleep(10)
    pids = {p.pid} | GB._children(p.pid)
    d = GB.assess(pids)
    log.write(json.dumps({k: d[k] for k in ("time", "ours_mib", "daemon_mib", "free_mib", "over_cap")} | {"active": d["daemon"]["active"]}) + "\n"); log.flush()
    if d["over_cap"] or d["daemon"]["active"]:
        os.killpg(p.pid, signal.SIGKILL)
        log.write(json.dumps({"event": "killed_for_daemon", **d}) + "\n")
        sys.exit(4)
sys.exit(p.returncode)
