"""Run one verification process that renders on GPU <g> (Amendment 6 courtesy rules): start only if GPU g keeps
>= 8 GB free with one more EGL context (strict window), kill it (process group) if free < 8 GB. Logs to <out>.guard.
  python guard_gpu.py <out> <g> cmd ...
"""
import json, os, signal, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_budget as GB

out, g, cmd = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
assert g not in GB.NEVER_RENDER
log = open(out + ".guard", "a")
ok, info = GB.single_slot_ok(g)
if not ok:
    log.write(json.dumps({"event": "not_started", "gpu": g, **info}) + "\n"); sys.exit(3)
p = subprocess.Popen(cmd, start_new_session=True)
log.write(json.dumps({"event": "started", "gpu": g, "pid": p.pid, **info}) + "\n"); log.flush()
while p.poll() is None:
    time.sleep(10)
    try:
        _, free = GB.gpu_table()
    except Exception as e:
        log.write(json.dumps({"event": "smi_error", "e": repr(e)}) + "\n"); continue
    log.write(json.dumps({"time": time.strftime("%H:%M:%S"), "gpu": g, "free_mib": free[g]}) + "\n"); log.flush()
    if free[g] < GB.HEADROOM_MIB:
        os.killpg(p.pid, signal.SIGKILL)
        log.write(json.dumps({"event": "killed_headroom", "gpu": g, "free_mib": free[g]}) + "\n")
        sys.exit(4)
sys.exit(p.returncode)
