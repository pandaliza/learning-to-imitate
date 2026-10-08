"""GPU-3 EGL memory of one evaluator env over 6 canonical resets, with (CTXFIX=1) or without the reset wrapper."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
import rollout_eval as RE
from rollout_eval_matrix import bank_resets
os.chdir(RE.OOPSIE)
def mem():
    for line in subprocess.check_output(["nvidia-smi"]).decode().splitlines():
        p = line.strip("| ").split()
        if len(p) >= 7 and p[0] == "3" and p[3] == str(os.getpid()):
            return float(p[-1].rstrip("MiB"))
    return 0.0
env = RE.make_env("pick_egg", use_camera_obs=False)
if os.environ.get("CTXFIX") == "1":
    from ctxfix import install_ctx_free; install_ctx_free(env)
out = []
for r in bank_resets("pick_egg", "eval_dev")[:6]:
    RE.apply_reset(env, r)
    for _ in range(5):
        env.step(env.action_spec[0] * 0)
    out.append(mem())
print("MEM", os.environ.get("CTXFIX"), out, flush=True)
os._exit(0)
