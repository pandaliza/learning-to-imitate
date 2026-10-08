"""Stage-A probe: boot OmniGibson/Isaac Sim 4.5 headless on this host, load an empty
scene, step physics, and try one render. No BEHAVIOR dataset needed."""
import json, os, sys, time

# Extra Kit settings (e.g. GPU confinement) appended to sys.argv; SimulationApp forwards unknown args to Kit.
sys.argv += [a for a in os.environ.get("B1K_KIT_ARGS", "").split() if a]
t0 = time.time()
import omnigibson as og
from omnigibson.macros import gm

gm.HEADLESS = True
out = {"stage": "import", "t_import": time.time() - t0}
try:
    env = og.Environment(configs={"scene": {"type": "Scene"}})
    out["t_env"] = time.time() - t0
    t1 = time.time()
    for _ in range(300):
        og.sim.step()
    out["physics_steps_per_s"] = 300 / (time.time() - t1)
    t1 = time.time()
    for _ in range(30):
        og.sim.render()
    out["render_calls_per_s"] = 30 / (time.time() - t1)
    out["stage"] = "ok"
except Exception as e:  # report and exit
    import traceback
    out["error"] = traceback.format_exc()
print("PROBE_RESULT " + json.dumps(out), flush=True)
with open(os.environ.get("PROBE_OUT", "probe_result.json"), "w") as f:
    json.dump(out, f, indent=1)
og.shutdown()
