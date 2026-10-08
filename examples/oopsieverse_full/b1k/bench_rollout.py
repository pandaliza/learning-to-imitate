"""Policy-rollout throughput probe for a B1K OopsieVerse task (no policy; small random actions).

Builds the env like scripts/teleop_b1k.py (OGDamageableEnvironment from TaskConfig, no cameras),
resets with the task's reset() (pkl init state + noise) and steps with the damage evaluators on.
Reports env-build time, reset time, env.step rate, task_completion_check cost and health readout.
Usage: python bench_rollout.py TASK OUT_JSON [N_RESETS] [N_STEPS]
"""
import json
import os
import sys
import time

OOPSIE_ROOT = "/edrive1/lizad/oopsieverse_full_study/src/oopsieverse"
os.chdir(OOPSIE_ROOT)
sys.path.insert(0, OOPSIE_ROOT)
sys.path.insert(0, os.path.join(OOPSIE_ROOT, "scripts"))
import torch as th  # noqa: E402
import playback_b1k as pb  # noqa: E402
import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402
from damagesim.omnigibson.damageable_env import OGDamageableEnvironment  # noqa: E402

task, out_path = sys.argv[1:3]
n_resets = int(sys.argv[3]) if len(sys.argv) > 3 else 3
n_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 150
task_mod = pb.load_task_module(task)
cfg = task_mod.get_task_config()
gm.USE_GPU_DYNAMICS = cfg.use_gpu_dynamics
gm.ENABLE_TRANSITION_RULES = cfg.enable_transition_rules
scene_config = dict(cfg.scene_config)
scene_config.setdefault("type", "InteractiveTraversableScene")
rc = dict(cfg.robot_config)
rc["obs_modalities"] = ["proprio"]
env_config = {
    "env": {"action_frequency": cfg.action_frequency, "rendering_frequency": cfg.rendering_frequency,
            "physics_frequency": cfg.physics_frequency},
    "scene": scene_config, "robots": [rc],
    "objects": [dict(o) for o in cfg.task_objects.values()],
    "task": {"type": "DummyTask", "activity_name": cfg.task_name},
}
out = {"task": task, "freqs": env_config["env"]}
t0 = time.time()
env = OGDamageableEnvironment(configs=env_config)
out["t_env_build_s"] = time.time() - t0


def do_reset():
    env.reset()
    if hasattr(task_mod, "reset"):
        task_mod.reset(env)
    env._reset_damage_tracking()


reset_t = []
for _ in range(n_resets):
    t = time.time()
    do_reset()
    reset_t.append(time.time() - t)
out["reset_s"] = reset_t
robot = env.robots[0]
adim = robot.action_dim
out["action_dim"] = adim
zero = th.zeros(adim)
# Hold: IK controller with absolute? use the robot's current command for a no-op baseline
step_t, check_t, healths, succ = [], [], [], []
g = th.Generator().manual_seed(0)
for i in range(n_steps):
    a = zero.clone()
    a[: adim - 1] = 0.01 * th.randn(adim - 1, generator=g)
    a[-1] = 1.0
    t = time.time()
    obs, r, te, tr, info = env.step(a)
    step_t.append(time.time() - t)
    t = time.time()
    try:
        succ.append(bool(task_mod.task_completion_check(env)))
    except Exception as e:  # noqa: BLE001
        succ.append(f"err:{e}"[:100])
    check_t.append(time.time() - t)
    h = obs.get("health") if isinstance(obs, dict) else None
    if h is not None:
        healths.append(float(th.as_tensor(h).min()))
st = sorted(step_t)
out.update(
    n_steps=n_steps,
    step_median_s=st[len(st) // 2],
    steps_per_s=n_steps / sum(step_t),
    task_check_median_s=sorted(check_t)[len(check_t) // 2],
    min_health_seen=min(healths) if healths else None,
    health_readout_ok=bool(healths),
    any_success=any(s is True for s in succ),
    check_errors=[s for s in succ if isinstance(s, str)][:3],
)
with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
print("BENCH " + json.dumps(out), flush=True)
og.shutdown()
