"""Diagnose stored-vs-runtime serialized state layout for a B1K paper file.

Builds the env exactly like the official playback (create_from_hdf5, no cameras), then:
  * runtime dump_state size vs stored state_size,
  * per-object serialized sizes at runtime (registry order),
  * walks stored state[0] with the runtime per-object sizes and reports the first object whose
    root_link position decoded from the stored vector is implausible / differs from scene_file,
  * whether the size fallback in _load_state_with_size_fallback is exercised and with which size,
  * the AG block decoded for the robot from the loaded state.
Usage: python diag_state_layout.py TASK HDF5 OUT_JSON
"""
import json
import os
import sys

OOPSIE_ROOT = "/edrive1/lizad/oopsieverse_full_study/src/oopsieverse"
os.chdir(OOPSIE_ROOT)
sys.path.insert(0, OOPSIE_ROOT)
sys.path.insert(0, os.path.join(OOPSIE_ROOT, "scripts"))
import h5py  # noqa: E402
import torch as th  # noqa: E402
import playback_b1k as pb  # noqa: E402
import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402
from damagesim.omnigibson.damageable_env import OGDamageableDataPlaybackWrapper as W  # noqa: E402
from damagesim.utils.playback_schema import playback_proprio_keys  # noqa: E402

task, path, out_path = sys.argv[1:4]
task_cfg = pb.load_task_config(task)
gm.USE_GPU_DYNAMICS = task_cfg.use_gpu_dynamics
gm.ENABLE_TRANSITION_RULES = False
env = W.create_from_hdf5(
    input_path=path, output_path=os.path.join(os.environ["TMPDIR"], f"diag_{task}.hdf5"),
    robot_obs_modalities=["proprio"], robot_proprio_keys=playback_proprio_keys(arm="0"),
    robot_sensor_config=None, external_sensors_config=[], exclude_sensor_names=task_cfg.exclude_sensor_names,
    n_render_iterations=1, only_successes=False, activity_name=task,
)
out = {"task": task}
f = h5py.File(path, "r")
g = f["data/demo_0"]
stored = th.as_tensor(g["state"][0][: int(g["state_size"][0])], dtype=th.float32)
out["stored_size"] = int(stored.numel())

env.reset()
env.scene.restore(env.scene_file, update_initial_file=True)
env.objects = env.scene.objects.copy()
env.env.initialize_damageable_objects()
env.env._reset_damage_tracking()
if not og.sim.is_playing():
    og.sim.play()
for _ in range(5):
    og.sim.step()
cur = og.sim.dump_state(serialized=True)
out["runtime_size"] = int(cur.numel())

scene = og.sim.scenes[0]
reg = scene._registry
objreg = scene.object_registry
per_obj = []
for obj in objreg.objects:
    s = obj.dump_state(serialized=True)
    per_obj.append((obj.name, type(obj).__name__, int(s.numel())))
out["n_objects_runtime"] = len(per_obj)
# registry order used by serialize
order = [o.name for o in sorted(objreg.objects, key=lambda o: o.name)] if False else None
out["per_object_sizes_runtime_first10"] = per_obj[:10]
robot = env.robots[0]
out["robot_runtime_size"] = int(robot.dump_state(serialized=True).numel())
out["robot_ag_block_size"] = int(robot._ag_serialized_state_size(robot.default_arm))
sys_size = int(scene.system_registry.serialize(scene.system_registry.dump_state(serialized=False)).numel()) if hasattr(scene, "system_registry") else None
out["system_registry_runtime_size"] = sys_size

# try the official fallback and record which size works
ok_size = None
err = None
for try_size in [out["stored_size"]] + list(range(out["stored_size"] - 1, max(0, out["stored_size"] - 10), -1)):
    try:
        og.sim.load_state(stored[:try_size], serialized=True)
        ok_size = try_size
        break
    except AssertionError as e:
        err = str(e)[:400]
out["fallback_size_that_loaded"] = ok_size
out["first_load_error"] = err
try:
    d, n = og.sim.deserialize(stored)
    out["deserialize_consumed"] = int(n)
except Exception as e:  # noqa: BLE001
    out["deserialize_error"] = repr(e)[:400]
# Locate the layout offset: dump right after loading stored state[0] (no step) and align.
try:
    after = og.sim.dump_state(serialized=True).float()
    n = min(after.numel(), stored.numel())
    diff = (after[:n] - stored[:n]).abs()
    first = int(th.nonzero(diff > 1e-3)[0]) if bool((diff > 1e-3).any()) else None
    out["post_load_dump_size"] = int(after.numel())
    out["first_mismatch_index"] = first
    # map index -> object via runtime per-object sizes in registry order (7 scene pos/ori first)
    cum = 7 + 1  # scene pos/ori + system_registry count (size 1 at runtime)
    owner = None
    sizes = []
    for obj in scene.object_registry.objects:
        sz = int(obj.dump_state(serialized=True).numel())
        sizes.append((obj.name, cum, sz))
        if first is not None and owner is None and cum <= first < cum + sz:
            owner = (obj.name, first - cum, sz)
        cum += sz
    out["registry_layout_total"] = cum
    out["mismatch_owner_obj_offset_size"] = owner
    out["last_objects_in_layout"] = sizes[-4:]
    if first is not None:
        out["stored_around_first"] = [round(float(x), 4) for x in stored[max(0, first - 3): first + 18]]
        out["runtime_around_first"] = [round(float(x), 4) for x in after[max(0, first - 3): first + 18]]
    out["n_tail_mismatch"] = int((diff > 1e-3).sum())
    ri = [i for i, (nm, _, _) in enumerate(sizes) if nm == robot.name]
    if ri:
        nm, st0, sz = sizes[ri[0]]
        out["robot_layout_start_size"] = [st0, sz]
        out["robot_block_stored"] = [round(float(x), 4) for x in stored[st0: st0 + sz]]
        out["robot_block_runtime"] = [round(float(x), 4) for x in after[st0: st0 + sz]]
except Exception as e:  # noqa: BLE001
    out["align_error"] = repr(e)[:300]
# Robot AG decode on the loaded state
try:
    out["robot_ag_after_load"] = {k: str(v)[:200] for k, v in robot._ag_obj_constraint_params.items()}
except Exception as e:  # noqa: BLE001
    out["robot_ag_after_load_error"] = repr(e)[:200]
# object pose sanity: compare to scene_file init poses after load
sf = json.loads(f["data"].attrs["scene_file"])["state"]["registry"]["object_registry"]
bad = []
for obj in objreg.objects:
    if obj.name in sf and "root_link" in sf[obj.name]:
        p0 = th.tensor(sf[obj.name]["root_link"]["pos"])
        p1 = obj.get_position_orientation()[0]
        dist = float(th.linalg.norm(p0 - p1))
        if dist > 0.5:
            bad.append((obj.name, round(dist, 3)))
out["objects_moved_gt_0.5m_vs_scene_file_after_loading_state0"] = sorted(bad, key=lambda x: -x[1])[:20]
out["n_objects_moved"] = len(bad)
with open(out_path, "w") as fo:
    json.dump(out, fo, indent=1)
print("DIAG " + json.dumps(out), flush=True)
og.shutdown()
