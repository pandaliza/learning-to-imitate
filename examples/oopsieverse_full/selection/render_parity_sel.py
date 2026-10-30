"""Selection-study render parity (Amendment 6): the SAME procedure as matrix/render_parity.py (baseline
verify_obs_alignment.main, unchanged; reference = official playback seg frames via PlaybackDataset; PASS = step path
pixel-identical on every row, all 3 views x 2 frames), only the output goes to
outputs/oopsieverse_full_study/selection/verification/render_parity_<label>.json.
  MUJOCO_EGL_DEVICE_ID=<g> CUDA_VISIBLE_DEVICES= [CTXFIX=1] python render_parity_sel.py egl_gpu<g>[_ctxfix]_run1
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(os.path.dirname(HERE), "matrix")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
import verify_obs_alignment as V  # noqa: E402
# I/O plumbing only: verify_obs_alignment writes a scratch HDF5 of external links at one fixed path; several parity
# processes in parallel would truncate each other's file, so each process uses its own copy of that link file
# (same external links -> same demo data and reference frames).
import h5py  # noqa: E402
_VF = "/edrive1/lizad/oopsieverse_full_study/data/playback/_verify_obs.hdf5"
_VF_MINE = f"/edrive1/lizad/oopsieverse_full_study/data/playback/_verify_obs_pid{os.getpid()}.hdf5"
_File = h5py.File
h5py.File = lambda name, *a, **k: _File(_VF_MINE if str(name) == _VF else name, *a, **k)
if os.environ.get("CTXFIX") == "1":
    sys.path.insert(0, MATRIX)
    from ctxfix import install_ctx_free
    _mk = V.R.make_env
    V.R.make_env = lambda *a, **k: install_ctx_free(_mk(*a, **k))

label = sys.argv[1]
run_dir = sys.argv[2] if len(sys.argv) > 2 else "/edrive1/lizad/oopsieverse_full_study/runs/baseline/train/amix_pick_egg_v1_s0"
t0 = time.time()
rows = V.main(run_dir, False)
step = [r for r in rows if r["path"] == "step"]
state = [r for r in rows if r["path"] == "state"]
out = {"label": label, "procedure": "matrix/render_parity.py (identical), output relocated", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
       "wall_s": time.time() - t0,
       "env": {k: os.environ.get(k) for k in ("CTXFIX", "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID", "CUDA_VISIBLE_DEVICES", "PYOPENGL_PLATFORM",
                                               "__EGL_VENDOR_LIBRARY_FILENAMES", "LIBGL_ALWAYS_SOFTWARE", "EGL_PLATFORM", "GALLIUM_DRIVER")},
       "n_rows": len(rows),
       "step_min_seg_equal_frac": min(min(r["seg_equal_frac"].values()) for r in step),
       "state_min_seg_equal_frac": min(min(r["seg_equal_frac"].values()) for r in state),
       "step_max_action_absdiff": max(r["action_maxabs_eval_vs_dataset_input"] for r in step),
       "load_avg": os.getloadavg(), "rows": rows}
maps = open(f"/proc/{os.getpid()}/maps").read()
out["libs_loaded"] = {k: (k in maps) for k in ("libEGL_nvidia", "libnvidia-eglcore", "libEGL_mesa", "swrast", "libgallium")}
out["pass"] = out["step_min_seg_equal_frac"] == 1.0
p = f"/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/selection/verification/render_parity_{label}.json"
json.dump(out, open(p, "w"), indent=1)
print("PARITY", label, out["pass"], out["step_min_seg_equal_frac"], out["state_min_seg_equal_frac"], round(out["wall_s"], 1), flush=True)
try:
    os.remove(_VF_MINE)
except OSError:
    pass
os._exit(0)
