"""Rendering parity (Amendment 1): run the baseline evaluator's pixel-parity check (baseline/verify_obs_alignment.main,
unchanged) under a given rendering backend and write the result to matrix/verification/render_parity_<label>.json
instead of the baseline record. Reference frames = official playback seg (known-good EGL, GPUs 6/7) through
PlaybackDataset. PASS = step path pixel-identical (all 3 views x 2 frames, every row), as in the baseline check.
  python render_parity.py <label> [run_dir]      (backend chosen by environment variables, recorded)
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
import verify_obs_alignment as V  # noqa: E402
if os.environ.get("CTXFIX") == "1":  # same reset wrapper as the evaluator
    sys.path.insert(0, HERE)
    from ctxfix import install_ctx_free
    _mk = V.R.make_env
    V.R.make_env = lambda *a, **k: install_ctx_free(_mk(*a, **k))

label = sys.argv[1]
run_dir = sys.argv[2] if len(sys.argv) > 2 else "/edrive1/lizad/oopsieverse_full_study/runs/baseline/train/amix_pick_egg_v1_s0"
t0 = time.time()
rows = V.main(run_dir, False)
step = [r for r in rows if r["path"] == "step"]
state = [r for r in rows if r["path"] == "state"]
out = {"label": label, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": time.time() - t0,
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
p = f"/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/matrix/verification/render_parity_{label}.json"
json.dump(out, open(p, "w"), indent=1)
print("PARITY", label, out["pass"], out["step_min_seg_equal_frac"], out["state_min_seg_equal_frac"], round(out["wall_s"], 1), flush=True)
os._exit(0)
