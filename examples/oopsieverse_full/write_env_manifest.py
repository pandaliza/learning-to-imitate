"""Write outputs/oopsieverse_full_study/env/env_manifest.json (run inside env/oopsie)."""

import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

from common import OUT, SRC, STORAGE, HF_SNAPSHOT


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as e:
        return f"ERR {e}"


def du(p):
    return sh(f"du -sh --apparent-size {p} 2>/dev/null | cut -f1") + " (apparent), " + sh(f"du -sh {p} 2>/dev/null | cut -f1") + " (on disk)"


def main():
    import mujoco, numpy, scipy, numba, torch, h5py, robosuite, robocasa  # noqa: F401
    m = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
         "python": sys.version.split()[0], "python_executable": sys.executable, "platform": platform.platform(),
         "venv": os.path.join(STORAGE, "env/oopsie"),
         "install_procedure": [
             "uv venv env/oopsie --python 3.10 (install.py:12 PYTHON_VERSION=3.10; conda replaced by uv venv)",
             "robocasa/setup.py numba pin relaxed ==0.61.2 -> >=0.61.2 (install.py:79-87 patch_robocasa_numba_pin)",
             "uv pip install -e robosuite ; uv pip install -e robocasa ; uv pip install -e oopsieverse (install.py:167-170, docs/install.md:45; WITH deps, no --no-deps, as the official procedure does)",
             "python robocasa/scripts/download_kitchen_assets.py (all types, answered y) ; python robocasa/scripts/setup_macros.py (install.py:172-173)",
             "torch came in as a dependency of lerobot==0.3.3 (robocasa setup.py); docs/troubleshooting.md only gives torch==2.9.1 as an example pin for driver mismatch, not used"],
         "packages": {k: getattr(sys.modules[k], "__version__", None) for k in ["mujoco", "numpy", "scipy", "numba", "torch", "h5py", "robosuite", "robocasa"]},
         "torch_cuda": {"build": torch.version.cuda, "available": torch.cuda.is_available()},
         "lockfile": "outputs/oopsieverse_full_study/env/uv_lock.txt",
         "sources": {}, "assets": {}, "gpu": {}, "disk": {}, "data": {}}
    for r in ["oopsieverse", "robosuite", "robocasa"]:
        p = os.path.join(SRC, r)
        m["sources"][r] = {"path": p, "sha": sh(f"git -C {p} rev-parse HEAD"), "describe": sh(f"git -C {p} describe --tags --always"),
                           "commit_date": sh(f"git -C {p} log -1 --format=%ci"), "remote": sh(f"git -C {p} remote get-url origin"),
                           "local_modifications": sh(f"git -C {p} status --porcelain --untracked-files=no") or "none"}
    m["sources"]["pins_verified_from"] = "oopsieverse install.py:162-169 (robosuite aaa8b9b, robocasa 97a4060); oopsieverse has no tags/releases, HEAD a739f99 is the latest commit (2026-09-04)"
    m["sources"]["oopsieverse_externals"] = "src/oopsieverse/externals/{robosuite,robocasa} are symlinks to src/{robosuite,robocasa} (the layout install.py creates)"
    ad = os.path.join(SRC, "robocasa/robocasa/models/assets")
    bl = json.load(open(os.path.join(ad, "box_links/box_links_assets.json")))
    m["assets"] = {"path": ad, "size": du(ad), "box_links_assets.json_sha256": hashlib.sha256(open(os.path.join(ad, "box_links/box_links_assets.json"), "rb").read()).hexdigest(),
                   "box_links": bl, "revision": "robocasa @ 97a4060 box_links_assets.json (Box shared links have no version id; the json hash pins the link set)",
                   "subdirs": {d: du(os.path.join(ad, d)) for d in sorted(os.listdir(ad)) if os.path.isdir(os.path.join(ad, d))},
                   "macros_private": os.path.exists(os.path.join(SRC, "robocasa/robocasa/macros_private.py"))}
    m["gpu"] = {"nvidia_smi": sh("nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader"),
                "cuda_driver_version": sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"),
                "egl_rendering": "MUJOCO_GL=egl with __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d (the shell default points at miniconda's Mesa-only vendor dir). EGL frames are corrupt on GPU 0 and 2 and correct on GPUs 1,3-7; use render_gpu_device_id=3 (or MUJOCO_EGL_DEVICE_ID)."}
    m["disk"] = {"storage_total": du(STORAGE),
                 "env": du(os.path.join(STORAGE, "env")), "src_incl_assets": du(SRC), "cache": du(os.path.join(STORAGE, "cache")),
                 "runs": du(os.path.join(STORAGE, "runs")),
                 "df": sh("df -h /edrive1 / /scr")}
    m["data"] = {"hf_snapshot": HF_SNAPSHOT, "files_sha256": "see outputs/oopsieverse_full_study/replay/replay_sample.json"}
    os.makedirs(os.path.join(OUT, "env"), exist_ok=True)
    with open(os.path.join(OUT, "env", "env_manifest.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps({k: m[k] for k in ["packages", "torch_cuda", "disk"]}, indent=1))


if __name__ == "__main__":
    main()
