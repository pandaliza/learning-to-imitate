# Source before any baseline job. All caches/tmp on /edrive1.
export OVS=/edrive1/lizad/oopsieverse_full_study
export PY=$OVS/env/oopsie/bin/python
export TMPDIR=$OVS/cache/tmp
export HF_HOME=$OVS/cache/hf
export XDG_CACHE_HOME=$OVS/cache/xdg
export TORCH_HOME=$OVS/cache/torch
export PIP_CACHE_DIR=$OVS/cache/pip
export UV_CACHE_DIR=$OVS/cache/uv
export MPLCONFIGDIR=$OVS/cache/mpl
export NUMBA_CACHE_DIR=$OVS/cache/numba
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export PYNPUT_BACKEND=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $TMPDIR $XDG_CACHE_HOME $TORCH_HOME $MPLCONFIGDIR $NUMBA_CACHE_DIR
