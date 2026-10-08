# Shared env for the oops spatial-robustness jobs (mirrors slurm-scripts/train/pi05/eval_pi05_*.sbatch).
set -euo pipefail
cd "${REPO_DIR:-/home/ldahiya/max_vla/much-ado-about-noising}"
mkdir -p logs
source .venv/bin/activate
export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JAX_COMPILATION_CACHE_DIR=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/jax_cache_eval
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT
OOPS=${OOPS:-/data/user_data/ldahiya/oops_runs}
MANIFEST=$OOPS/manifest_pv1
EPISODES=$OOPS/episodes_pv1
