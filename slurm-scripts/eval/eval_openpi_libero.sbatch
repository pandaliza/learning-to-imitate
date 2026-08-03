#!/bin/bash
#SBATCH --job-name=openpi-libero
#SBATCH --output=logs/openpi_libero_%j.out
#SBATCH --error=logs/openpi_libero_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00

set -uo pipefail
mkdir -p logs
cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export HF_LEROBOT_HOME=/data/user_data/ldahiya/lerobot
export MUJOCO_GL=egl
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/numba_${SLURM_JOB_ID:-m}_XXXXXX)"
export MPLCONFIGDIR="$(mktemp -d /tmp/mpl_${SLURM_JOB_ID:-m}_XXXXXX)"
export XDG_RUNTIME_DIR="$(mktemp -d /tmp/xdg_${SLURM_JOB_ID:-m}_XXXXXX)"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# Reference eval of a STOCK openpi checkpoint using openpi's OWN client/server LIBERO harness
# (not our eval_libero_intent.py, which is convention-locked to our finetunes). The server loads
# the JAX checkpoint + its bundled norm stats and handles ALL PI input conventions (rotation,
# state encoding, action decode) itself, so no convention-matching on our side.
B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
CONFIG=${CONFIG:-pi05_libero}
CKPT=${CKPT:-$B/pi05_libero}
SUITE=${SUITE:-libero_goal}
TRIALS=${TRIALS:-20}
PORT=${PORT:-8000}

echo "[server] launching JAX policy server: config=$CONFIG dir=$CKPT port=$PORT"
python external/openpi/scripts/serve_policy.py --port $PORT --env LIBERO \
    policy:checkpoint --policy.config "$CONFIG" --policy.dir "$CKPT" \
    > logs/openpi_server_${SLURM_JOB_ID}.out 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"' EXIT

# wait for the server to bind the port (JAX load + compile can take a few minutes)
echo "[server] waiting for port $PORT ..."
for i in $(seq 1 60); do
  python -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" && break
  ! kill -0 $SERVER_PID 2>/dev/null && { echo "[server] DIED during startup:"; tail -20 logs/openpi_server_${SLURM_JOB_ID}.out; exit 1; }
  sleep 10
done
echo "[server] up. launching client on suite=$SUITE trials=$TRIALS"

python examples/openpi/_openpi_libero_client.py \
    --args.task-suite-name "$SUITE" \
    --args.num-trials-per-task $TRIALS \
    --args.host 127.0.0.1 --args.port $PORT

echo "=== openpi $CONFIG on $SUITE done ==="
