#!/usr/bin/env bash
# Launch one official DexMimicGen BC-RNN run (robomimic dexmimicgen branch) in the background.
# Usage: scripts/dexmg_bcrnn_run.sh TASK ARM SEED GPU [extra train_bcrnn.py args, e.g. --debug]
#   TASK: threading | drawer_cleanup | three_piece_assembly   ARM: baseline | wp4 | hand | hand_img   SEED: 201 | 202 | 203
#   hand = rendered-hand latent intent (examples/dexmg_hand_intent/), replaces wp4 as the sole intent.
#   hand_img = same target, predicted from the image-backbone features (intent loss trains the image encoders).
# Drawer Cleanup always gets --match_dataset_scene (see mip/envs/robomimic/robomimic_env.py).
set -euo pipefail
T=$1; ARM=$2; SEED=$3; GPU=$4; shift 4
REPO=$(cd "$(dirname "$0")/.." && pwd)
CFG_DIR=${BCRNN_CFG_DIR:-/edrive4/ldahiya/dexmg_datasets/bcrnn_configs}
CFG=$CFG_DIR/bc_rnn_image_ds_two_arm_${T}_D0_seed_${SEED}.json
[ -f "$CFG" ] || { echo "missing config $CFG" >&2; exit 1; }
ROOT=${BCRNN_RUN_ROOT:-/edrive4/ldahiya/mip_runs/dexmg_bcrnn}
RUN=${T}_${ARM}_s${SEED}; DIR=$ROOT/$RUN; mkdir -p "$DIR"
EXTRA=()
case $ARM in
  baseline) ;;
  wp4) EXTRA+=(--intent /edrive4/ldahiya/dexmg_datasets/intent_sidecar/two_arm_${T}_wp4.hdf5);;
  hand) EXTRA+=(--intent /edrive4/ldahiya/dexmg_datasets/intent_sidecar/two_arm_${T}_hand.hdf5 --intent_key hand_intent);;
  hand_img) EXTRA+=(--intent /edrive4/ldahiya/dexmg_datasets/intent_sidecar/two_arm_${T}_hand.hdf5 --intent_key hand_intent --intent_input image);;
  *) echo "unknown arm: $ARM" >&2; exit 1;;
esac
[ "$T" = drawer_cleanup ] && EXTRA+=(--match_dataset_scene)
export PYTHONPATH=$REPO/external/robomimic_dexmg${PYTHONPATH:+:$PYTHONPATH}
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$DIR"
setsid nohup "$REPO/.venv/bin/python" "$REPO/examples/dexmg_bcrnn/train_bcrnn.py" --config "$CFG" \
  --output_dir "$DIR" --seed "$SEED" "${EXTRA[@]}" "$@" >> "$DIR/train.out" 2>&1 &
echo "$RUN  pid $!  gpu $GPU  -> $DIR/train.out"
