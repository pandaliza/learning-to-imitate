#!/bin/bash
#SBATCH --job-name=oops-parity
#SBATCH --output=logs/oops_parity_%j.out
#SBATCH --error=logs/oops_parity_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
# Native-evaluator parity on two unperturbed resets per arm (spec section 4). ARM=M0|M1|M3.
# Native: eval_libero_intent.py on suite task 1 (bowl on stove), episodes 0-1 = benchmark resets 0-1.
source slurm-scripts/oops/_env.sh
ARM=${ARM:?set ARM=M0|M1|M3}
P=$OOPS/parity/$ARM
case $ARM in
  M0) NATIVE="--config-name pi05_base_nointent --checkpoint-dir $(python -c 'import sys;sys.path.insert(0,"examples/openpi/oops");from eval_spatial_shifts import PRESETS as P;print(P["M0"]["checkpoint_dir"])')";;
  M1) NATIVE="--config-name pi05_base_intent --checkpoint-dir $(python -c 'import sys;sys.path.insert(0,"examples/openpi/oops");from eval_spatial_shifts import PRESETS as P;print(P["M1"]["checkpoint_dir"])') --intent --intent-task-config libero_goal_suite_image_slot_intent --intent-ckpt $(python -c 'import sys;sys.path.insert(0,"examples/openpi/oops");from eval_spatial_shifts import PRESETS as P;print(P["M1"]["intent_ckpt"])')";;
  M3) B=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi
      NATIVE="--config-name pi05_base_intent_m3 --checkpoint-dir $B/ldahiya_checkpoints/pi05_base_intent_m3/base_intent_m3_goal_run1/29999 --vl-cotrain --intent-task-config libero_goal_suite_image_slot_intent_vl --intent-stack $B/ldahiya_checkpoints/slot_intent_vl/intent_stack_32000.pt --frozen-base-weights $B/pi05_base_pytorch --frozen-base-config pi05_base_nointent";;
esac
python examples/openpi/eval_libero_intent.py $NATIVE --task-suite libero_goal --tasks 1 --num-trials-per-task 2 \
    --episode-seeding 0 --dump-actions "$P" --out "$P/native_sr.json"
python examples/openpi/oops/eval_spatial_shifts.py --preset $ARM --manifest-dir "$MANIFEST" \
    --out-dir "$P/episodes" --tasks put_the_bowl_on_the_stove --reset-ids 0,1 --deltas 0.0 \
    --policy-seeds 0 --dump-actions "$P"
python examples/openpi/oops/parity_check.py --dump-dir "$P" --arm $ARM --out "$P/parity.json"
