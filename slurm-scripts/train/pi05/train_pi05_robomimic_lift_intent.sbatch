#!/bin/bash
#SBATCH --job-name=pi05-robomimic-lift-intent
#SBATCH --output=logs/pi05_robomimic_lift_intent_%j.out
#SBATCH --error=logs/pi05_robomimic_lift_intent_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00

mkdir -p logs

cd /home/ldahiya/max_vla/much-ado-about-noising
source .venv/bin/activate

export NUMBA_CACHE_DIR=/tmp
export XDG_RUNTIME_DIR=/tmp
export HF_LEROBOT_HOME=/data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi

python external/openpi/scripts/train.py \
    pi05_robomimic_lift_intent \
    --exp_name run1 \
    --checkpoint_base_dir /data/group_data/maxlab/common_datasets/pandaliza/maxvla/openpi/checkpoints
