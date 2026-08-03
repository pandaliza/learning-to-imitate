#!/bin/bash
#SBATCH --job-name=w3-square-mh-baseline
#SBATCH --output=logs/w3_square_mh_baseline_%j.out
#SBATCH --error=logs/w3_square_mh_baseline_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising

# W3 step 0: Plain flow BC baseline on square-mh-state.
# Target SR: ~30-70% (difficulty-matched for control ladder).
# Rollouts are ~1000x cheaper than RoboCasa; evaluate at 30-70% for fair comparison.

python examples/train/train_robomimic.py \
    task=square_mh_state \
    network=mlp \
    optimization.gradient_steps=300000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=10000 \
    log.save_freq=50000 \
    log.project=w3-derisk-square \
    log.group=square-mh-state-w3 \
    log.exp_name=baseline \
    task.num_envs=5
