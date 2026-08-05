#!/bin/bash
#SBATCH --job-name=w3-square-mh-flow-intent
#SBATCH --output=logs/w3_square_mh_flow_intent_%j.out
#SBATCH --error=logs/w3_square_mh_flow_intent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs
source /home/ldahiya/max_vla/much-ado-about-noising/.venv/bin/activate
cd /home/ldahiya/max_vla/much-ado-about-noising
export HF_HOME=/data/user_data/ldahiya/hf_cache

# W3 step 0: Flow-intent BC (Config A) on square-mh-state.
# Architecture:
#   - Shared context encoder (MLP)
#   - Flow intent model: samples intent (7D EEF pos+quat) from noise via ODE (Ta=1)
#   - MLP action decoder: deterministic (obs_emb, intent) → action
#
# Trains encoder + intent_flow_map via flow loss (step 1),
# then action_decoder via MSE on GT intent (step 2), independently.
# Target SR: ~30-70% (same as baseline for fair comparison).

python examples/train/train_robomimic.py \
    --config-path /home/ldahiya/max_vla/much-ado-about-noising/examples/configs \
    task=square_mh_state_flow_intent \
    network=mlp_flow_intent \
    optimization.gradient_steps=300000 \
    optimization.batch_size=256 \
    optimization.loss_type=flow \
    log.eval_episodes=50 \
    log.eval_freq=10000 \
    log.save_freq=50000 \
    log.project=w3-derisk-square \
    log.group=square-mh-state-w3 \
    log.exp_name=flow_intent \
    task.num_envs=5
