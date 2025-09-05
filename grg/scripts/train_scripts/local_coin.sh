#!/bin/bash
project_name='Group_Repu_Gossip'
scenario='local_coin_game'
substrate_name='coin_game__grid'
algo='DEV'
exp="q_plus_next"

# agent number
env_dim=2
grid_size=3

# norm_type='RL'
norm_type='attention'
repu_rewards_type='plain'
repu_value_flag='reward'
baseline_type='None'

# CPU Cores
n_rollout_threads=8 # batch size is n_steps * n_env where n_env is number of environment copies running in parallel
total_timesteps=5100000


# Custom seed list
seeds=(1)

n_steps=$((64 / ${n_rollout_threads}))
# n_steps: The number of steps to run for each environment per update


for seed in "${seeds[@]}"; do
    echo "Running seed ${seed}"
    CUDA_VISIBLE_DEVICES=1 python ../train/train_coins.py  --cuda --seed ${seed}  --log_interval 5 --project_name ${project_name} --substrate_name ${substrate_name} --n_rollout_threads ${n_rollout_threads} --algorithm_name ${algo} --total_timesteps ${total_timesteps} --exp_name ${exp} --scenario_name ${scenario} --env_dim ${env_dim} --grid_size ${grid_size} --repu_rewards_type ${repu_rewards_type} --repu_value_flag ${repu_value_flag} --norm_type ${norm_type} --baseline_type ${baseline_type}\
    --episode_length 32  --dilemma_lr 0.005 --n_steps ${n_steps}  --d_mini_batch 16 --r_mini_batch 16 \
    --dilemma_train_freq 1.0  --dilemma_n_epochs 4 --repu_n_epochs 4 --dilemma_next_weight 1.0   \
    --dilemma_ent_coef_final_eps 0.01  --dilemma_ent_coef_fraction 1.0 --dilemma_ent_coef_initial_eps 0.01 \
    --repu_ent_coef_final_eps 0.01  --repu_ent_coef_fraction 0.1 --repu_ent_coef_initial_eps 0.01 \
    --repu_lr 0.005 --repu_baseline_weight 1.0  \
    --dynamic_training  --initial_dilemma_train_freq 0.50 --dilemma_train_delay 0.0 --reach_fraction 0.8  \
    # --save_interval 500 \
    # --use_render --render_episodes 1 --ifi 1  --render_interval 500 --save_gif  --n_eval_rollout_threads 1\
    # --user_name 'tyren' --use_wandb --use_linear_lr_decay  \



# --user_name 'tyren' --use_wandb --use_render --n_training_threads 4 --lr 5e-4 --use_linear_lr_decay --use_linear_update --use_linear_update
done
