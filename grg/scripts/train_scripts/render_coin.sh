#!/bin/bash
project_name='Group_Repu_Gossip'
scenario='local_coin_game'
substrate_name='coin_game__grid'
algo='DEV'
exp="q_plus_next"

# agent number
env_dim=2
grid_size=3

norm_type='RL'
# norm_type='attention'
repu_rewards_type='plain'
#repu_rewards_type='IS'
repu_value_flag='reward'
# repu_value_flag='return'

# CPU Cores
n_rollout_threads=1 # batch size is n_steps * n_env where n_env is number of environment copies running in parallel
total_timesteps=5100000


# Custom seed list
seeds=(6)

n_steps=$((64 / ${n_rollout_threads}))
# n_steps: The number of steps to run for each environment per update


for seed in "${seeds[@]}"; do
    echo "Running seed ${seed}"
    CUDA_VISIBLE_DEVICES=1 python ../render/render_coins.py  --cuda --seed ${seed}  --log_interval 5 --project_name ${project_name} --substrate_name ${substrate_name} --n_rollout_threads ${n_rollout_threads} --algorithm_name ${algo} --total_timesteps ${total_timesteps} --exp_name ${exp} --scenario_name ${scenario} --env_dim ${env_dim} --grid_size ${grid_size} --repu_rewards_type ${repu_rewards_type} --repu_value_flag ${repu_value_flag} --norm_type ${norm_type}\
    --episode_length 32  --dilemma_lr 0.005 --n_steps ${n_steps}   \
     --save_interval 50 --use_render  --render_episodes 4 --ifi 1  \
     --model_dir '../../results/Group_Repu_Gossip/0513_coin_game_1_step_return_save/coin_game__grid/DEV/wandb/run-20250513_082456-630chjr5/files/' \
    # --user_name 'tyren' --use_wandb --use_linear_lr_decay  \

# --user_name 'tyren' --use_wandb --use_render --n_training_threads 4 --lr 5e-4 --use_linear_lr_decay --use_linear_update --use_linear_update
done
