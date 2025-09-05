#!/bin/bash
project_name='Group_Repu_Gossip'
scenario='act_flag'
substrate_name='matrix_game__group'
algo='DEV'
exp="local_test"

# agent number
env_dim=3
network_type="mixed"
group_num=1
num_recipients=1
in_group_prob=1
norm_type='random'
repu_rewards_type='plain'
repu_value_flag='reward'
baseline_type='NL'


# CPU Cores
n_rollout_threads=8 
total_timesteps=5100000

# Set initial, final, and interval values for dilemma_S and dilemma_T
initial_S=-0.10
final_S=-0.10
interval_S=0.02

initial_T=1.10
final_T=1.10
interval_T=0.02

# Custom seed list
seeds=(6)

n_steps=$((64 / ${n_rollout_threads}))


# Loop through dilemma_S_values and dilemma_T_values
dilemma_S=$initial_S
while (( $(echo "$dilemma_S <= $final_S" | bc -l) )); do
  dilemma_T=$initial_T
  while (( $(echo "$dilemma_T <= $final_T" | bc -l) )); do
    for seed in "${seeds[@]}"; do
        echo "Running seed ${seed}"
        CUDA_VISIBLE_DEVICES=1 python ../train/train_matrix.py  --cuda --seed ${seed}  --log_interval 30 --project_name ${project_name} --substrate_name ${substrate_name} --n_rollout_threads ${n_rollout_threads} --algorithm_name ${algo} --total_timesteps ${total_timesteps} --exp_name ${exp} --scenario_name ${scenario} --env_dim ${env_dim} --network_type ${network_type} --group_num ${group_num} --in_group_prob ${in_group_prob} --num_recipients ${num_recipients} --dilemma_S ${dilemma_S} --dilemma_T ${dilemma_T} --repu_rewards_type ${repu_rewards_type} --repu_value_flag ${repu_value_flag} --norm_type ${norm_type} --baseline_type ${baseline_type}\
        --episode_length 16  --dilemma_lr 0.001 --n_steps ${n_steps}  --d_mini_batch 16 --r_mini_batch 64 \
        --dilemma_train_freq 1.0  --dilemma_n_epochs 4 --repu_n_epochs 1 --dilemma_next_weight 0.0   \
        --dilemma_ent_coef_final_eps 0.05  --dilemma_ent_coef_fraction 1.0 --dilemma_ent_coef_initial_eps 0.05 \
        --repu_ent_coef_final_eps 0.05  --repu_ent_coef_fraction 0.1 --repu_ent_coef_initial_eps 0.05 \
        --repu_lr 0.001 --repu_baseline_weight 0.0  \

    done
    dilemma_T=$(echo "$dilemma_T + $interval_T" | bc)
  done
  dilemma_S=$(echo "$dilemma_S + $interval_S" | bc)
done