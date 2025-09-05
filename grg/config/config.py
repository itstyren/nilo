import argparse
from typing import Dict


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp_name",
        type=str,
        default="check",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=10e6,
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n_eval_rollout_threads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="Group_Repu_Gossip",
    )
    parser.add_argument(
        "--substrate_name",
        type=str,
        choices=[
            "prisoners_dilemma_in_the_matrix__arena",
            "commons_harvest__open",
            "matrix_game__group",
            "coin_game__grid",
        ],
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="Test",
    )
    parser.add_argument(
        "--env_dim",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--dilemma_T",
        type=lambda x: round(float(x), 2),
        default=5,
    )
    parser.add_argument(
        "--dilemma_S",
        type=lambda x: round(float(x), 2),
        default=1,
    )
    parser.add_argument(
        "--initial_ratio",
        nargs="+",
        type=float,
        default=[0.5, 0.5],
    )
    parser.add_argument(
        "--episode_length", type=int, default=200
    )
    parser.add_argument(
        "--group_num", type=int, default=2
    )
    parser.add_argument(
        "--network_type",
        type=str,
        choices=["lattice", "mixed"],
        default="mixed",
    )
    parser.add_argument(
        "--num_recipients",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--in_group_prob",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--norm_type",
        choices=['SJ', 'IS','SH','SS', 'RL', 'random', 'attention'],
        default='RL',
    )


    parser.add_argument(
        "--grid_size",
        type=int,
        default=3,
    )


    parser.add_argument(
        "--algorithm_name",
        choices=["PPO", 'DEV'],
        default="PPO",
    )
    parser.add_argument(
        "--dilemma_n_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--repu_n_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5e8,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--d_mini_batch",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--r_mini_batch",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dilemma_train_freq",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--initial_dilemma_train_freq",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--dilemma_train_delay",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--reach_fraction",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--attention_weights",
        action="store_true",
        default=False, 
    )


    parser.add_argument(
        "--dilemma_ent_coef",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--repu_ent_coef",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--dilemma_next_weight",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--dilemma_ent_coef_fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--dilemma_ent_coef_initial_eps",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--dilemma_ent_coef_final_eps",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--repu_ent_coef_fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--repu_ent_coef_initial_eps",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--repu_ent_coef_final_eps",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--dynamic_training",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--repu_rewards_type",
        type=str,
        choices=["SJ", "IS", 'plain', 'category', 'None'],
        default="plain",
    )
    parser.add_argument(
        "--repu_value_flag",
        type=str,
        choices=["reward", "return",],
        default="reward",
    )
    parser.add_argument(
        "--repu_baseline_weight",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--baseline_type",
        type=str,
        choices=["None",'NL','NoRepu'],
        default="None",
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="Anonymous",
    )


    parser.add_argument("--save_interval", type=int, default=1e8)


    parser.add_argument(
        "--use_render",
        action="store_true",
        default=False,
    )
    parser.add_argument("--render_episodes", type=int, default=5)
    parser.add_argument("--save_gifs", action='store_true', default=False)
    parser.add_argument("--ifi", type=float, default=0.1)


    parser.add_argument(
        "--dilemma_lr", type=float, default=5e-4)
    parser.add_argument(
        "--repu_lr", type=float, default=5e-4)
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_linear_update",
        action="store_true",
        default=False,
    )


    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
    )


    parser.add_argument(
        "--repu_net_arch",
        nargs="+",
        type=int,
        default=dict(embed=[16], query=[16], key=[16],
                     latent=[16], value=[16], vf=[16, 16]),
    )
    parser.add_argument(
        "--dilemma_net_arch",
        nargs="+",
        type=int,
        default=dict(pi=[16, 16], vf=[16, 16]),
    )


    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--render_interval",
        type=int,
        default=1,
    )

    return parser


def update_config(parsed_args) -> Dict:
    import os

    current_directory = os.getcwd()
    print("Current Directory:", current_directory)
    print(
        "========  all config   ======== \n {} \n ==============================".format(
            parsed_args
        )
    )
    return parsed_args