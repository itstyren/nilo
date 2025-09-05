import os
import sys
import time
import wandb
from tensorboardX import SummaryWriter
from stable_baselines3.common.type_aliases import (
    MaybeCallback,
    TrainFreq,
    TrainFrequencyUnit,
)
import torch
import numpy as np
from typing import Optional
import re
from grg.utils import tools as utils
from stable_baselines3.common.utils import (
    set_random_seed,
)
from gymnasium import spaces
from grg.utils.sepereated_buffer import CustomRolloutBuffer, DictRolloutBuffer
from torchinfo import summary
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from collections import deque
from typing import Any, ClassVar, Optional, TypeVar, Union
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape


class BaseRunner(object):
    """
    Base class for all runners
    """

    def __init__(self, config):
        # pass the configuration
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_env=config["eval_env"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        if self.envs is not None:
            self.observation_space = self.envs.observation_space
            self.action_space = self.envs.action_space
            self.max_dilemma_action_value=self.action_space['dilemma'].n-1
            self.n_envs = self.envs.num_envs

        # setting from cli arguments
        self._total_timesteps = self.all_args.total_timesteps  # the total train steps
        self.episode_length = self.all_args.episode_length
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.algorithm_name = self.all_args.algorithm_name
        self.exp_name = self.all_args.exp_name
        self.substrate_name = self.all_args.substrate_name
        self.n_steps = self.all_args.n_steps
        self.d_minibatch_size = self.all_args.d_mini_batch
        self.r_minibatch_size = self.all_args.r_mini_batch
        self.num_recipients = None
        

        # interval
        self._save_interval = self.all_args.save_interval
        self._log_interval = self.all_args.log_interval
        self._render_interval = self.all_args.render_interval


        self.ep_info_buffer = None  # type: Optional[deque]
        self.dilemma_train_info_buffer = None  # type: Optional[deque]
        self.repu_train_info_buffer = None  # type: Optional[deque]

        self._num_timesteps_at_start = 0
        self.num_timesteps = 0
        self.start_time = 0.0
        self.repu_train_times = 0

        self._last_repu_obs = None
        self._last_dilemma_obs = None
        self._last_episode_starts = None
        self._finish_first_dilemma = False

        # discount factor
        self.gamma = self.all_args.gamma

        # report environment configuration
        self.env_report()
        self._setup_logging(config)

        # dir
        self.model_dir = self.all_args.model_dir


        # if decay learning rate for optimizer
        if self.all_args.use_linear_lr_decay:
            dilemma_learning_rate = utils.linear_schedule_to_0(self.all_args.dilemma_lr)
            repu_learning_rate = utils.linear_schedule_to_0(self.all_args.repu_lr)
        else:
            dilemma_learning_rate = self.all_args.dilemma_lr
            repu_learning_rate = self.all_args.repu_lr

        # if decay update time for optimizer
        if self.all_args.use_linear_update:
            self.repu_n_epochs = utils.linear_countdown_to_1(
                self.all_args.repu_n_epochs
            )
            self.dilemma_n_epochs = utils.linear_schedule_from_to_int(
                1, self.all_args.dilemma_n_epochs
            )
        else:
            self.repu_n_epochs = self.all_args.repu_n_epochs
            self.dilemma_n_epochs = self.all_args.dilemma_n_epochs

        if self.all_args.dynamic_training :
            self.dilemma_train_freq = utils.delayed_linear_schedule(
                self.all_args.initial_dilemma_train_freq,
                self.all_args.dilemma_train_freq,
                self.all_args.dilemma_train_delay,
            )
        else:
            self.dilemma_train_freq = self.all_args.dilemma_train_freq

        if self.all_args.baseline_type =="NoRepu":
            self.dilemma_train_freq = 1

        # Alternate training of assessment and action networks
        self._train_repu = True if self.dilemma_train_freq != 0 else False

        self.set_random_seed(self.all_args.seed)

        self.norm_type = self.all_args.norm_type
        self.baseline_type = self.all_args.baseline_type
        self._setup_model(dilemma_learning_rate, repu_learning_rate)

        if self.model_dir is not None:
            self.restore()


        self.repu_rewards_type = self.all_args.repu_rewards_type
        self.repu_value_flag = self.all_args.repu_value_flag



        self.norm_pattern = self.norm_type


    def env_report(self):
        """
        Print the observation and action space of the environment
        """
        print("====== Environment Configuration ======")
        print("observation_space: ", self.observation_space)
        print("action_space: ", self.action_space)
        print("=" * 40)

    def _setup_logging(self, config):
        """
        Setup logging for the training process
        """
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
            self.logger = None
            self.gif_dir = str(self.run_dir + "/gifs")
            self.plot_dir = str(self.run_dir + "/plots")

        else:
            #  Configure directories for logging and saving models manually
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / "logs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.logger = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / "models")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.gif_dir = str(self.run_dir / "gifs")
            self.plot_dir = str(self.run_dir / "plots")

        if not os.path.exists(self.gif_dir):
            os.makedirs(self.gif_dir)

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # training start time
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None:
            self.ep_info_buffer = deque(maxlen=self.episode_length)
        if (
            self.dilemma_train_info_buffer is None
        ):  # only store the one training info, but all agents
            self.dilemma_train_info_buffer = deque(maxlen=self.num_agents)
        if self.repu_train_info_buffer is None:
            self.repu_train_info_buffer = deque(maxlen=self.num_agents)

    def _setup_model(self, dilemma_learning_rate, repu_learning_rate):
        """
        Setup the model for training
        """
        if self.norm_type == "RL":
            from grg.algorithm.reputation.policy import ReputationPolicy as ReputationPolicy
            from grg.algorithm.reputation.policy import ReputationCnnPolicy as ReputationCnnPolicy
            from grg.algorithm.reputation.repu_trainer import (
                Reputation as ReputationTrainAlgo,
            )
        else:
            from grg.algorithm.attention.policy import (
                CnnGRUMultiHeadReputationPolicy as ReputationCnnPolicy,
            )
            from grg.algorithm.attention.attention_trainer import (
                RepuAttention as ReputationTrainAlgo,
            )


        from grg.algorithm.dilemma.policy import DilemmaPolicy as DilemmaPolicy
        from grg.algorithm.dilemma.policy import DilemmaCnnPolicy as DilemmaCnnPolicy
        from grg.algorithm.dilemma.dilemma_trainer import Dilemma as DilemmaTrainAlgo


        self.repu_trainers = []
        self.repu_buffers = []
        self.dilemma_trainers = []
        self.dilemma_buffers = []

        dilemma_policy_kwargs = dict(
            net_arch=self.all_args.dilemma_net_arch,
            vf_observation_space=self.observation_space["dilemma_vf"] if self.baseline_type != "NL" else self.observation_space["dilemma"],
            share_features_extractor=False,
        )
        repu_policy_kwargs = dict(
            net_arch=self.all_args.repu_net_arch,
            vf_observation_space=self.observation_space["reputation_vf"] ,
            share_features_extractor=False,
        )
        for _ in range(self.num_agents):
            self.repu_trainers.append(
                ReputationTrainAlgo(
                    all_args=self.all_args,
                    logger=self.logger,
                    env=self.envs,
                    sub_env="reputation",
                    policy_class=ReputationCnnPolicy,
                    policy_kwargs=repu_policy_kwargs,
                    learning_rate=repu_learning_rate,
                    ent_coef=self.all_args.dilemma_ent_coef,
                    n_epochs=self.repu_n_epochs,
                    n_steps=self.n_steps,
                    # batch_size=self.n_steps*self.n_envs,
                    batch_size=self.r_minibatch_size,
                    normalize_advantage=True,
                    device=self.device,
                    ent_coef_fraction=self.all_args.repu_ent_coef_fraction,
                    ent_coef_initial_eps=self.all_args.repu_ent_coef_initial_eps,
                    ent_coef_final_eps=self.all_args.repu_ent_coef_final_eps,
                )
            )

            trainer_kwargs = {
                "all_args": self.all_args,
                "logger": self.logger,
                "env": self.envs,
                "sub_env": "dilemma",
                "policy_class": DilemmaCnnPolicy,
                "policy_kwargs": dilemma_policy_kwargs,
                "learning_rate": dilemma_learning_rate,
                "ent_coef": self.all_args.dilemma_ent_coef,
                "n_epochs": self.dilemma_n_epochs,
                "n_steps": self.n_steps,
                "batch_size": self.d_minibatch_size,
                "normalize_advantage": True,
                "device": self.device,
                "ent_coef_fraction": self.all_args.dilemma_ent_coef_fraction,
                "ent_coef_initial_eps": self.all_args.dilemma_ent_coef_initial_eps,
                "ent_coef_final_eps": self.all_args.dilemma_ent_coef_final_eps,
            }
            
            
            self.dilemma_trainers.append(DilemmaTrainAlgo(**trainer_kwargs))

        self.dilemma_rollout_buffer_class = CustomRolloutBuffer
        self.repu_rollout_buffer_class = CustomRolloutBuffer

        repu_obs_shape = get_obs_shape(self.observation_space["reputation"])
        repu_obs_shape = self.reshape_with_best_split(repu_obs_shape)
        dilemma_value_obs_shape = get_obs_shape(self.observation_space["dilemma_vf"]) 
        dilemma_value_obs_shape= self.reshape_with_best_split(dilemma_value_obs_shape)

        

        for _ in range(self.num_agents):
            self.dilemma_buffers.append(
                self.dilemma_rollout_buffer_class(
                    self.all_args.n_steps,
                    self.observation_space["dilemma"],
                    self.action_space["dilemma"],
                    self.device,
                    gamma=self.gamma,
                    n_envs=self.n_envs,
                    num_recipients=self.num_recipients,
                    num_agents=self.num_agents,
                    buffer_type="dilemma",
                    last_repu_obs_shape=repu_obs_shape,
                    buffer_index=_,
                    
                )
            )
            self.repu_buffers.append(
                self.repu_rollout_buffer_class(
                    # self.all_args.n_steps-int(self.n_steps/self.episode_length),
                    self.all_args.n_steps,
                    # - repu_buffer_reduction,  # the first dilemma step reward is not used
                    spaces.Box(0,1,shape=repu_obs_shape),
                    self.action_space["reputation"],
                    self.device,
                    gamma=self.gamma,
                    n_envs=self.n_envs,
                    num_recipients=self.num_recipients,
                    num_agents=self.num_agents,
                    buffer_type="reputation",
                    # norm_type=self.norm_type,
                    norm_type="attention",
                    last_repu_obs_shape=repu_obs_shape,
                    last_dilemma_vale_obs_shape=dilemma_value_obs_shape,
                    buffer_index=_,
                )
            )
        print("Model setup complete")

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        set_random_seed(seed, using_cuda=self.device.type == torch.device("cuda").type)
        self.action_space.seed(seed)
    
    def reshape_with_best_split(self,shape):
        *head, last = shape
        factors = [(i, last // i) for i in range(1, int(last**0.5)+1) if last % i == 0]
        best_pair = min(factors, key=lambda x: abs(x[0] - x[1]))
        # Sort to have larger dimension first if desired
        best_pair = tuple(sorted(best_pair, reverse=True))
        return tuple(head + list(best_pair))


    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError
    
    def save_model(self):
        """
        Save the Dilemma and Reputation policies and their optimizers for all agents.
        """
        for i in range(self.num_agents):
            # Save Dilemma Policy
            dilemma_policy = self.dilemma_trainers[i].policy
            dilemma_path = os.path.join(self.save_dir, f"dilemma_agent_{i}.pt")
            torch.save({
                'policy_state_dict': dilemma_policy.state_dict(),
                'optimizer_state_dict': dilemma_policy.optimizer.state_dict(),
            }, dilemma_path)
            print(f"Saved Dilemma policy for agent {i} to {dilemma_path}")

            # Save Reputation Policy
            repu_policy = self.repu_trainers[i].policy
            repu_path = os.path.join(self.save_dir, f"reputation_agent_{i}.pt")
            torch.save({
                'policy_state_dict': repu_policy.state_dict(),
                'optimizer_state_dict': repu_policy.optimizer.state_dict(),
            }, repu_path)
            print(f"Saved Reputation policy for agent {i} to {repu_path}")


    def load_model(self):
        """
        Load the Dilemma and Reputation policies and their optimizers for all agents.
        """
        for i in range(self.num_agents):
            # Load Dilemma Policy
            dilemma_policy = self.dilemma_trainers[i].policy
            dilemma_path = os.path.join(self.model_dir, f"dilemma_agent_{i}.pt")
            if os.path.exists(dilemma_path):
                checkpoint = torch.load(dilemma_path, map_location=self.device, weights_only=True)
                dilemma_policy.load_state_dict(checkpoint['policy_state_dict'])
                dilemma_policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                dilemma_policy.set_training_mode(False)
                print(f"Loaded Dilemma policy for agent {i} from {dilemma_path}")
            else:
                print(f"[Warning] No Dilemma model found for agent {i} at {dilemma_path}")

            # Load Reputation Policy
            repu_policy = self.repu_trainers[i].policy
            repu_path = os.path.join(self.model_dir, f"reputation_agent_{i}.pt")
            if os.path.exists(repu_path):
                checkpoint = torch.load(repu_path, map_location=self.device,weights_only=True)
                repu_policy.load_state_dict(checkpoint['policy_state_dict'])
                repu_policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                repu_policy.set_training_mode(False)
                print(f"Loaded Reputation policy for agent {i} from {repu_path}")
            else:
                print(f"[Warning] No Reputation model found for agent {i} at {repu_path}")
        print("Model loading complete")
        print('=' * 40)

    
    def restore(self):
        """
        Automatically called if model_dir is provided.
        """
        if self.model_dir is not None:
            print("Restoring models from:", self.model_dir)
            self.load_model()


    def _update_current_progress_remaining(
        self, num_timesteps: int, total_timesteps: int
    ) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        return 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_info_buffer(self, infos: list[dict[str, Any]]) -> None:
        """
        Retrieve env info and store it in the buffer
        """
        assert self.ep_info_buffer is not None, "self.ep_info_buffer is None"

        for idx, info in enumerate(infos):
            self.ep_info_buffer.extend([info])

    def print_episode_stats(self, logger_info):
        """
        Print the stats of the episode
        """
        print("-" * 44)
        print("| Reward/ {:>32} |".format(" " * 10))
        print(
            "|    Average Rewards  {:>20.3f} |".format(
                logger_info["results/episode_rewards"]
            )
        )
        # Reputation
        print("| Reputation/ {:>29}|".format(" " * 10))
        print(
            "|    Average Repu  {:>23.3f} |".format(
                logger_info["reputation/average_reputation"]
            )
        )
        print(
            "|    Collected Repu  {:>21.3f} |".format(
                logger_info["reputation/own_coin_collected_repu"]
            )
        )
        print(
            "|    Taken Repu  {:>25.3f} |".format(
                logger_info["reputation/collected_other_coin_repu"]
            )
        )
        if self.dilemma_train_info_buffer:
            print("| Dilemma Train/ {:>26}|".format(" " * 10))
            print(
                "|    Dilemma n_updates  {:>18.0f} |".format(
                    logger_info["dilemma_train/n_updates"]
                )
            )
            # print(
            #     "|    Dilemma Learning Rate  {:>14.3f} |".format(
            #         logger_info["dilemma_train/learning_rate"]
            #     )
            # )
            print(
                "|    Dilemma train_freq  {:>17.3f} |".format(
                    logger_info["time/dilemma_train_freq"]
                )
            )
            print(
                "|    Dilemma Entroy  {:>21.3f} |".format(
                    logger_info["dilemma_train/entropy_coef"]
                )
            )
            print(
                "|    Dilemma Policy Loss  {:>16.3f} |".format(
                    logger_info["dilemma_train/policy_loss"]
                )
            )
            print(
                "|    Average Dilemma Loss  {:>15.3f} |".format(
                    logger_info["dilemma_train/loss"]
                )
            )
            print(
                "|    Explained Variance  {:>17.3f} |".format(
                    logger_info["dilemma_train/explained_variance"]
                )
            )
        if self.repu_train_info_buffer:
            print("| Repu Train/ {:>29}|".format(" " * 10))
            print(
                "|    Repu n_updates  {:>21.0f} |".format(
                    logger_info["repu_train/n_updates"]
                )
            )
            print(
                "|    Repu Entroy  {:>24.3f} |".format(
                    logger_info["repu_train/entropy_coef"]
                )
            )
            print(
                "|    Repu Policy Loss  {:>19.3f} |".format(
                    logger_info["repu_train/policy_gradient_loss"]
                )
            )
            print(
                "|    Average Repu Loss  {:>18.3f} |".format(
                    logger_info["repu_train/loss"]
                )
            )
        # Environment
        print("| Environment/ {:>28}|".format(" " * 10))
        print(
            "|    Episode Collected Coins  {:>12.3f} |".format(
                logger_info["results/coins_collected_episode"]
            )
        )
        print(
            "|    Episode Own Coins  {:>18.3f} |".format(
                logger_info["results/own_coin_collected_episode"]
            )
        )
        print(
            "|    Episode Others Coins  {:>15.3f} |".format(
                logger_info["results/coin_taken_by_others_episode"]
            )
        )
        print(
            "|    Own Coins Proportion  {:>15.3f} |".format(
                logger_info["results/own_coin_proportion"]
            )
        )
        print("-" * 44, "\n")

    def log_metrics(self, logging_infos,iteration_episode):
        """
        Log episode info.
        :param logging_infos: (dict) information about episode update.
        :param total_num_steps: (int) total number of training env steps.
        """
        if self.use_wandb:
            wandb.log({**logging_infos, "Steps": self.num_timesteps})

    def log_video(self, video_path, iteration_episode):
        """
        Log video to wandb with custom step tracking.
        """
        if self.use_wandb:
            wandb.log({
                "render_video": wandb.Video(video_path, format="mp4"),
                "Episodes": iteration_episode
            })

