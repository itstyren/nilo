import random
import time
import numpy as np
from ..base_runner import BaseRunner
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import sys
import grg.utils.tools as utils
from stable_baselines3.common.utils import get_schedule_fn


class PairwiseRunner(BaseRunner):
    """
    Runner class for pairwise training

    :param n_steps: The number of steps to run for each environment per update
    """

    def __init__(self, config):
        super(PairwiseRunner, self).__init__(config)
        self.continue_training = True
        
        # Early stopping for low cooperation
        self.low_coop_threshold = 0.05
        self.max_low_coop_episodes = 100
        self.low_coop_count = 0
        self.episode_coop_levels = []

    def run(self):
        """
        Run the training and evaluation loop
        """
        self._warmup()

        iteration_episode = 0

        while self.num_timesteps < self._total_timesteps:
            continue_training = self.collect_rollouts(self.envs, self.n_steps)

            if not continue_training:
                break

            iteration_episode += self.n_rollout_threads
            # pass to all trainers
            progress_remaining = self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            # Display training information
            if iteration_episode % self._log_interval == 0:
                assert self.ep_info_buffer is not None
                # Check for early stopping before logging
                self._dump_logs(iteration_episode)

            if self.early_stop and self._check_early_stop():
                print(f"\n=== EARLY STOPPING ===")
                print(f"Average cooperation level below {self.low_coop_threshold} for {self.max_low_coop_episodes} consecutive episodes.")
                print(f"Stopping training at episode {iteration_episode}")
                break
            self.train_agent_policy(progress_remaining)

    def _warmup(self):
        """
        Warmup the environment
        """
        if self._last_dilemma_obs is None:
            assert self.envs is not None, "Environment is not initialized"
            # reset the environment
            initial_obs, coop_level = self.envs.reset()
            dilemma_obs = self.process_obs_to_specific_usage(
                initial_obs, "dilemma_obs")
            repu_obs = self.process_obs_to_specific_usage(
                initial_obs, "repu_obs")
            self._last_dilemma_obs = dilemma_obs.copy()
            self._last_last_repu_obs = repu_obs.copy()

            self.dilemma_updates_freq_schedule = get_schedule_fn(
                self.dilemma_train_freq
            )

            self._last_episode_starts = np.ones(
                (self.envs.num_envs,), dtype=bool)

            print(
                "====== Initial Cooperative Level {:.2f} ======".format(
                    np.mean(coop_level)
                )
            )

        # total episode number
        self._total_episodes = int(
            self._total_timesteps) // self.episode_length

        # current timesteps within the single thread
        self._internal_timesteps = 0

    def _check_early_stop(self):
        """
        Check if training should be stopped early due to consistently low cooperation levels.
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        # Skip early stopping if disabled
        if not self.early_stop:
            return False
            
        if self.ep_info_buffer is None or len(self.ep_info_buffer) == 0:
            return False
            
        # Get the most recent cooperation level
        recent_coop_level = safe_mean(
            [ep_info["coop_level"] for ep_info in self.ep_info_buffer]
        )
        
        # Add to our tracking list
        self.episode_coop_levels.append(recent_coop_level)
        
        # Keep only the last max_low_coop_episodes entries
        if len(self.episode_coop_levels) > self.max_low_coop_episodes:
            self.episode_coop_levels.pop(0)
        
        # Check if we have enough episodes to evaluate
        if len(self.episode_coop_levels) < self.max_low_coop_episodes:
            return False
            
        # Check if all recent episodes are below threshold
        all_below_threshold = all(coop_level < self.low_coop_threshold 
                                 for coop_level in self.episode_coop_levels)
        
        if all_below_threshold:
            print(f"\nLow cooperation detected:")
            print(f"Last {self.max_low_coop_episodes} cooperation levels: {[f'{level:.4f}' for level in self.episode_coop_levels]}")
            print(f"All below threshold: {self.low_coop_threshold}")
            return True
            
        return False

    def collect_rollouts(self, env: VecEnv, n_rollout_steps: int) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.

        :param env: The training environment
        :param n_rollout_steps: The number of rollout steps per update

        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert (
            self._last_dilemma_obs is not None
        ), "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)

        n_steps = 0
        # set training mode to False
        # set the buffer to reset
        for agent_idx in range(self.num_agents):
            self.dilemma_trainers[agent_idx].policy.set_training_mode(False)
            self.dilemma_buffers[agent_idx].reset()

            if self.baseline_type == "None":
                self.repu_trainers[agent_idx].policy.set_training_mode(False)
                self.repu_buffers[agent_idx].reset()

        # Create an array of roullout env indices
        batch_indices = np.arange(self.n_rollout_threads)[:, np.newaxis]
        # dilemma action ---> setp ----->reputaion action ----> step, this a one rollout step

        while n_steps < n_rollout_steps:
            # Get the dilemma action
            dilemma_actions, dilemma_values, dilemma_log_probs,_ = self.get_action(
                action_type="dilemma"
            )
            updated_obs, rewards, _, truncations, dilemma_infos = env.step(
                dilemma_actions, action_type="dilemma"
            )
            agent_recipients_dilemma_actions = np.array(
                [dilemma_info["recipient_actions"]
                    for dilemma_info in dilemma_infos]
            )

            dilemma_values = self.get_dilemma_value(
                dilemma_actions, agent_recipients_dilemma_actions
            )

            # use obs after dilemma action to evaluate reputation
            repu_obs = self.process_obs_to_specific_usage(
                updated_obs, "repu_obs")

            _dilemma_recipients_idx = np.array(
                [info["agent_recipients_idx"] for info in dilemma_infos]
            )

            # get reputation obs after dilemma action
            self._last_repu_obs = repu_obs.copy()

            # Get the reputation action
            if self.baseline_type=="None" and (self.norm_pattern == "RL" or self.norm_pattern == "attention"):
                repu_actions, repu_values, repu_log_probs,attention_weight = self.get_action(
                    action_type="reputation"
                )

                repu_values = self.get_repu_value(repu_actions)

            else:
                # for predefined norm, dont need to get action
                repu_actions, repu_values, repu_log_probs = np.random.rand(
                    3, self.n_rollout_threads, self.num_agents, self.num_agents
                )

            if self.action_space["reputation"].__class__ == spaces.Box:
                clipped_repu_actions = np.clip(
                    repu_actions,
                    self.action_space["reputation"].low,
                    self.action_space["reputation"].high,
                )
            else:
                clipped_repu_actions = repu_actions
            next_dilemma_obs, _, _, _, repu_infos = env.step(
                clipped_repu_actions, action_type="reputaiton"
            )

            if self.norm_pattern == "attention" and len(attention_weight)>0:
                for i, info_dict in enumerate(repu_infos):
                    info_dict['attention_weight'] = attention_weight[i].tolist()


            # if (self.repu_rewards_type == 'plain' or self.repu_rewards_type == 'None'):
            if self.repu_rewards_type == "plain":
                # if True:
                repu_rewards = rewards.copy()
            elif self.repu_rewards_type == "category":
                repu_rewards = self.categorize_and_normalize_repu_rewards(
                    rewards.copy(), dilemma_actions
                )
            else:
                repu_rewards = self.categorize_repu_rewards_by_norm(
                    repu_obs,
                    clipped_repu_actions,
                    dilemma_actions,
                    agent_recipients_dilemma_actions,
                    rewards.copy(),
                    _dilemma_recipients_idx,
                )

            self.num_timesteps += env.num_envs

            combined_infos = np.array(
                [{**d1, **d2} for d1, d2 in zip(dilemma_infos, repu_infos)],
                dtype=object,
            )

            self._update_info_buffer(combined_infos)

            # if the episode is done, the env will be reset in this if statement
            if repu_infos[0]["epsidoe_end"]:
                # get the terminal observation after last reputation action
                dilemma_terminal_obs = self.process_obs_to_specific_usage(
                    next_dilemma_obs, "dilemma_obs"
                )
                dilemma_terminal_actions, _, _,_ = self.get_action(
                    action_type="dilemma", obs=dilemma_terminal_obs
                )
                # env reset here
                new_episide_obs, _, _, truncations, terminal_infos = env.step(
                    dilemma_terminal_actions, action_type="dilemma"
                )
                # get the recipient action for the last dilemma action
                agent_recipients_dilemma_actions = np.array(
                    [
                        dilemma_info["recipient_actions"]
                        for dilemma_info in terminal_infos
                    ]
                )

                # get terminal observation after last dilemma action
                terminal_obs = np.array(
                    [info["terminal_observation"] for info in terminal_infos]
                )
                dilemma_terminal_values = self.get_dilemma_value(
                    dilemma_terminal_actions,
                    agent_recipients_dilemma_actions,
                    dilemma_obs=dilemma_terminal_obs,
                )

                repu_terminal_obs = self.process_obs_to_specific_usage(
                    terminal_obs, "repu_obs"
                )
                if self.baseline_type=="None" and (self.norm_pattern == "RL" or self.norm_pattern == "attention"):
                    repu_terminal_values = self.get_repu_value(
                        repu_actions, repu_obs=repu_terminal_obs
                    )
                    repu_terminal_values = np.take_along_axis(
                        repu_terminal_values, _dilemma_recipients_idx, axis=2
                    )
                else:
                    repu_terminal_values = np.random.rand(
                        self.n_rollout_threads, self.num_agents, self.num_recipients
                    )
                rewards += self.gamma * dilemma_terminal_values
                if self.repu_value_flag == "return":
                    repu_rewards += self.gamma * repu_terminal_values

            # repu info should be at after the first dilemma action
            # no none make it skip the first step
            if self._finish_first_dilemma:
                if self.baseline_type=="None":
                    for agent_idx in range(self.num_agents):
                        # get the recipient index for current agent
                        recipient_idx = _dilemma_recipients_idx[:, agent_idx]
                        # add last round repu info and this round dilemma reward to the bufferb
                        if self.norm_pattern == "attention":
                            self.repu_buffers[agent_idx].add(
                                # here repu obse haven't been update
                                self._last_last_repu_obs[:, agent_idx],
                                # self._last_repu_obs[:, agent_idx],  # here repu obse haven't been update
                                self._last_repu_actions[:, agent_idx],
                                # use just received rewards
                                repu_rewards[:, agent_idx],
                                self._last_last_episode_starts,
                                self._last_repu_values[:, agent_idx][
                                    batch_indices, recipient_idx
                                ],
                                self._last_repu_log_probs[:, agent_idx],
                                recipients_index=recipient_idx,
                                last_repu_observations=self._last_repu_obs[:, agent_idx].copy(),
                            )
                        else:
                            self.repu_buffers[agent_idx].add(
                                # here repu obse haven't been update
                                self._last_last_repu_obs[:, agent_idx],
                                # self._last_repu_obs[:, agent_idx],  # here repu obse haven't been update
                                self._last_repu_actions[:, agent_idx],
                                # use just received rewards
                                repu_rewards[:, agent_idx],
                                self._last_last_episode_starts,
                                self._last_repu_values[:, agent_idx][
                                    batch_indices, recipient_idx
                                ],
                                self._last_repu_log_probs[:, agent_idx],
                                recipients_index=recipient_idx,
                                last_repu_observations=self._last_repu_obs[:, agent_idx].copy(),
                            )


                # print('current step:',n_steps)
                if repu_infos[0]["epsidoe_end"]:
                    self._finish_first_dilemma = False
            else:
                # else means last round is the firsst setp of reset env
                self._finish_first_dilemma = True

                if self.baseline_type=="None":
                    for agent_idx in range(self.num_agents):
                        self.repu_buffers[agent_idx].remove_current_pos()
            # _dilemma_recipients_idx[:, agent_idx]
            # add to dilenna rollout buffer

            for agent_idx in range(self.num_agents):
                self.dilemma_buffers[agent_idx].add(
                    self._last_dilemma_obs[:, agent_idx],
                    dilemma_actions[:, agent_idx],
                    rewards[:, agent_idx],
                    self._last_episode_starts,
                    dilemma_values[:, agent_idx],
                    dilemma_log_probs[:, agent_idx],
                    recipient_action=agent_recipients_dilemma_actions[:, agent_idx],
                    last_repu_observations=self._last_last_repu_obs[:, agent_idx],
                )


            self._last_repu_actions = repu_actions.copy()
            self._last_repu_values = repu_values.copy()
            self._last_repu_rewards = repu_rewards.copy()
            self._last_repu_log_probs = repu_log_probs.copy()

            self._last_last_episode_starts = self._last_episode_starts.copy()
            self._last_last_repu_obs = self._last_repu_obs.copy()

            if not repu_infos[0]["epsidoe_end"]:
                dilemma_obs = self.process_obs_to_specific_usage(
                    next_dilemma_obs, "dilemma_obs"
                )
            else:
                dilemma_obs = self.process_obs_to_specific_usage(
                    new_episide_obs, "dilemma_obs"
                )

            self._last_dilemma_obs = dilemma_obs.copy()
            self._last_episode_starts = truncations
            n_steps += 1
            self._internal_timesteps += 1

        last_repu_value = []
        with torch.no_grad():
            # compute the last value for the last timestep

            last_dilemma_actions, _, _,_ = self.get_action(action_type="dilemma")
            _, _, _, _, last_dilemma_infos = env.step(
                last_dilemma_actions, action_type="dilemma", move_step=False
            )
            last_agent_recipients_dilemma_actions = np.array(
                [
                    last_dilemma_info["recipient_actions"]
                    for last_dilemma_info in last_dilemma_infos
                ]
            )

            last_dilemma_values = self.get_dilemma_value(
                last_dilemma_actions, last_agent_recipients_dilemma_actions
            )

            for agent_idx in range(self.num_agents):
                recipient_idx = _dilemma_recipients_idx[:, agent_idx]
                last_repu_value.append(
                    self._last_repu_values[:,
                                           agent_idx][batch_indices, recipient_idx]
                )

        last_dilemma_values = np.array(last_dilemma_values)
        last_repu_value = np.array(last_repu_value)
        # compute the return and advantage for dilemma and reputation
        for agent_idx in range(self.num_agents):
            
            # self.dilemma_buffers[agent_idx].compute_td0_returns(
            self.dilemma_buffers[agent_idx].compute_returns_and_advantage(
                # last_dilemma_values[agent_idx, :], dones=truncations
                last_dilemma_values[:, agent_idx],
                dones=truncations,
            )
            if self.baseline_type=="None" and self.dilemma_train_freq != 1 and self.repu_value_flag == "return":
                # self.repu_buffers[agent_idx].compute_td0_returns(
                self.repu_buffers[agent_idx].compute_returns_and_advantage(
                    last_repu_value[agent_idx, :],
                    dones=truncations,
                )
        return True

    def get_action(self, action_type="dilemma", obs=None):
        """
        Get one step action for all agents
        """
        action_set, value_set, log_probs_set,attention_weight_set = [], [], [],[]
        action_dim = self.num_recipients if action_type == "dilemma" else -1
        value_dim = -1
        trainer = (
            self.dilemma_trainers if action_type == "dilemma" else self.repu_trainers
        )
        _last_obs = obs if obs is not None else (
            self._last_dilemma_obs if action_type == "dilemma" else self._last_repu_obs
        )

        for agent_idx in range(self.num_agents):
            with torch.no_grad():
                agent_obs = _last_obs[:, agent_idx]

                if self.norm_pattern == "attention":
                    obs_shape = (-1, agent_obs.shape[-1]) if action_type == "dilemma" else (
                        *agent_obs.shape[:-2], -1)
                else:
                    obs_shape = (-1, _last_obs.shape[-1]) if action_type == "dilemma" else (-1, np.prod(
                        agent_obs.shape[-2:]))
                obs_tensor = obs_as_tensor(
                    agent_obs.reshape(obs_shape), self.device)

                if self.norm_pattern == "attention" and action_type == "reputation":
                    actions, values, log_probs,attention_weight = trainer[agent_idx].policy.forward(
                        obs_tensor,attention_weights=self.attention_weights
                    )
                    # input agent action, opponent reputation
                    second_norm_attention_weight=utils.category_attention_weight_to_norm(
                        _last_obs[:, agent_idx].squeeze(2)[:, :, :4][:, :, [2, 1]],attention_weight)
                    
                    attention_weight_set.append(second_norm_attention_weight)

                else:
                    actions, values, log_probs = trainer[agent_idx].policy.forward(
                        obs_tensor
                    )

                if (
                    action_type == "reputation"
                    and self.action_space["reputation"].__class__ == spaces.Box
                ):
                    actions = actions.squeeze(1)
                # if action_type == "dilemma":
                #     breakpoint()
            # if action_type == "reputation":
            #     breakpoint()
            actions = utils.t2n(actions, self.n_rollout_threads, action_dim)
            log_probs = utils.t2n(
                log_probs, self.n_rollout_threads, action_dim)
            values = (
                utils.t2n(values, self.n_rollout_threads, value_dim)
                if len(values) > 1
                else np.array(values)
            )

            # not need to clip dilemma action
            if action_type == "dilemma":
                action_set.append(actions.astype(int).tolist())
            else:
                action_set.append(actions.tolist())

            value_set.append(values.astype(float).tolist())
            log_probs_set.append(log_probs.astype(float).tolist())
        return (
            np.stack(action_set, axis=1),
            np.stack(value_set, axis=1),
            np.stack(log_probs_set, axis=1),
            # np.stack(attention_weight_set, axis=1) if attention_weight_set else None,
            np.array(attention_weight_set),
        )

    def get_dilemma_value(
        self,
        dilemma_actions,
        agent_recipients_dilemma_actions,
        dilemma_obs=None,
    ):
        """
        Get the value of the dilemma action
        """
        # Use the last observed dilemma if none provided
        dilemma_obs = self._last_dilemma_obs if dilemma_obs is None else dilemma_obs

        # Expand actions once
        dilemma_actions = np.expand_dims(
            dilemma_actions, axis=-1)  # e.g., (1, 36, 1, 1)
        agent_recipients_actions = np.expand_dims(
            agent_recipients_dilemma_actions, axis=-1)

        # Combine observations
        extended_obs = np.concatenate(
            [dilemma_obs, dilemma_actions, agent_recipients_actions],
            axis=-1,
        )  # (1, 36, ..., new_feature_dim)

        value_set = []

        for agent_idx in range(self.num_agents):
            with torch.no_grad():
                # Select agent-specific observations
                agent_obs = extended_obs[:, agent_idx]
                # Convert to tensor using agent's policy method
                critic_obs_tensor = self.dilemma_trainers[
                    agent_idx
                ].policy.obs_to_tensor(
                    np.concatenate(agent_obs),
                    observation_type="value_pred",
                )[
                    0
                ]

                # Predict value
                values = self.dilemma_trainers[agent_idx].policy.predict_values(
                    critic_obs_tensor
                )
            # Convert tensor to numpy
            values = utils.t2n(values, self.n_rollout_threads, -1)
            value_set.append(values)

        return np.stack(value_set, axis=1)

    def get_repu_value(
        self,
        repu_actions,
        repu_obs=None,
    ):
        """
        Get the value of the reputation action
        """
        repu_obs = (
            self._last_repu_obs if repu_obs is None else repu_obs
        )  # [batch,agent,obs_dim], e.g. (8, 16, 16, 1, 3)

        # Flatten if needed
        if repu_obs.shape[3] != 1:
            repu_obs = repu_obs.reshape(
                *repu_obs.shape[:3], 1, -1)  # Merge last two dims

        # Expand actions once
        # shape (batch, agent, agent, 1, 1)
        repu_actions_expanded = repu_actions[..., None, None]

        extended_repu_obs = np.concatenate(
            [repu_obs, repu_actions_expanded], axis=-1
        )

        # Preprocess all inputs for all agents at once
        batch_size, num_agents, _, _, input_dim = extended_repu_obs.shape
        extended_repu_obs_flat = extended_repu_obs.reshape(
            batch_size, num_agents, -1, input_dim)

        value_set = []

        with torch.no_grad():
            for agent_idx in range(self.num_agents):
                # Select the interaction dimension properly
                # (batch, agent, input_dim)
                critic_obs = extended_repu_obs_flat[:, agent_idx, :, :]
                # (batch*agent, input_dim)
                critic_obs = critic_obs.reshape(-1, critic_obs.shape[-1])
                critic_obs_tensor = self.repu_trainers[agent_idx].policy.obs_to_tensor(
                    critic_obs, observation_type="value_pred"
                )[0]

                # Predict values
                values = self.repu_trainers[agent_idx].policy.predict_values(
                    critic_obs_tensor)

                # Convert to numpy
                values = utils.t2n(values, self.n_rollout_threads, -1)

                value_set.append(values)

        return np.stack(value_set, axis=1)

    def process_obs_to_specific_usage(self, obs, usage_type):
        """
        Process the observation to get the reputation usage
        """
        usage_obs = np.array([
            [obs[rollout_idx, agent_idx][usage_type]
                for agent_idx in range(self.num_agents)]
            for rollout_idx in range(self.n_rollout_threads)
        ], dtype=np.float64)

        return usage_obs

    def train_agent_policy(self, progress_remaining):
        """
        Train either the reputation or dilemma trainers
        """
        self.train_freq = self.dilemma_updates_freq_schedule(
            progress_remaining)
        # print(dilemma_train_freq)

        if random.random() < self.train_freq:
            # print("Train dilemma")
            # Train dilemma
            for agent_idx in range(self.num_agents):
                self.dilemma_trainers[agent_idx]._current_progress_remaining = (
                    progress_remaining
                )
                dilemma_train_info = self.dilemma_trainers[agent_idx].train(
                    self.dilemma_buffers[agent_idx],
                    norm_pattern=(
                        self.norm_pattern
                        if (self.norm_type != "RL" and self.norm_type != "attention")
                        else self.repu_trainers[agent_idx].policy
                    ),
                    self_index=agent_idx,
                )
                self.dilemma_train_info_buffer.extend([dilemma_train_info])
        else:
            # print("Train reputation")
            # Train reputation
            for agent_idx in range(self.num_agents):
                self.repu_trainers[agent_idx]._current_progress_remaining = (
                    progress_remaining
                )
                repu_train_info = self.repu_trainers[agent_idx].train(
                    self.repu_buffers[agent_idx],
                    value_type=self.repu_value_flag,
                    rewards_type=self.repu_rewards_type,
                    dilemma_policy=self.dilemma_trainers[agent_idx].policy,
                    self_index=agent_idx,
                )
                self.repu_train_info_buffer.extend([repu_train_info])

    def _dump_logs(self, iteration):
        """
        Dump logs
        """
        assert self.ep_info_buffer is not None

        logger_info = {}

        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        fps = int(
            (self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)

        logger_info["time/fps"] = fps
        logger_info["time/time_elapsed"] = time_elapsed
        logger_info["time/total_timesteps"] = self.num_timesteps
        logger_info["time/episode"] = iteration
        logger_info["time/dilemma_train_freq"] = self.train_freq

        print(
            "\n Dillemma_T({})S({}) Exp {} Substrate {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.all_args.dilemma_T,
                self.all_args.dilemma_S,
                self.exp_name,
                self.substrate_name,
                iteration,
                self._total_episodes,
                self.num_timesteps,
                self._total_timesteps,
                fps,
            )
        )

        logger_info["results/avg_cooperation_level"] = safe_mean(
            [ep_info["coop_level"] for ep_info in self.ep_info_buffer]
        )
        logger_info["results/average_rewards"] = safe_mean(
            [ep_info["average_rewards"] for ep_info in self.ep_info_buffer]
        )
        logger_info["results/self_loop_fraction"] = safe_mean(
            [ep_info["self_loop_fraction"] for ep_info in self.ep_info_buffer]
        )
        logger_info["result/repu_based_coop"] = safe_mean(
            [ep_info["repu_based_action"][0]
                for ep_info in self.ep_info_buffer]
        )
        logger_info["result/repu_based_defect"] = safe_mean(
            [ep_info["repu_based_action"][1]
                for ep_info in self.ep_info_buffer]
        )
        # skip first round has no training info
        logger_info["dilemma_train/learning_rate"] = safe_mean(
            [ep_info["learning_rate"]
                for ep_info in self.dilemma_train_info_buffer]
        )
        logger_info["dilemma_train/policy_gradient_loss"] = safe_mean(
            [
                ep_info["policy_gradient_loss"]
                for ep_info in self.dilemma_train_info_buffer
            ]
        )
        logger_info["dilemma_train/explained_variance"] = safe_mean(
            [
                ep_info["explained_variance"]
                for ep_info in self.dilemma_train_info_buffer
            ]
        )
        logger_info["dilemma_train/loss"] = safe_mean(
            [ep_info["loss"] for ep_info in self.dilemma_train_info_buffer]
        )
        logger_info["dilemma_train/policy_loss"] = safe_mean(
            [
                ep_info["policy_gradient_loss"]
                for ep_info in self.dilemma_train_info_buffer
            ]
        )
        logger_info["dilemma_train/n_updates"] = safe_mean(
            [ep_info["n_updates"]
                for ep_info in self.dilemma_train_info_buffer]
        )
        logger_info["dilemma_train/entropy_coef"] = safe_mean(
            [ep_info["entropy_coef"]
                for ep_info in self.dilemma_train_info_buffer]
        )
        # reputation training info
        logger_info["repu_train/learning_rate"] = safe_mean(
            [ep_info["learning_rate"]
                for ep_info in self.repu_train_info_buffer]
        )
        logger_info["repu_train/policy_gradient_loss"] = safe_mean(
            [ep_info["policy_gradient_loss"]
                for ep_info in self.repu_train_info_buffer]
        )
        logger_info["repu_train/explained_variance"] = safe_mean(
            [ep_info["explained_variance"]
                for ep_info in self.repu_train_info_buffer]
        )
        logger_info["repu_train/loss"] = safe_mean(
            [ep_info["loss"] for ep_info in self.repu_train_info_buffer]
        )
        logger_info["repu_train/n_updates"] = safe_mean(
            [ep_info["n_updates"] for ep_info in self.repu_train_info_buffer]
        )
        logger_info["repu_train/entropy_coef"] = safe_mean(
            [ep_info["entropy_coef"]
                for ep_info in self.repu_train_info_buffer]
        )

        logger_info["reputation/average_reputation"] = safe_mean(
            [ep_info["average_reputation"] for ep_info in self.ep_info_buffer]
        )
        logger_info["reputation/average_coop_repu"] = safe_mean(
            [ep_info["strategy_reputation"][0]
                for ep_info in self.ep_info_buffer]
        )
        logger_info["reputation/average_defect_repu"] = safe_mean(
            [ep_info["strategy_reputation"][1]
                for ep_info in self.ep_info_buffer]
        )
        logger_info["dilemma_train/adv_coop"] = safe_mean(
            [ep_info["act_adv_sum"][0]
                for ep_info in self.dilemma_train_info_buffer]
        )
        logger_info["dilemma_train/adv_defect"] = safe_mean(
            [ep_info["act_adv_sum"][1]
                for ep_info in self.dilemma_train_info_buffer]
        )

        logger_info['attention_weight/CG'] = safe_mean(
            [ep_info["attention_weight"][0]
                for ep_info in self.ep_info_buffer if "attention_weight" in ep_info]
        )
        logger_info['attention_weight/CB'] = safe_mean(
            [ep_info["attention_weight"][1]
                for ep_info in self.ep_info_buffer if "attention_weight" in ep_info]
        )
        logger_info['attention_weight/DG'] = safe_mean(
            [ep_info["attention_weight"][2]
                for ep_info in self.ep_info_buffer if "attention_weight" in ep_info]
        )
        logger_info['attention_weight/DB'] = safe_mean(
            [ep_info["attention_weight"][3]
                for ep_info in self.ep_info_buffer if "attention_weight" in ep_info]
        )
        
        # Add early stopping tracking info only if early stopping is enabled
        if self.early_stop:
            logger_info["early_stop/recent_coop_levels"] = len(self.episode_coop_levels)
            logger_info["early_stop/low_coop_count"] = sum(1 for level in self.episode_coop_levels if level < self.low_coop_threshold)
            if len(self.episode_coop_levels) > 0:
                logger_info["early_stop/min_recent_coop"] = min(self.episode_coop_levels)
                logger_info["early_stop/avg_recent_coop"] = np.mean(self.episode_coop_levels)
        # print([ep_info["act_adv_sum"][0] for ep_info in self.dilemma_train_info_buffer])

        # print('')
        # print("coop assign good",logger_info["repu_train/adv_coop_G"]," coop assign bad",logger_info["repu_train/adv_coop_B"])
        # print("defect assign bad",logger_info["repu_train/adv_defect_B"], "defect assign good",logger_info["repu_train/adv_defect_G"])
        if not self.use_wandb:
            print("adv coop", logger_info["dilemma_train/adv_coop"])
            print("adv defect", logger_info["dilemma_train/adv_defect"])
            # Print early stopping status only if early stopping is enabled
            if self.early_stop and len(self.episode_coop_levels) > 0:
                print(f"Early stop tracking: {len(self.episode_coop_levels)}/{self.max_low_coop_episodes} episodes, " +
                      f"recent avg coop: {np.mean(self.episode_coop_levels):.4f}, " +
                      f"below threshold: {sum(1 for level in self.episode_coop_levels if level < self.low_coop_threshold)}")
            self.print_episode_stats(logger_info)
        # print(self._internal_timesteps,self.num_timesteps)
        else:
            self.log_metrics(logger_info)

    def categorize_and_normalize_repu_rewards(self, repu_rewards, dilemma_actions):
        """
        Categorize the reputation rewards and normalize them
        """
        # 1. Extract the categorical feature from the last dim of `_last_dilemma_obs`
        # categories  = self._last_dilemma_obs[..., -1]  # shape: (8, 9, 1)
        # categories  = self._last_last_repu_obs[..., -1]  # shape: (8, 9, 1)

        # Flatten
        # flat_categories = categories.reshape(-1)  # shape: (72,)
        flat_act_categories = dilemma_actions.reshape(-1)  # shape: (72,)
        flat_rewards = repu_rewards.reshape(-1)  # shape: (72,)

        # Create masks
        mask_0 = flat_act_categories == 0
        mask_1 = flat_act_categories == 1

        # Init output
        normalized_flat = np.zeros_like(flat_rewards)

        # Normalize per category
        if np.any(mask_0):
            r0 = flat_rewards[mask_0]
            normalized_flat[mask_0] = (r0 - r0.mean()) / (r0.std() + 1e-8)

        if np.any(mask_1):
            r1 = flat_rewards[mask_1]
            normalized_flat[mask_1] = (r1 - r1.mean()) / (r1.std() + 1e-8)

        # Reshape back
        normalized_rewards = normalized_flat.reshape(
            repu_rewards.shape)  # (8, 9, 1)
        return normalized_rewards

    def categorize_repu_rewards_by_norm(
        self,
        repu_obs,
        clipped_repu_actions,
        self_d_act,
        recipt_d_act,
        rewards,
        _dilemma_recipients_idx,
    ):
        """
        Categorize the reputation rewards and normalize them
        """
        # # 1. Extract the categorical feature from the last dim of `_last_dilemma_obs`
        # if self._last_last_repu_obs.shape[3] != 1:
        #     _last_last_repu_obs = self._last_last_repu_obs.reshape(
        #         *self._last_last_repu_obs.shape[:3], 1, -1
        #     )
        #     array = _last_last_repu_obs.squeeze(3)
        # else:
        #     array = self._last_last_repu_obs.squeeze(3)
        # # breakpoint()

        # # array = repu_obs.squeeze(3)

        # # First, expand _dilemma_recipients_idx to match last dimension for take_along_axis
        # recipient_idx_expanded = np.expand_dims(
        #     _dilemma_recipients_idx, axis=-1)  # shape: (8, 16, 1, 1)

        # # Now use take_along_axis along axis=2 (the recipient dim)
        # selected_vectors = np.take_along_axis(
        #     array, recipient_idx_expanded, axis=2)  # shape: (8, 16, 1, 2)

        # # # Finally, squeeze to get shape (8, 16, 2)
        # selected_last_last_repu = selected_vectors.squeeze(2).reshape(-1,  self.repu_obs_shape[-1])

        # # # equivalent to last repu action
        assign_repu = self._last_dilemma_obs[..., -1].reshape(-1)
        # assign_repu=np.take_along_axis(clipped_repu_actions, _dilemma_recipients_idx, axis=2).reshape(-1)  # shape: (8, 16, 1, 2)
        d_rewards = rewards.reshape(-1)
        repu_rewards = np.zeros_like(d_rewards)
        # print('clip repu:', clipped_repu_actions[0][0])
        # repu_rewards_2 = np.zeros_like(d_rewards)
        # for i,(last_r_obs,r_act) in enumerate(zip(selected_last_last_repu,assign_repu)):
        #     # if self.repu_rewards_type=='IS':
        #     # # if False:
        #     #     if last_r_obs[-2]==0:
        #     #         if r_act == 1:
        #     #             repu_rewards[i] += 1
        #     #         else:
        #     #             repu_rewards[i] -= 1
        #     #     else:
        #     #         if r_act >= 0.5:
        #     #             repu_rewards[i] -= 1
        #     #         else:
        #     #             repu_rewards[i] += 1
        #     # # elif self.repu_rewards_type=='SJ':
        #     # elif True:
        #     #     if last_r_obs[-2]==0:
        #     #         if last_r_obs[1]==1.0: # recipient is good reputation
        #     #             if r_act==1: # assign good reputation
        #     #                 repu_rewards[i]+=1
        #     #             else:
        #     #                 repu_rewards[i]-=1
        #     #         else: # recipient is bad reputation
        #     #             if r_act==1: # assign good reputation
        #     #                 repu_rewards[i]-=1
        #     #             else:
        #     #                 repu_rewards[i]+=1
        #     #     else: # agent decide to defect
        #     #         if last_r_obs[1]==1.0: # recipient is good reputation
        #     #             if r_act==1: # assign good reputation
        #     #                 repu_rewards[i]-=1
        #     #             else:
        #     #                 repu_rewards[i]+=1
        #     #         else: # recipient is bad reputation
        #     #             if r_act==1: # assign good reputation
        #     #                 repu_rewards[i]+=1
        #     #             else:
        #     #                 repu_rewards[i]-=1

        #     agent_coop = last_r_obs[-2] == 0  # assume 0 = cooperate
        #     recipient_good = last_r_obs[1] == 1.0
        #     assigned_good = r_act == 1

        #     # desired_repu = int(agent_coop ^ (not recipient_good))
        #     # scale = 10  # sharpness of transition

        #     # # Aligning reputation with Stern Judging rule
        #     # # Reward is positive if assign_repu ≈ desired_repu, negative if not
        #     # repu_rewards[i] = np.tanh((0.5 - abs(r_act - desired_repu)) * scale)

        #     if (agent_coop and recipient_good) or (not agent_coop and not recipient_good):
        #         # Should assign good rep
        #         if assigned_good:
        #             repu_rewards_2[i] += 1
        #         else:
        #             repu_rewards_2[i] -= 1
        #     else:
        #         # Should assign bad rep
        #         if assigned_good:
        #             repu_rewards_2[i] -= 1
        #         else:
        #             repu_rewards_2[i] += 1

        # Extract agent cooperation and recipient reputation
        # agent_coop = (selected_last_last_repu[:, -2] == 0)     # shape (N,) boolean
        # recipient_good = (selected_last_last_repu[:, 1] == 1.0)  # shape (N,) boolean

        # # Compute desired reputation using SJ logic
        # desired_repu = np.logical_xor(agent_coop, ~recipient_good).astype(float)  # shape (N,)

        # # Compute reward
        # scale = 10
        # diff = np.abs(assign_repu - desired_repu)  # shape (N,)
        # repu_rewards = np.tanh((0.5 - diff) * scale)  # shape (N,)

        d_rewards = (d_rewards - d_rewards.mean()) / (d_rewards.std() + 1e-8)
        # d_rewards=2 * (d_rewards - d_rewards.min()) / (d_rewards.max() - d_rewards.min() + 1e-8) - 1

        scale = 10
        repu_rewards = d_rewards * np.tanh((assign_repu - 0.5) * scale)

        # repu_rewards=d_rewards

        # repu_rewards = d_rewards * (2 * assign_repu - 1)

        # assign_repu= (assign_repu >= 0.5).astype(int)

        # self_action = self_d_act.reshape(-1)
        # recipient_action = recipt_d_act.reshape(-1)
        # d_rewards = rewards.reshape(-1)
        # assign_repu  = self._last_dilemma_obs[..., -1].reshape(-1)  # equivalent to last repu action

        # repu_rewards = np.zeros_like(d_rewards)

        # for i, (repu_act,recip_act) in enumerate(zip(assign_repu, recipient_action)):
        #     repu_rewards[i] = abs(recip_act-repu_act)*2

        # if self._internal_timesteps>5:
        #     last_repu_act=np.take_along_axis(self._last_repu_actions, _dilemma_recipients_idx, axis=2).squeeze(-1)
        #     current_d_obs=self._last_dilemma_obs[..., -1].squeeze(-1)

        #     breakpoint()
        # breakpoint()
        # repu_rewards=(repu_rewards-repu_rewards.mean())/repu_rewards.std()+1e-8
        return repu_rewards.reshape(rewards.shape)
        # return rewards
