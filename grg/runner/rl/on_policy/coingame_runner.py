from stable_baselines3.common.utils import should_collect_more_steps
import random
import time
import numpy as np
from ..baseCoin_runner import BaseRunner
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import sys
import grg.utils.tools as utils
from stable_baselines3.common.utils import get_schedule_fn
import imageio
from PIL import Image
from tqdm import tqdm
import os


class CoinGamenner(BaseRunner):

    def __init__(self, config):
        super(CoinGamenner, self).__init__(config)
        self.continue_training = True
        self.last_best_mean_payoff = -np.inf
        self.last_best_cooperation_level = -np.inf
        self.no_improvement_evals = 0
        self.max_no_improvement_evals = 10
        self.continue_training = True

    def run(self):
        self._warmup()

        iteration_episode = 0
        roullout_count = 0

        while self.num_timesteps < self._total_timesteps:
            continue_training = self.collect_rollouts(self.envs, self.n_steps)


            if not continue_training:
                break

            iteration_episode += self.n_rollout_threads
            roullout_count += 1

            progress_remaining = self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            if roullout_count % self._log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration_episode)

            if (
                roullout_count % self._save_interval == 0
                or iteration_episode == self._total_episodes - 1
            ):

                self.save_model()

            if (
                self.use_render
                and self.eval_env is not None
                and roullout_count % self._render_interval == 0
            ):

                self.render(iteration_episode)

            self.train_agent_policy(progress_remaining)

    def _warmup(self):
        if self._last_dilemma_obs is None:
            assert self.envs is not None, "Environment is not initialized"

            initial_obs = self.envs.reset()
            dilemma_obs = self.process_obs_to_specific_usage(initial_obs, "dilemma_obs")
            repu_obs = self.process_obs_to_specific_usage(initial_obs, "repu_obs")
            self._last_dilemma_obs = dilemma_obs.copy()
            self._last_last_repu_obs = repu_obs.copy()

            self.dilemma_updates_freq_schedule = get_schedule_fn(
                self.dilemma_train_freq
            )

            self._last_episode_starts = np.ones((self.envs.num_envs,), dtype=bool)


        self._total_episodes = int(self._total_timesteps) // self.episode_length


        self._internal_timesteps = 0

    def collect_rollouts(self, env: VecEnv, n_rollout_steps: int) -> bool:
        assert (
            self._last_dilemma_obs is not None
        ), "No previous observation was provided"


        n_steps = 0


        for agent_idx in range(self.num_agents):
            self.dilemma_trainers[agent_idx].policy.set_training_mode(False)
            self.repu_trainers[agent_idx].policy.set_training_mode(False)
            self.dilemma_buffers[agent_idx].reset()
            self.repu_buffers[agent_idx].reset()


        batch_indices = np.arange(self.n_rollout_threads)
        while n_steps < n_rollout_steps:

            dilemma_actions, dilemma_values, dilemma_log_probs = self.get_action(
                action_type="dilemma"
            )

            updated_obs, rewards, _, truncations, dilemma_infos = env.step(
                dilemma_actions, action_type="dilemma"
            )
            agent_recipients_dilemma_actions = self.get_other_agents_actions(
                dilemma_actions
            )
            if self.baseline_type != "NL":
                dilemma_values = self.get_dilemma_value(
                    dilemma_actions, agent_recipients_dilemma_actions
                )


            repu_obs = self.process_obs_to_specific_usage(updated_obs, "repu_obs")

            _dilemma_recipients_idx = np.array(
                [info["agent_recipients_idx"] for info in dilemma_infos]
            )

            self._last_repu_obs = repu_obs.copy()


            if self.norm_pattern == "RL" or self.norm_pattern == "attention":
                repu_actions, repu_values, repu_log_probs = self.get_action(
                    action_type="reputation"
                )
                repu_values = self.get_repu_value(repu_actions)

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

            repu_rewards = rewards.copy()
            self.num_timesteps += env.num_envs
            combined_infos = np.array(
                [{**d1, **d2} for d1, d2 in zip(dilemma_infos, repu_infos)],
                dtype=object,
            )


            self._update_info_buffer(combined_infos)


            if repu_infos[0]["epsidoe_end"]:

                dilemma_terminal_obs = self.process_obs_to_specific_usage(
                    next_dilemma_obs, "dilemma_obs"
                )
                dilemma_terminal_actions, _, _ = self.get_action(
                    action_type="dilemma", obs=dilemma_terminal_obs
                )

                new_episide_obs, _, _, truncations, terminal_infos = env.step(
                    dilemma_terminal_actions, action_type="dilemma"
                )

                agent_recipients_dilemma_actions = self.get_other_agents_actions(
                    dilemma_actions
                )


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


                rewards += self.gamma * dilemma_terminal_values
                if self.repu_value_flag == "return":
                    repu_terminal_values = self.get_repu_value(
                        repu_actions, repu_obs=repu_terminal_obs
                    )
                    repu_terminal_values = np.take_along_axis(
                        repu_terminal_values, _dilemma_recipients_idx, axis=2
                    )
                    repu_rewards += self.gamma * repu_terminal_values


            for agent_idx in range(self.num_agents):

                recipient_idx = _dilemma_recipients_idx[:, agent_idx]

                self.repu_buffers[agent_idx].add(


                    self._last_repu_obs[:, agent_idx],


                    repu_actions[:, agent_idx],


                    repu_rewards[:, agent_idx],
                    self._last_episode_starts,
                    repu_values[:, agent_idx][batch_indices, recipient_idx],
                    repu_log_probs[:, agent_idx],
                    recipients_index=recipient_idx,
                    last_repu_observations=self._last_repu_obs[:, agent_idx],
                    last_dilemma_value_observations=self._last_dilemma_value_obs[
                        :, agent_idx
                    ] if self.baseline_type != "NL" else [],
                )


            for agent_idx in range(self.num_agents):
                self.dilemma_buffers[agent_idx].add(
                    self._last_dilemma_obs[:, agent_idx],
                    dilemma_actions[:, agent_idx],
                    rewards[:, agent_idx],
                    self._last_episode_starts,
                    dilemma_values[:, agent_idx],
                    dilemma_log_probs[:, agent_idx],
                    recipient_action=agent_recipients_dilemma_actions[:, agent_idx],
                    last_repu_observations=self._last_repu_obs[:, agent_idx],
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


            last_dilemma_actions, _, _ = self.get_action(action_type="dilemma")
            _, _, _, _, last_dilemma_infos = env.step(
                last_dilemma_actions, action_type="dilemma", move_step=False
            )
            last_agent_recipients_dilemma_actions = self.get_other_agents_actions(
                dilemma_actions
            )
            last_dilemma_values = self.get_dilemma_value(
                last_dilemma_actions, last_agent_recipients_dilemma_actions
            )

            for agent_idx in range(self.num_agents):
                recipient_idx = _dilemma_recipients_idx[:, agent_idx]
                last_repu_value.append(
                    self._last_repu_values[:, agent_idx][batch_indices, recipient_idx]
                )

        last_dilemma_values = np.array(last_dilemma_values)
        last_repu_value = np.array(last_repu_value)

        for agent_idx in range(self.num_agents):

            self.dilemma_buffers[agent_idx].compute_td0_returns_coin_game(

                last_dilemma_values[:, agent_idx],
                dones=truncations,
            )

            if self.dilemma_train_freq != 1 and self.repu_value_flag == "return":

                self.repu_buffers[agent_idx].compute_returns_and_advantage_coin_game(
                    last_repu_value[agent_idx, :],
                    dones=truncations,
                )
        return True

    def process_obs_to_specific_usage(self, obs, usage_type, mode="train"):
        n_rollout_threads = (
            self.n_rollout_threads if mode == "train" else self.n_eval_rollout_threads
        )
        obs_set = np.empty((n_rollout_threads, self.num_agents), dtype=object)

        for rollout_idx in range(n_rollout_threads):
            for agent_idx in range(self.num_agents):
                obs_set[rollout_idx, agent_idx] = np.array(
                    obs[rollout_idx, agent_idx][usage_type], dtype=np.float64
                )

        return np.array(obs_set.tolist(), dtype=np.float64)

    def get_action(self, action_type="dilemma", obs=None, mode="train"):
        action_set, value_set, log_probs_set = [], [], []
        action_dim = 1 if action_type == "dilemma" else -1
        value_dim = -1
        trainer = (
            self.dilemma_trainers if action_type == "dilemma" else self.repu_trainers
        )
        _last_obs = (
            obs
            if obs is not None
            else (
                self._last_dilemma_obs
                if action_type == "dilemma"
                else self._last_repu_obs
            )
        )
        n_rollout_threads = (
            self.n_rollout_threads if mode == "train" else self.n_eval_rollout_threads
        )

        for agent_idx in range(self.num_agents):
            with torch.no_grad():
                agent_obs = _last_obs[:, agent_idx]


                if action_type == "reputation":
                    if self.norm_pattern == "RL":
                        obs_shape = agent_obs.shape[2:]
                        obs_shape = (
                            obs_shape[0],
                            obs_shape[1],
                            obs_shape[2] * obs_shape[3],
                        )

                        obs_tensor = obs_as_tensor(
                            agent_obs.reshape(-1, *obs_shape), self.device
                        )
                    elif self.norm_pattern == "attention":
                        obs_shape = agent_obs.shape[2:]
                        obs_shape = (
                            obs_shape[0],
                            obs_shape[1],
                            obs_shape[2] * obs_shape[3],
                        )
                        obs_tensor=obs_as_tensor(agent_obs.reshape(*agent_obs.shape[:-2], -1), self.device)

                    else:
                        obs_shape = agent_obs.shape[2:]
                        obs_shape = (
                            obs_shape[0],
                            obs_shape[1],
                            obs_shape[2] * obs_shape[3],
                        )
                        obs_tensor = obs_as_tensor(
                            agent_obs.reshape(1, 2, *obs_shape), self.device
                        )
                else:
                    obs_tensor = obs_as_tensor(agent_obs, self.device)


                actions, values, log_probs = trainer[agent_idx].policy.forward(
                    obs_tensor
                )


                if (
                    action_type == "reputation"
                    and self.action_space["reputation"].__class__ == spaces.Box
                ):
                    actions = actions.squeeze(1)


            actions = utils.t2n(actions, n_rollout_threads, action_dim)
            log_probs = utils.t2n(log_probs, n_rollout_threads, action_dim)
            if self.baseline_type == "NL" and action_type == "dilemma":
                values=values.squeeze(1)

            values = (
                utils.t2n(values, n_rollout_threads)
                if len(values) > 1
                else np.array(values)
            )


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
        )

    def get_dilemma_value(
        self,
        dilemma_actions,
        recipients_dilemma_actions,
        dilemma_obs=None,
    ):

        dilemma_obs = self._last_dilemma_obs if dilemma_obs is None else dilemma_obs


        normalized_agent_actions = dilemma_actions / self.max_dilemma_action_value
        normalized_recipients_actions = (
            recipients_dilemma_actions / self.max_dilemma_action_value
        )

        dilemma_obs_expand = np.repeat(dilemma_obs[..., np.newaxis], 2, axis=-1)


        for bath_indx in range(dilemma_obs.shape[0]):
            for agent_index in range(dilemma_obs.shape[1]):
                agent_position = np.argwhere(
                    dilemma_obs[bath_indx, agent_index][0] != 0
                )
                dilemma_obs_expand[bath_indx, agent_index][0][tuple(agent_position[0])][
                    1
                ] = normalized_agent_actions[bath_indx, agent_index]
                recipient_position = np.argwhere(
                    dilemma_obs[bath_indx, agent_index][2] != 0
                )
                dilemma_obs_expand[bath_indx, agent_index][2][
                    tuple(recipient_position[0])
                ][1] = normalized_recipients_actions[bath_indx, agent_index]

        self._last_dilemma_value_obs = dilemma_obs_expand

        value_set = []

        original_shape = dilemma_obs_expand[:, 0].shape
        new_shape = original_shape[:-2] + (original_shape[-2] * original_shape[-1],)

        for agent_idx in range(self.num_agents):
            with torch.no_grad():
                if self.baseline_type != "NL":
                    agent_obs = dilemma_obs_expand[
                        :, agent_idx
                    ] 


                    critic_obs_tensor = self.dilemma_trainers[
                        agent_idx
                    ].policy.obs_to_tensor(
                        agent_obs.reshape(new_shape),
                        observation_type="value_pred",
                    )[
                        0
                    ]
                else:
                    agent_obs=dilemma_obs[:, agent_idx]
                    critic_obs_tensor=self.dilemma_trainers[agent_idx].policy.obs_to_tensor(
                        agent_obs, observation_type="value_pred")[0]


                values = self.dilemma_trainers[agent_idx].policy.predict_values(
                    critic_obs_tensor
                )
            values = values.squeeze(1)

            values = utils.t2n(values, self.n_rollout_threads)
            value_set.append(values)
        return np.stack(value_set, axis=1)


    def get_repu_value(
        self,
        repu_actions,
        repu_obs=None,
    ):
        repu_obs = (
            self._last_repu_obs if repu_obs is None else repu_obs
        )
        extended_obs = np.pad(
            repu_obs,
            pad_width=[(0, 0)] * 6 + [(0, 1)],
            mode="constant",
            constant_values=0,
        )
        for bath_indx in range(repu_obs.shape[0]):
            for agent_index in range(repu_obs.shape[1]):
                for obs_agent_idx in range(repu_obs.shape[2]):
                    agent_within_map = extended_obs[bath_indx, agent_index][
                        obs_agent_idx
                    ][0]


                    for idx in np.ndindex(agent_within_map.shape[:2]):
                        vec = agent_within_map[idx]
                        if np.any(vec):

                            if vec[-1] == 0:
                                extended_obs[bath_indx, agent_index][obs_agent_idx][0][
                                    idx
                                ][-1] = repu_actions[bath_indx, agent_index][
                                    obs_agent_idx
                                ]

        value_set = []

        with torch.no_grad():
            for agent_idx in range(self.num_agents):


                critic_obs = extended_obs[:, agent_idx, :, :]
                obs_shape = (
                    *critic_obs.shape[-4:-2],
                    critic_obs.shape[-1] * critic_obs.shape[-1],
                )


                critic_obs = critic_obs.reshape(-1, *obs_shape)
                critic_obs_tensor = self.repu_trainers[agent_idx].policy.obs_to_tensor(
                    critic_obs, observation_type="value_pred"
                )[0]


                values = self.repu_trainers[agent_idx].policy.predict_values(
                    critic_obs_tensor
                )


                values = utils.t2n(values, self.n_rollout_threads, -1)

                value_set.append(values)

        return np.stack(value_set, axis=1)


    def get_other_agents_actions(self, dilemma_actions):
        batch_size, num_agents, _ = dilemma_actions.shape


        dilemma_actions = np.squeeze(dilemma_actions, axis=-1)

        other_actions = []

        for agent_idx in range(num_agents):
            mask = np.arange(num_agents) != agent_idx


            other_agent_actions = dilemma_actions[:, mask]
            other_actions.append(other_agent_actions)


        other_actions = np.stack(other_actions, axis=1)

        return other_actions

    def train_agent_policy(self, progress_remaining):
        self.train_freq = self.dilemma_updates_freq_schedule(progress_remaining)

        if random.random() < self.train_freq:
            for agent_idx in range(self.num_agents):
                self.dilemma_trainers[agent_idx]._current_progress_remaining = (
                    progress_remaining
                )

                dilemma_train_info = self.dilemma_trainers[agent_idx].train(
                    self.dilemma_buffers[agent_idx],
                    norm_pattern=self.repu_trainers[agent_idx].policy,
                    self_index=agent_idx,
                )
                self.dilemma_train_info_buffer.extend([dilemma_train_info])
        else:
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

    def _dump_logs(self, iteration_episode):
        assert self.ep_info_buffer is not None

        logger_info = {}

        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)

        logger_info["time/fps"] = fps
        logger_info["time/time_elapsed"] = time_elapsed
        logger_info["time/total_timesteps"] = self.num_timesteps
        logger_info["time/episode"] = iteration_episode
        logger_info["time/dilemma_train_freq"] = self.train_freq

        print(
            "\n Dillemma_T({})S({}) Exp {} Substrate {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.all_args.dilemma_T,
                self.all_args.dilemma_S,
                self.exp_name,
                self.substrate_name,
                iteration_episode,
                self._total_episodes,
                self.num_timesteps,
                self._total_timesteps,
                fps,
            )
        )


        logger_info["results/coin_taken_by_others_episode"] = np.sum(
            [ep_info["coin_taken_by_others"] for ep_info in self.ep_info_buffer]
        )
        logger_info["results/coins_collected_episode"] = np.sum(
            [ep_info["total_coins_collected"] for ep_info in self.ep_info_buffer]
        )
        logger_info["results/own_coin_collected_episode"] = np.sum(
            [ep_info["own_coin_collected"] for ep_info in self.ep_info_buffer]
        )

        logger_info['results/agent_1_own_coin_collected_episode'] = np.sum(
           [ep_info["own_coin_collected"][0] for ep_info in self.ep_info_buffer]
        )
        logger_info['results/agent_2_own_coin_collected_episode'] = np.sum(
           [ep_info["own_coin_collected"][1] for ep_info in self.ep_info_buffer]
        )
        logger_info['results/agent_1_coins_collected_episode'] = np.sum(
              [ep_info["total_coins_collected"][0] for ep_info in self.ep_info_buffer]
          )
        logger_info['results/agent_2_coins_collected_episode'] = np.sum(
                [ep_info["total_coins_collected"][1] for ep_info in self.ep_info_buffer]
            )
        logger_info["results/agent_1_take_other_coins_episode"] = np.sum(
            [ep_info["coin_taken_by_others"][1] for ep_info in self.ep_info_buffer]
        )
        logger_info["results/agent_2_take_other_coins_episode"] = np.sum(
            [ep_info["coin_taken_by_others"][0] for ep_info in self.ep_info_buffer]
        )
        logger_info["results/agent_1_own_coin_proportion_episode"] = safe_mean(
            [
                ep_info["own_coin_collected"][0] / ep_info["total_coins_collected"][0]
                for ep_info in self.ep_info_buffer
                if ep_info["total_coins_collected"][0] > 0
            ]
        )
        logger_info["results/agent_2_own_coin_proportion_episode"] = safe_mean(
            [
                ep_info["own_coin_collected"][1] / ep_info["total_coins_collected"][1]
                for ep_info in self.ep_info_buffer
                if ep_info["total_coins_collected"][1] > 0
            ]
        )
        logger_info["results/agent_1_episode_rewards"] = np.sum(
            [ep_info["step_rewards"][0] for ep_info in self.ep_info_buffer]
        )
        logger_info["results/agent_2_episode_reawrds"] = np.sum(
            [ep_info["step_rewards"][1] for ep_info in self.ep_info_buffer]
        )

        logger_info["results/own_coin_proportion"] = safe_mean(
            [
                sum(ep_info["own_coin_collected"])
                / sum(ep_info["total_coins_collected"])
                for ep_info in self.ep_info_buffer
                if sum(ep_info["total_coins_collected"]) > 0
            ]
        )

        logger_info["results/episode_rewards"] = np.sum(
            [ep_info["step_rewards"].mean() for ep_info in self.ep_info_buffer]
        )

        logger_info["reputation/average_reputation"] = safe_mean(
            [ep_info["average_reputation"] for ep_info in self.ep_info_buffer]
        )
        logger_info["reputation/own_coin_collected_repu"] = safe_mean(
            [ep_info["own_coin_collected_repu"] for ep_info in self.ep_info_buffer]
        )
        logger_info["reputation/collected_other_coin_repu"] = safe_mean(
            [ep_info["collected_other_coin_repu"] for ep_info in self.ep_info_buffer]
        )


        logger_info["dilemma_train/learning_rate"] = safe_mean(
            [ep_info["learning_rate"] for ep_info in self.dilemma_train_info_buffer]
        )
        logger_info["dilemma_train/entropy_loss"] = safe_mean(
            [ep_info["entropy_loss"] for ep_info in self.dilemma_train_info_buffer]
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
        logger_info["dilemma_train/value_loss"] = safe_mean(
            [ep_info["value_loss"] for ep_info in self.dilemma_train_info_buffer]
        )
        logger_info["dilemma_train/n_updates"] = safe_mean(
            [ep_info["n_updates"] for ep_info in self.dilemma_train_info_buffer]
        )
        logger_info["dilemma_train/entropy_coef"] = safe_mean(
            [ep_info["entropy_coef"] for ep_info in self.dilemma_train_info_buffer]
        )


        logger_info["repu_train/learning_rate"] = safe_mean(
            [ep_info["learning_rate"] for ep_info in self.repu_train_info_buffer]
        )
        logger_info["repu_train/policy_gradient_loss"] = safe_mean(
            [ep_info["policy_gradient_loss"] for ep_info in self.repu_train_info_buffer]
        )
        logger_info["repu_train/explained_variance"] = safe_mean(
            [ep_info["explained_variance"] for ep_info in self.repu_train_info_buffer]
        )
        logger_info["repu_train/loss"] = safe_mean(
            [ep_info["loss"] for ep_info in self.repu_train_info_buffer]
        )
        logger_info["repu_train/n_updates"] = safe_mean(
            [ep_info["n_updates"] for ep_info in self.repu_train_info_buffer]
        )
        logger_info["repu_train/entropy_coef"] = safe_mean(
            [ep_info["entropy_coef"] for ep_info in self.repu_train_info_buffer]
        )

        if not self.use_wandb:
            self.print_episode_stats(logger_info)
        else:
            self.log_metrics(logger_info, iteration_episode)

    @torch.no_grad()
    def render(self,iteration_episode):
        all_frames = []
        env = self.envs if self.eval_env is None else self.eval_env
        for episode in tqdm(
            range(self.all_args.render_episodes), desc="Rendering Episodes"
        ):
            episode_rewards = []
            initial_obs = env.reset()
            dilemma_obs = self.process_obs_to_specific_usage(
                initial_obs, "dilemma_obs", mode="eval"
            )

            eval_dilemma_obs = dilemma_obs.copy()
            if self.all_args.save_gifs:
                image = env.render("rgb_array")
                all_frames.append(image)
            else:
                env.render()

            for step in tqdm(
                range(self.episode_length), desc=f"Episode {episode+1}", leave=False
            ):

                calc_start = time.time()


                dilemma_actions, dilemma_values, dilemma_log_probs = self.get_action(
                    action_type="dilemma", obs=eval_dilemma_obs, mode="eval"
                )

                updated_obs, rewards, _, truncations, dilemma_infos = env.step(
                    dilemma_actions, action_type="dilemma"
                )

                repu_obs = self.process_obs_to_specific_usage(
                    updated_obs, "repu_obs", mode="eval"
                )

                eval_repu_obs = repu_obs.copy()
                repu_actions, repu_values, repu_log_probs = self.get_action(
                    action_type="reputation", obs=eval_repu_obs, mode="eval"
                )
                next_dilemma_obs, _, _, _, repu_infos = env.step(
                    repu_actions, action_type="reputaiton"
                )
                dilemma_obs = self.process_obs_to_specific_usage(
                    next_dilemma_obs, "dilemma_obs", mode="eval"
                )
                eval_dilemma_obs = dilemma_obs.copy()

                episode_rewards.append(rewards)

                if self.all_args.save_gifs:
                    image = env.render("rgb_array")
                    all_frames.append(image)
                else:
                    env.render()
                calc_end = time.time()

                elapsed = calc_end - calc_start
                if elapsed < self.all_args.ifi and not self.all_args.save_gifs:
                    time.sleep(self.all_args.ifi - elapsed)

            if not self.use_wandb:
                episode_rewards = np.array(episode_rewards)
                for agent_id in range(self.num_agents):
                    average_episode_rewards = np.mean(
                        np.sum(episode_rewards[:, :, agent_id], axis=0)
                    )
                    print(
                        "eval average episode rewards of agent%i: " % agent_id
                        + str(average_episode_rewards)
                    )

        if self.all_args.save_gifs:


            fps = self.all_args.ifi
            target_size = (512, 560)


            video_filename = f"Episode_{iteration_episode}.mp4"
            video_path = os.path.join(self.gif_dir, video_filename)

            with imageio.get_writer(video_path, fps=fps) as writer:
                for frame in all_frames:
                    resized = Image.fromarray(frame).resize(target_size, Image.BILINEAR)
                    writer.append_data(np.array(resized))
            if self.use_wandb:
                self.log_video(
                    video_path,
                    iteration_episode
                )