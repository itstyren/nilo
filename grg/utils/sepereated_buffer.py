from stable_baselines3.common.buffers import RolloutBuffer, BaseBuffer
from gymnasium import spaces
from typing import Union
import torch
import numpy as np
from collections.abc import Generator
from typing import Optional, Union
from stable_baselines3.common.vec_env import VecNormalize             

from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Protocol, SupportsFloat, Union
import torch as th



class CustomRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    rewards: th.Tensor
    recipient_actions: th.Tensor
    recipient_idx: th.Tensor
    last_repu_observations: th.Tensor
    last_dilemma_value_observations: th.Tensor


class CustomRolloutBuffer(RolloutBuffer):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        num_recipients: int = 1,
        num_agents: int = 1,
        buffer_type='dilemma',
        last_repu_obs_shape: tuple = (),
        last_dilemma_vale_obs_shape: tuple = (),
        norm_type: str = 'RL',
        buffer_index: int = 0,
    ):
        self._num_recipients = num_recipients
        self._value_dim = num_recipients
        self.buffer_index=buffer_index

        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.buffer_type = buffer_type


        self.last_repu_obs_shape = (num_agents,) + last_repu_obs_shape
        self.last_dilemma_vale_obs_shape = last_dilemma_vale_obs_shape

        

        if self.buffer_type == 'dilemma':
            self.obs_shape = (num_recipients,) + self.obs_shape if num_recipients!=None else self.obs_shape
            self.action_dim = self.action_dim * num_recipients if num_recipients!=None else self.action_dim
        else:
            if norm_type == 'attention':
                self.obs_shape = (num_agents,) + self.obs_shape
                self.action_dim = self.action_dim * num_agents
            else:
                self.obs_shape = (num_recipients,) + self.obs_shape if num_recipients!=None else self.obs_shape
                self.action_dim = self.action_dim * num_recipients if num_recipients!=None else self.action_dim
        self.original_buffer_size = buffer_size

        self.reset()

    def reset(self) -> None:
        """
        extend all list think agent may interact with more then one agent
        """
        if self.buffer_type == 'reputation':
            if self._num_recipients != None:
                self.recipients_index = np.zeros(
                    (self.buffer_size, self.n_envs,
                    self._num_recipients), dtype=np.float32
                )
            else:
                self.recipients_index = np.zeros(
                    (self.buffer_size, self.n_envs), dtype=np.float32
                )
            
        self.buffer_size = self.original_buffer_size
        # observations store current repu info when do dilemma action
        # use for update repu during dilemma training
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32
        )
        self.last_repu_observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.last_repu_obs_shape), dtype=np.float32
        )
        self.last_dilemma_value_observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.last_dilemma_vale_obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.recipient_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )   

        if self._num_recipients != None:
            self.rewards = np.zeros(
                (self.buffer_size, self.n_envs, self._num_recipients), dtype=np.float32
            )

            self.recipient_idx = np.zeros(
                (self.buffer_size, self.n_envs, self._num_recipients), dtype=np.float32
            )
            self.returns = np.zeros(
                (self.buffer_size, self.n_envs, self._num_recipients), dtype=np.float32
            )
            self.values = np.zeros(
                (self.buffer_size, self.n_envs, self._value_dim), dtype=np.float32
            )

            self.advantages = np.zeros(
                (self.buffer_size, self.n_envs, self._value_dim), dtype=np.float32
            )            
        else:
            self.rewards = np.zeros(
                (self.buffer_size, self.n_envs), dtype=np.float32
            )

            self.recipient_idx = np.zeros(
                (self.buffer_size, self.n_envs), dtype=np.float32
            )
            self.returns = np.zeros(
                (self.buffer_size, self.n_envs), dtype=np.float32
            )            
            self.values = np.zeros(
                (self.buffer_size, self.n_envs), dtype=np.float32
            )

            self.advantages = np.zeros(
                (self.buffer_size, self.n_envs), dtype=np.float32
            )  

        self.log_probs = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

        self.generator_ready = False

        BaseBuffer.reset(self)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        recipient_action: np.ndarray = np.array([]),
        recipients_index: np.ndarray = np.array([]),
        last_repu_observations: np.ndarray = np.array([]),
        last_dilemma_value_observations: np.ndarray = np.array([]),
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """

        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)
        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392


        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.last_repu_observations[self.pos] = np.array(
            last_repu_observations) if len(last_repu_observations) > 0 else None


        self.last_dilemma_value_observations[self.pos] = np.array(
            last_dilemma_value_observations) if len(last_dilemma_value_observations) > 0 else None
        
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)

        self.values[self.pos] = np.array(value)
        self.recipient_actions[self.pos] = np.array(
            recipient_action) if recipient_action.shape[-1] > 0 else None
        self.log_probs[self.pos] = np.array(log_prob)
        self.recipient_idx[self.pos] = np.array(
            recipients_index) if recipients_index.shape[-1] > 0 else None
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True



    def remove_current_pos(self):
        """
        Removes all content at the current position and reduces buffer_size by 1.
        """
        # breakpoint()
        if self.pos < self.buffer_size:
            self.observations[self.pos] = None
            self.actions[self.pos] = None
            self.rewards[self.pos] = None
            self.episode_starts[self.pos] = None
            self.values[self.pos] = None
            self.recipient_actions[self.pos] = None
            self.log_probs[self.pos] = None

        # Shift the remaining elements to fill the gap
        self.observations = np.delete(self.observations, self.pos, axis=0)
        self.actions = np.delete(self.actions, self.pos, axis=0)
        self.rewards = np.delete(self.rewards, self.pos, axis=0)
        self.episode_starts = np.delete(self.episode_starts, self.pos, axis=0)
        self.values = np.delete(self.values, self.pos, axis=0)
        self.recipient_actions = np.delete(
            self.recipient_actions, self.pos, axis=0)
        self.log_probs = np.delete(self.log_probs, self.pos, axis=0)
        self.advantages = np.delete(self.advantages, self.pos, axis=0)
        self.returns = np.delete(self.returns, self.pos, axis=0)

        # Reduce buffer size by 1
        self.buffer_size -= 1

        # Adjust position pointer
        if self.pos >= self.buffer_size:
            self.pos = max(0, self.buffer_size - 1)

        # Adjust full flag
        if self.pos < self.buffer_size:
            self.full = False
        # breakpoint()

    def update_episode_start(self, episode_start: np.ndarray) -> None:
        """
        Update the episode_start for the current timestep
        """
        self.episode_starts[self.pos - 1] = np.array(episode_start)

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """

        last_gae_lam = 0
        # iterate over reversed range (from the last one)
        for step in reversed(range(self.buffer_size)):
            # last rollouts setp, enter first
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            extended_next_non_terminal = np.expand_dims(
                next_non_terminal, axis=1
            ).repeat(next_values.shape[1], axis=1)
            # delta = r + gamma * V(s') - V(s)
            delta = (
                self.rewards[step]
                + self.gamma * next_values * extended_next_non_terminal
                - self.values[step]
            )

            # Advantage = delta + gamma * gae_lambda * next_non_terminal * Advantage
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * extended_next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def compute_td0_returns(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        for step in range(self.buffer_size):
            if step == self.buffer_size - 1:
                next_value = last_values
                next_non_terminal = 1.0 - dones.astype(np.float32)
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
            extended_next_non_terminal = np.expand_dims(
                next_non_terminal, axis=1
            ).repeat(next_value.shape[1], axis=1)

            self.returns[step] = self.rewards[step] + \
                self.gamma * next_value * extended_next_non_terminal
        self.advantages = self.returns - self.values

    def compute_td0_returns_coin_game(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        for step in range(self.buffer_size):
            if step == self.buffer_size - 1:
                next_value = last_values
                next_non_terminal = 1.0 - dones.astype(np.float32)
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.episode_starts[step + 1]

            self.returns[step] = self.rewards[step] + \
                self.gamma * next_value * next_non_terminal
        self.advantages = self.returns - self.values


    def compute_returns_and_advantage_coin_game(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """

        last_gae_lam = 0
        # iterate over reversed range (from the last one)
        for step in reversed(range(self.buffer_size)):
            # last rollouts setp, enter first
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]

            # delta = r + gamma * V(s') - V(s)
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )

            # Advantage = delta + gamma * gae_lambda * next_non_terminal * Advantage
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values




    def get(self, batch_size: Optional[int] = None) -> Generator[CustomRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # if self.buffer_type == 'dilemma':
        #     # print(self.observations.shape)
        #     # print("buffer type is dilemma ", self.buffer_index)
        #     # print(self.observations[6])
        #     # print(self.last_repu_observations.shape)
        #     # print(self.last_repu_observations[6][:,self.buffer_index,:])
        #     breakpoint()
            

        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "rewards",
                'recipient_actions',
                'recipient_idx',
                'last_repu_observations',
                'last_dilemma_value_observations',
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(
                    self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> CustomRolloutBufferSamples:

        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.rewards[batch_inds].flatten(),
            self.recipient_actions[batch_inds],
            self.recipient_idx[batch_inds],
            self.last_repu_observations[batch_inds],
            self.last_dilemma_value_observations[batch_inds],
        )
        return CustomRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class DictRolloutBuffer(CustomRolloutBuffer):
    observation_space: spaces.Dict
    obs_shape: dict[str, tuple[int, ...]]  # type: ignore[assignment]
    observations: dict[str, np.ndarray]  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(RolloutBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        assert isinstance(
            self.obs_shape, dict
        ), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros(
                (self.buffer_size, self.n_envs, *obs_input_shape), dtype=np.float32
            )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.values = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()
