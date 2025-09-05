from ..baseAlgorithm import baseAlgorithm
from .policy import ReputationPolicy
from typing import Dict, Optional, Tuple, Type, Union
import numpy as np
import torch
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from typing import Any, Type
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
import warnings
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.utils import get_linear_fn

class Reputation(baseAlgorithm):
    policy: ReputationPolicy
    def __init__(
        self,
        all_args,
        logger,
        env: Union[GymEnv, str],
        sub_env: str,
        policy_class: Union[str, Type[ReputationPolicy]],
        policy_kwargs: Optional[Dict[str, Any]] = None,
        learning_rate: Union[float, Schedule] = 1e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        ent_coef_fraction: float = 1,
        ent_coef_initial_eps: float = 1,
        ent_coef_final_eps: float = 0.01,
    ):
        super(Reputation, self).__init__(
            all_args=all_args,
            logger=logger,
            env=env,
            sub_env=sub_env,
            policy_class=policy_class,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=device,
        )
        self._sanity_check(normalize_advantage, n_steps, batch_size)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.repu_baseline_weight = self.all_args.repu_baseline_weight
        self.ent_coef_fraction = ent_coef_fraction
        self.ent_coef_initial_eps = ent_coef_initial_eps
        self.ent_coef_final_eps = ent_coef_final_eps
        if _init_setup_model:
            self._setup_model()
    def _setup_model(self) -> None:
        super()._setup_model()
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
        self.ent_coef_schedule = get_linear_fn(
            self.ent_coef_initial_eps,
            self.ent_coef_final_eps,
            self.ent_coef_fraction,
        )
    def train(
        self,
        rollout_buffer,
        value_type="return",
        rewards_type="plain",
        dilemma_policy=None,
        self_index=None,
    ) -> Dict[str, float]:
        self.policy.set_training_mode(True)
        learning_rate = self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        n_epochs = self._update_n_epochs()
        ent_coef = self.ent_coef_schedule(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        train_infos = {}
        continue_training = True
        for epoch in range(n_epochs):
            approx_kl_divs = []
            for rollout_data in rollout_buffer.get(self.batch_size):
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                else:
                    actions = rollout_data.actions.view(-1, 1)
                rollout_obs = self._reshape_obs(rollout_data)
                current_q_value, log_prob, entropy, distribution = (
                    self.policy.evaluate_actions(rollout_obs, actions)
                )
                current_q_value = current_q_value.flatten()
                recipient_idx = rollout_data.recipient_idx.long()
                selected_q_value = (
                    current_q_value.reshape(*rollout_data.actions.shape)
                    .gather(1, recipient_idx)
                    .squeeze(-1)
                )
                selection_actions = (
                    rollout_data.actions.gather(1, recipient_idx)
                    .squeeze(-1)
                    .reshape(-1)
                )
                if False:
                    selected_log_prob = (
                        log_prob.reshape(*rollout_data.actions.shape)
                        .gather(1, recipient_idx)
                        .squeeze(-1)
                    )
                    selected_old_log_prob = (
                        rollout_data.old_log_prob.reshape(
                            *rollout_data.actions.shape)
                        .gather(1, recipient_idx)
                        .squeeze(-1)
                    )
                    ratio = torch.exp(selected_log_prob - selected_old_log_prob)
                    advantages = rollout_data.advantages.reshape(-1)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(
                        ratio, 1 - clip_range, 1 + clip_range
                    )
                    policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
                else:
                    advantages = (current_q_value).detach()
                    ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(
                        ratio, 1 - clip_range, 1 + clip_range
                    )
                    policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
                rewards = rollout_data.rewards
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)
                if self.clip_range_vf is None:
                    values_pred = selected_q_value
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        current_q_value - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                if value_type == "return":
                    returns = rollout_data.returns
                    d_returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                    scale = 10
                    repu_returns = d_returns * torch.tanh((selection_actions - 0.5) * scale)
                    value_loss = F.mse_loss(repu_returns, values_pred)
                else:
                    if rewards_type == "plain":
                        if rollout_obs.ndim >= 5:
                            current_repu_obs = rollout_data.last_repu_observations
                            d_rewards = self.policy.get_value_for_critic_2(rollout_obs, current_repu_obs, dilemma_policy,
                                                                           self_index, recipient_idx, self.repu_baseline_weight, rewards)
                        else:
                            current_repu_obs = rollout_data.last_repu_observations.reshape(
                                *rollout_data.last_repu_observations.shape[:-2], -1
                            )
                            d_rewards = self.policy.get_value_for_critic(
                                rollout_obs,
                                current_repu_obs,
                                dilemma_policy,
                                self_index,
                                recipient_idx,
                                self.repu_baseline_weight,
                                rewards,
                            ).detach()
                        scale = 10
                        repu_rewards = d_rewards * torch.tanh((selection_actions - 0.5) * scale)
                        values_pred = values_pred.reshape(-1)
                        value_loss = F.mse_loss(repu_rewards, values_pred)
                    elif rewards_type == "None":
                        value_loss = F.mse_loss(rewards, values_pred)
                value_losses.append(value_loss.item())
                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + ent_coef * entropy_loss + self.vf_coef * value_loss
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(
            rollout_buffer.values.flatten(), rollout_buffer.returns.flatten()
        )
        return {
            "learning_rate": learning_rate,
            "entropy_coef": ent_coef,
            "policy_gradient_loss": np.mean(pg_losses),
            "explained_variance": explained_var,
            "entropy_loss": np.mean(entropy_losses),
            "clip_fraction": np.mean(clip_fractions),
            "approx_kl": np.mean(approx_kl_divs),
            "n_updates": self._n_updates,
            "loss": loss.item(),
        }
    def _sanity_check(self, normalize_advantage, n_steps, batch_size):
        if normalize_advantage:
            assert (batch_size > 1)
        if self.env is not None:
            buffer_size = self.env.num_envs * n_steps
            assert buffer_size > 1 or (not normalize_advantage)
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size},"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={n_steps} and n_envs={self.env.num_envs})"
                )
    def _reshape_obs(self, rollout_data):
        obs = rollout_data.observations
        if obs.ndim >= 6:
            return obs.reshape(-1, *obs.shape[2:])
        else:
            return obs.reshape(-1, obs.shape[-2] * obs.shape[-1])
    def categorize_and_normalize_repu_rewards(self, roll_obs, repu_rewards):
        categories = roll_obs[..., -1]
        flat_categories = categories.reshape(-1)
        flat_rewards = repu_rewards.reshape(-1)
        mask_0 = flat_categories == 0
        mask_1 = flat_categories == 1
        normalized_flat = torch.zeros_like(flat_rewards)
        if mask_0.any():
            r0 = flat_rewards[mask_0]
            if r0.numel() > 1:
                std0 = r0.std(unbiased=False)
            else:
                std0 = 1.0
            normalized_flat[mask_0] = (r0 - r0.mean()) / (std0 + 1e-8)
        if mask_1.any():
            r1 = flat_rewards[mask_1]
            if r1.numel() > 1:
                std1 = r1.std(unbiased=False)
            else:
                std1 = 1.0
            normalized_flat[mask_1] = (r1 - r1.mean()) / (std1 + 1e-8)
        return normalized_flat