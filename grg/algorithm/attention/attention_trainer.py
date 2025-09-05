from ..baseAlgorithm import baseAlgorithm
from .policy import MultiHeadReputationPolicy
import torch
import copy
import torch.nn.functional as F
from stable_baselines3.common.utils import safe_mean
from typing import Any, Dict, Optional, Type, Union
from stable_baselines3.common.type_aliases import GymEnv, Schedule
import numpy as np
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from gymnasium import spaces
from stable_baselines3.common.utils import get_linear_fn


class RepuAttention(baseAlgorithm):
    policy: MultiHeadReputationPolicy

    def __init__(
        self,
        all_args,
        logger,
        env: Union[GymEnv, str],
        sub_env: str,
        policy_class: Union[str, Type[MultiHeadReputationPolicy]],
        policy_kwargs: Optional[Dict[str, Any]] = None,
        learning_rate: Union[float, Schedule] = 1e-3,
        n_steps: int = 128,
        batch_size: int = 32,
        n_epochs: int = 4,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        device: Union[str, torch.device] = "auto",
        _init_setup_model: bool = True,
        ent_coef_fraction: float = 1,
        ent_coef_initial_eps: float = 1,
        ent_coef_final_eps: float = 0.01,
    ):
        super().__init__(
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

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.repu_baseline_weight = self.all_args.repu_baseline_weight
        self.repu_rewards_type = self.all_args.repu_rewards_type

        self.ent_coef_fraction = ent_coef_fraction
        self.ent_coef_initial_eps = ent_coef_initial_eps
        self.ent_coef_final_eps = ent_coef_final_eps

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        super()._setup_model()

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        self.ent_coef_schedule = get_linear_fn(
            self.ent_coef_initial_eps,
            self.ent_coef_final_eps,
            self.ent_coef_fraction,
        )

    def train(
        self, rollout_buffer, value_type, rewards_type, dilemma_policy, self_index
    ) -> Dict[str, float]:
        self.policy.set_training_mode(True)
        learning_rate = self._update_learning_rate(self.policy.optimizer)
        n_epochs = self._update_n_epochs()
        ent_coef = self.ent_coef_schedule(self._current_progress_remaining)
        clip_range = self.clip_range(
            self._current_progress_remaining
        )

        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining
            )

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        approx_kl_divs = []
        act_adv_sum = [[[], []], [[], []]]
        continue_training = True

        for epoch in range(n_epochs):
            for rollout_data in rollout_buffer.get(self.batch_size):
                rollout_obs = rollout_data.observations.reshape(
                    *rollout_data.observations.shape[:-2], -1
                )
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long()
                else:
                    actions = rollout_data.actions
                current_q_values, log_probs, entropy, distribution = (
                    self.policy.evaluate_unmasked(rollout_obs, actions)
                )

                recipient_idx = rollout_data.recipient_idx.long()
                selected_q_value = current_q_values.gather(
                    1, recipient_idx).squeeze(-1)
                selected_log_prob = log_probs.gather(
                    1, recipient_idx).squeeze(-1)
                selected_entropy = entropy.gather(1, recipient_idx).squeeze(-1)

                log_probs = log_probs.reshape(-1)
                selection_actions = (
                    actions.gather(1, recipient_idx).squeeze(-1).reshape(-1)
                )

                current_repu_obs = rollout_data.last_repu_observations.reshape(
                    *rollout_data.last_repu_observations.shape[:-2], -1
                )

                rewards = rollout_data.rewards

                with torch.no_grad():
                    advantages = current_q_values.clone()
                    advantages = (
                        advantages ).reshape(-1).detach()

                ratio = torch.exp(log_probs - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = selected_q_value
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        selected_q_value - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )

                if value_type == "return":
                    returns = rollout_data.returns
                    d_returns = (returns - returns.mean()) / \
                        (returns.std() + 1e-8)
                    scale = 10
                    repu_returns = d_returns * \
                        torch.tanh((selection_actions - 0.5) * scale)
                    value_loss = F.mse_loss(repu_returns, values_pred)
                else:
                    if rewards_type == 'plain':
                        if rollout_obs.ndim >= 5:
                            current_repu_obs = rollout_data.last_repu_observations

                            d_rewards = self.policy.get_value_for_critic_2(rollout_obs, current_repu_obs, dilemma_policy,
                                                                           self_index, recipient_idx, self.repu_baseline_weight, rewards)
                        else:
                            d_rewards = self.policy.get_value_for_critic(rollout_obs, current_repu_obs, dilemma_policy,
                                                                            self_index, recipient_idx, self.repu_baseline_weight, rewards).reshape(-1).detach()
                        scale = 10
                        repu_rewards = d_rewards * \
                            torch.tanh((selection_actions - 0.5) * scale)
                        values_pred = values_pred.reshape(-1)

                        value_loss = F.mse_loss(repu_rewards, values_pred)

                    elif rewards_type == 'None':
                        value_loss = F.mse_loss(rewards, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_probs)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                total_loss = (
                    policy_loss + self.vf_coef * value_loss + ent_coef * entropy_loss
                )

                with torch.no_grad():
                    log_ratio = log_probs - rollout_data.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) -
                                   log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                total_loss.backward()
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
            "loss": total_loss.item(),
            "act_adv_sum": [[safe_mean(x) for x in row] for row in act_adv_sum],
        }

    def compute_selected_avg_advantages(self, advantages, rollout_obs, acts):
        categories = rollout_obs[:, :, 0]

        mask_c = categories == 0
        mask_d = categories == 1

        mask_c_good = mask_c & (acts > 0.6)
        adv_c_good = advantages[mask_c_good]
        avg_adv_c_good = adv_c_good.mean().item() if adv_c_good.numel() > 0 else 0.0

        mask_d_bad = mask_d & (acts < 0.4)
        adv_d_bad = advantages[mask_d_bad]
        avg_adv_d_bad = adv_d_bad.mean().item() if adv_d_bad.numel() > 0 else 0.0

        mask_c_bad = mask_c & (acts < 0.4)
        adv_c_bad = advantages[mask_c_bad]
        avg_adv_c_bad = adv_c_bad.mean().item() if adv_c_bad.numel() > 0 else 0.0

        mask_d_good = mask_d & (acts > 0.6)
        adv_d_good = advantages[mask_d_good]
        avg_adv_d_good = adv_d_good.mean().item() if adv_d_good.numel() > 0 else 0.0

        return avg_adv_c_good, avg_adv_d_bad, avg_adv_c_bad, avg_adv_d_good

    def normalize_advantages_by_category(self, advantages, rollout_obs, eps=1e-8):
        categories = rollout_obs[..., 0]
        norm_adv = torch.zeros_like(advantages)

        for cat in [0, 1]:
            mask = categories == cat
            if mask.sum() > 0:
                group_adv = advantages[mask]
                mean = group_adv.mean()
                std = group_adv.std(unbiased=False)
                norm_adv[mask] = (group_adv - mean) / (std + eps)

        return norm_adv