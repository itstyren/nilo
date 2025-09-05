from ..baseAlgorithm import baseAlgorithm
from .policy import ActorCriticPolicy
from typing import Dict, Optional, Tuple, Type, Union
import numpy as np
import torch
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from typing import Any, Type
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
import warnings
from gymnasium import spaces
from torch.nn import functional as F


class PPO(baseAlgorithm):

    policy: ActorCriticPolicy

    def __init__(
        self,
        all_args,
        logger,
        env: Union[GymEnv, str],
        sub_env: str,
        policy_class: Union[str, Type[ActorCriticPolicy]],
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
    ):
        super(PPO, self).__init__(
            all_args=all_args,
            logger=logger,
            env=env,
            sub_env=sub_env,
            policy_class=policy_class,
            learning_rate=learning_rate,
            n_steps=n_steps,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=device,
        )

        self._sanity_check(normalize_advantage,n_steps,batch_size)

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        if _init_setup_model:
            self._setup_model()
    def _setup_model(self) -> None:
        super()._setup_model()


        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self,rollout_buffer,train_type="dilemma") -> Dict[str, float]:

        self.policy.set_training_mode(True)

        learning_rate=self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)

        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        train_infos = {}

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in rollout_buffer.get(self.batch_size):
                if isinstance(self.action_space, spaces.Discrete):

                    actions = rollout_data.actions.long().flatten()
                else:

                    actions = rollout_data.actions.view(-1, 1)


                if isinstance(self.action_space, spaces.Box):
                    rollout_obs = rollout_data.observations.reshape(-1, *rollout_data.observations.shape[-2:])
                else:
                    rollout_obs=rollout_data.observations.reshape(-1, rollout_data.observations.size(-1))
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_obs, actions)
                values = values.flatten()


                advantages = rollout_data.advantages


                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


                ratio = torch.exp(log_prob - rollout_data.old_log_prob)


                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()


                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)


                if self.clip_range_vf is None:

                    values_pred = values
                else:


                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())


                if entropy is None:

                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss


                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break


                self.policy.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(rollout_buffer.values.flatten(), rollout_buffer.returns.flatten())
        train_infos['learning_rate']=learning_rate
        train_infos['explained_variance'] = explained_var
        train_infos['entropy_loss'] = np.mean(entropy_losses)
        train_infos['policy_gradient_loss'] = np.mean(pg_losses)
        train_infos['approx_kl']=np.mean(approx_kl_divs)
        train_infos['clip_fraction']=np.mean(clip_fractions)
        train_infos['loss']=loss.item()
        train_infos['n_updates']=self._n_updates

        print('value loss:',np.mean(value_losses))

        return train_infos


    def _sanity_check(self,normalize_advantage,n_steps,batch_size):
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440" 
        if self.env is not None:


            buffer_size = self.env.num_envs * n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"

            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={n_steps} and n_envs={self.env.num_envs})"
                )