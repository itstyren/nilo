from ..baseAlgorithm import baseAlgorithm
from .policy import DilemmaPolicy
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


class Dilemma(baseAlgorithm):
    policy: DilemmaPolicy

    def __init__(
        self,
        all_args,
        logger,
        env: Union[GymEnv, str],
        sub_env: str,
        policy_class: Union[str, Type[DilemmaPolicy]],
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
        super(Dilemma, self).__init__(
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
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.dilemma_next_weight = all_args.dilemma_next_weight
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
        self, rollout_buffer, norm_pattern=None, self_index=None
    ) -> Dict[str, float]:
        self.policy.set_training_mode(True)
        learning_rate = self._update_learning_rate(self.policy.optimizer)
        n_epochs = self._update_n_epochs()
        ent_coef = self.ent_coef_schedule(self._current_progress_remaining)
        clip_range = self.clip_range(self._current_progress_remaining) 
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining) 

        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        act_adv_sum = [[], []]
        continue_training = True

        for epoch in range(n_epochs):
            approx_kl_divs = []
            for rollout_data in rollout_buffer.get(self.batch_size):
                actions, recipient_actions = self._format_actions(rollout_data)
                rollout_obs = self._reshape_obs(rollout_data)

                current_q_values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_obs, actions, recipient_actions
                )
                current_q_values = current_q_values.flatten()

                if self.action_space.n < 3:
                    if self.dilemma_next_weight == 0:
                        next_state_value=0
                    else:
                        repu_obs_before_dilemma = rollout_data.last_repu_observations.reshape(
                            *rollout_data.last_repu_observations.shape[:-2], -1
                        )
                        obs_to_predict_recipient_act, operated_obs = (
                            self.process_next_state_obs(
                                rollout_obs,
                                repu_obs_before_dilemma,
                                actions,
                                norm_pattern,
                                self_index,
                                recipient_actions,
                            )
                        )
                        with torch.no_grad():
                            operacted_action, _, _ = self.policy.forward(
                                operated_obs.reshape(-1, operated_obs.shape[-1])
                            )
                            new_values, new_action_prob = self.policy.evaluate_next_state(
                                operated_obs, obs_to_predict_recipient_act, operacted_action
                            )
                            next_state_value = torch.sum(new_values * new_action_prob, dim=1)

                    advantages = (
                        current_q_values + self.dilemma_next_weight * next_state_value
                    ).detach()
                else:
                    if self.dilemma_next_weight == 0:
                        next_state_value=0
                    else:
                        repu_obs_after_dilemma = rollout_data.last_repu_observations
                        obs_to_predict_recipient_act, operated_obs = (
                            self.process_next_state_obs_2(
                                rollout_obs,
                                repu_obs_after_dilemma,
                                actions,
                                norm_pattern,
                                self_index,
                                recipient_actions,
                            )
                        )
                        operated_obs=operated_obs.squeeze(1)
                        with torch.no_grad():
                            operacted_action, _, _ = self.policy.forward(
                            operated_obs
                            ) 
                            new_values, new_action_prob = self.policy.evaluate_next_state_coin(
                                operated_obs, obs_to_predict_recipient_act, operacted_action
                            )
                            next_state_value = torch.sum(new_values * new_action_prob, dim=1)

                    advantages = (
                        current_q_values +self.dilemma_next_weight * next_state_value
                    ).detach()

                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
                if self.action_space.n < 3:
                    act_adv_sum[0].append(
                        (
                            torch.sum(advantages[actions == 0])
                            / len(advantages[actions == 0])
                        ).detach()
                    )
                    act_adv_sum[1].append(
                        (
                            torch.sum(advantages[actions == 1])
                            / len(advantages[actions == 1])
                        ).detach()
                    )

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

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

                value_loss = F.mse_loss(rollout_data.returns, current_q_values)
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
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
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
        return {
            "learning_rate": learning_rate,
            "entropy_coef": ent_coef,
            "explained_variance": explained_variance(
                rollout_buffer.values.flatten(), rollout_buffer.returns.flatten()
            ),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "policy_gradient_loss": np.mean(pg_losses),
            "approx_kl": np.mean(approx_kl_divs),
            "clip_fraction": np.mean(clip_fractions),
            "loss": loss.item(),
            "n_updates": self._n_updates,
            "act_adv_sum": [safe_mean(x) for x in act_adv_sum],
        }

    def process_next_state_obs(
        self,
        rollout_obs,
        repu_obs_before_dilemma,
        actions,
        norm_pattern,
        self_index=None,
        recipient_actions=None,
    ):
        operated_obs = rollout_obs.detach().cpu().numpy()
        repu_obs_before_dilemma = repu_obs_before_dilemma.detach().cpu().numpy()

        actions = actions.reshape(-1, rollout_obs.shape[-2])
        recipient_actions = recipient_actions.reshape(-1, rollout_obs.shape[-2])

        obs_to_predict_recipient_act = []  
        repu_obs = []  

        for i, (dilemma_obs, dilemma_act, recip_act) in enumerate(
            zip(rollout_obs, actions, recipient_actions)
        ):
            if isinstance(norm_pattern, list):
                if len(norm_pattern) > 0:
                    update_repu = norm_pattern[int(dilemma_obs[0][-1])][dilemma_act[0]]
                else:
                    update_repu = np.random.choice([0.0, 1.0])

                operated_obs[i][0][0] = update_repu
                obs_to_predict_recipient_act.append(
                    [float(dilemma_obs[0][1]), update_repu]
                )
            else: 
                temp_obs = []
                num_agents = len(repu_obs_before_dilemma[0])
                agent_one_hot = np.eye(num_agents)[self_index]
                for d_obs, d_act, r_act in zip(dilemma_obs, dilemma_act, recip_act):
                    temp_obs.append(
                        np.concatenate([
                            [d_obs[0]],
                            [d_obs[-1]],
                            [d_act],
                            [r_act],
                            agent_one_hot
                        ])
                    )
                repu_obs.append(temp_obs)

        if not isinstance(norm_pattern, list):
            repu_obs = torch.stack(
                [torch.cat([torch.tensor(t) for t in outer]) for outer in repu_obs]
            ).numpy()

            clipped_repu_actions = None
            if hasattr(norm_pattern, "attention"):
                for i in range(len(repu_obs_before_dilemma)):
                    repu_obs_before_dilemma[i][self_index] = repu_obs[i]
                obs_for_predict = torch.tensor(repu_obs_before_dilemma).to(self.device)
            else:
                obs_for_predict = torch.tensor(repu_obs).to(self.device)

            with torch.no_grad():
                repu_actions, _, _ = norm_pattern.forward(obs_for_predict)
                repu_actions = repu_actions.cpu().numpy()
                if self.env.action_space["reputation"].__class__ == spaces.Box:
                    clipped_repu_actions = np.clip(
                        repu_actions,
                        self.env.action_space["reputation"].low,
                        self.env.action_space["reputation"].high,
                    )
                else:
                    clipped_repu_actions = repu_actions

            if hasattr(norm_pattern, "attention"):
                clipped_repu_actions = clipped_repu_actions.flatten().reshape(
                    len(repu_obs_before_dilemma), len(repu_obs_before_dilemma[0])
                )[:, self_index]

            for i, single_obs in enumerate(rollout_obs):
                for j, _obs in enumerate(single_obs):
                    operated_obs[i][j][0] = clipped_repu_actions[i]
                    obs_to_predict_recipient_act.append(
                        np.concatenate(
                            [
                                np.atleast_1d(float(rollout_obs[i][j][1])),
                                np.atleast_1d(clipped_repu_actions[i]),
                            ]
                        )
                    )

        obs_to_predict_recipient_act = np.array(obs_to_predict_recipient_act, dtype=np.float32)
        return torch.tensor(obs_to_predict_recipient_act).to(self.device), torch.tensor(
            operated_obs
        ).to(self.device)


    def process_next_state_obs_2(
        self,
        rollout_obs,
        repu_obs_after_dilemma,
        actions,
        norm_pattern,
        self_index=None,
        recipient_actions=None,
    ):
        operated_obs = rollout_obs.detach().cpu().numpy()
        repu_obs_after_dilemma = repu_obs_after_dilemma.detach().cpu().numpy()

        actions = actions/(self.action_space.n-1)
        recipient_actions/(self.action_space.n-1)

        obs_to_predict_recipient_act = []  
        repu_obs = []  
        for _obs in repu_obs_after_dilemma:
            temp_obs = _obs[self_index]
            repu_obs.append(temp_obs)

        repu_obs_array = np.array(repu_obs)  
        obs_for_predict = torch.tensor(repu_obs_array).to(self.device)
        obs_for_predict=obs_for_predict.reshape(*obs_for_predict.shape[:-2], -1)

        clipped_repu_actions = None

        with torch.no_grad():
            if self.action_space.n==5 and hasattr(norm_pattern, 'attention'):
                obs_for_predict= obs_for_predict.reshape(obs_for_predict.shape[0]//2, 2, *obs_for_predict.shape[1:])
            repu_actions, _, _ = norm_pattern.forward(obs_for_predict)
            repu_actions = repu_actions.cpu().numpy()
            clipped_repu_actions = repu_actions
            if self.action_space.n==5 and hasattr(norm_pattern, 'attention'):
                clipped_repu_actions=clipped_repu_actions.reshape(-1)

        for i, single_obs in enumerate(rollout_obs):
            agent_position=np.argwhere(operated_obs[i][0] != 0)
            operated_obs[i][0][tuple(
                agent_position[0])] = clipped_repu_actions[i] if clipped_repu_actions[i] > 0 else 0.5
            recipient_idx=1-self_index
            temp_recipient_obs = repu_obs_after_dilemma[i][recipient_idx][..., 0].copy()
            temp_recipient_obs[2][tuple(
                agent_position[0])]=clipped_repu_actions[i] if clipped_repu_actions[i] > 0 else 0.5
            obs_to_predict_recipient_act.append(
                temp_recipient_obs
            )
        obs_to_predict_recipient_act = np.array(obs_to_predict_recipient_act, dtype=np.float32)
        return torch.tensor(obs_to_predict_recipient_act).to(self.device), torch.tensor(
            operated_obs
        ).to(self.device)


    def _sanity_check(self, normalize_advantage, n_steps, batch_size):
        if normalize_advantage:
            assert batch_size > 1

        if self.env is not None:
            buffer_size = self.env.num_envs * n_steps
            assert (
                buffer_size > 1 or not normalize_advantage
            )
            if buffer_size % batch_size:
                warnings.warn(
                    f"Mini-batch size {batch_size} leads to truncated batch (size {buffer_size % batch_size}) from buffer size {buffer_size}"
                )

    def _format_actions(self, rollout_data):
        if isinstance(self.action_space, spaces.Discrete):
            return (
                rollout_data.actions.long().flatten(),
                rollout_data.recipient_actions.long().flatten(),
            )
        return rollout_data.actions.view(-1, 1), rollout_data.recipient_actions.view(
            -1, 1
        )

    def _reshape_obs(self, rollout_data):
        obs = rollout_data.observations
        return obs