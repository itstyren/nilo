from ...algorithm.basePolicy import basePolicy
import warnings
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import torch.nn as nn
import torch.nn.functional as F
import torch
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from grg.algorithm.Extractor import CnnGRUExtractor
from typing import Any, Optional, TypeVar, Union
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from functools import partial
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


class DilemmaPolicy(basePolicy):
    def __init__(
        self,
        args,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        vf_observation_space: Optional[spaces.Space] = None,
    ) -> None:

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        self.args = args
        self.vf_observation_space = vf_observation_space

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            squash_output=squash_output,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        if net_arch is None:
            if features_extractor_class == CnnGRUExtractor:
                net_arch = []
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.share_features_extractor = share_features_extractor
        self.make_features_extractor()

        self.log_std_init = log_std_init
        dist_kwargs = None

        assert not (
            squash_output and not use_sde
        ), "squash_output=True is only available when using gSDE (use_sde=True)"
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs
        self.action_dist = make_proba_distribution(
            action_space, use_sde=use_sde, dist_kwargs=dist_kwargs
        )
        self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            [self.pi_features_dim, self.vf_features_dim],
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def make_critic_extractor(self) -> BaseFeaturesExtractor:
        return self.features_extractor_class(
            self.vf_observation_space, **self.features_extractor_kwargs
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(
            self.mlp_extractor.latent_dim_vf,
            1,
        )
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }

            del module_gains[self.features_extractor]
            module_gains[self.pi_features_extractor] = np.sqrt(2)
            module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def init_weights(self, module: nn.Module, gain: float = 1) -> None:
        pass

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        act_features = self.extract_features(obs, features_type="pi")
        latent_pi = self.mlp_extractor.forward_actor(act_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)

        values = []
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        mean_actions = self.action_net(latent_pi)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(
                mean_actions, self.log_std, latent_pi
            )
        else:
            raise ValueError("Invalid action distribution")

    def _get_action_logits(self, latent_pi: torch.Tensor) -> Distribution:
        mean_actions = self.action_net(latent_pi)
        return F.softmax(mean_actions, dim=-1)

    def extract_features(
        self,
        obs: PyTorchObs,
        features_type: str = "pi",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if features_type == "pi":
            features_extractor = self.pi_features_extractor
            feature = super().extract_features(obs, features_extractor)
        else:
            features_extractor = self.vf_features_extractor
            feature = super().extract_features(obs, features_extractor)

        return feature

    def evaluate_actions(
        self, obs: PyTorchObs, actions: torch.Tensor, recipient_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        obs = obs.reshape(-1,*self.observation_space.shape)
        act_features = self.extract_features(obs, features_type="pi")

        latent_pi = self.mlp_extractor.forward_actor(act_features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        if self.action_space.n>2:
            normalized_agent_actions = actions / (self.action_space.n-1)
            normalized_recipients_actions = recipient_actions / (self.action_space.n-1)
            critic_obs= obs.unsqueeze(-1).repeat(1, 1, 1, 1, 2)
            for b in range(obs.shape[0]):
                agent_pos = torch.nonzero(obs[b, 0]>0, as_tuple=False)

                if agent_pos.numel() > 0:
                    x, y = agent_pos[0]

                    critic_obs[b, 0, x, y, 1] = normalized_agent_actions[b]
                recipient_pos = torch.nonzero(obs[b, 2]>0, as_tuple=False)
                if recipient_pos.numel() > 0:
                    x, y = recipient_pos[0]
                    critic_obs[b, 2, x, y, 1] = normalized_recipients_actions[b]
            critic_obs=critic_obs.reshape(*critic_obs.shape[0:3], -1)
        else:
            actions = actions.view(-1, 1)
            recipient_actions = recipient_actions.view(-1, 1)

            critic_obs = torch.cat(
                [obs, actions, recipient_actions],
                dim=1,
            )

        vf_features = self.extract_features(critic_obs, features_type="vf")
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)

        return values, log_prob, entropy

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)

        return self.value_net(latent_vf)

    def evaluate_next_state(
        self, operated_obs, obs_to_predict_recipient_act, operacted_action
    ) -> torch.Tensor:
        with torch.no_grad():
            pi_features = self.extract_features(
                obs_to_predict_recipient_act, features_type="pi"
            )
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            distribution = self._get_action_dist_from_latent(latent_pi)
            actions_prob = self._get_action_logits(latent_pi)
            recipient_actions = distribution.get_actions()
            counter_recipient_actions = 1 - recipient_actions
            recipient_probs = actions_prob.gather(
                1, recipient_actions.unsqueeze(1)
            ).squeeze(
                1
            )
            counter_probs = actions_prob.gather(
                1, counter_recipient_actions.unsqueeze(1)
            ).squeeze(
                1
            )

            selected_probs = torch.stack(
                [recipient_probs, counter_probs], dim=1
            )

            _recipient_num = operated_obs.shape[-2]
            critic_obs = torch.cat(
                [
                    operated_obs,
                    operacted_action.view((-1, _recipient_num, 1)),
                    recipient_actions.view((-1, _recipient_num, 1)),
                ],
                dim=2,
            )

            values = self.predict_values(
                obs=critic_obs.reshape(-1, critic_obs.shape[-1])
            )

            critic_obs_revise = torch.cat(
                [
                    operated_obs,
                    operacted_action.view((-1, _recipient_num, 1)),
                    1 - recipient_actions.view((-1, _recipient_num, 1)),
                ],
                dim=2,
            ).to(dtype=torch.float64)

            values_revise = self.predict_values(
                obs=critic_obs_revise.reshape(-1, critic_obs_revise.shape[-1])
            )
        return torch.cat([values, values_revise], dim=1), selected_probs


    def evaluate_next_state_coin(
        self, operated_obs, obs_to_predict_recipient_act, operacted_action
    ) -> torch.Tensor:
        with torch.no_grad():

            pi_features = self.extract_features(
                obs_to_predict_recipient_act, features_type="pi"
            )
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            distribution = self._get_action_dist_from_latent(latent_pi)
            actions_prob = self._get_action_logits(latent_pi)
            recipient_actions = distribution.get_actions()

            all_actions = torch.arange(5, device=recipient_actions.device)

            recipient_act_exp = recipient_actions.unsqueeze(1)

            mask = all_actions.unsqueeze(0) != recipient_act_exp

            counter_recipient_actions = all_actions.expand(recipient_actions.shape[0], 5)[mask].reshape(recipient_actions.shape[0], 4)

            recipient_probs = actions_prob.gather(1, recipient_actions.unsqueeze(1)).squeeze(1)
            counter_probs = actions_prob.gather(1, counter_recipient_actions)

            selected_probs = torch.cat(
                [recipient_probs.unsqueeze(1), counter_probs], dim=1
            )

            selected_recipient_actions = torch.cat(
                [recipient_actions.unsqueeze(1), counter_recipient_actions], dim=1
            )

            max_actions =selected_recipient_actions.max()

            selected_recipient_actions=selected_recipient_actions/max_actions
            operacted_action=operacted_action/max_actions

            operated_obs_expand = np.repeat(
                operated_obs[..., np.newaxis], 2, axis=-1)
            all_possible_values=[]

            for action_idx in range(max_actions+1):
                temp_obs=operated_obs_expand.clone()
                for batch_idx,recipient_action in enumerate(selected_recipient_actions):
                    agent_position=torch.nonzero(
                            operated_obs_expand[batch_idx][0].any(dim=-1), as_tuple=False
                        ) 
                    recipient_position=torch.nonzero(
                            operated_obs_expand[batch_idx][2].any(dim=-1), as_tuple=False
                        )
                    temp_obs[batch_idx][0][tuple(agent_position[0])][1]=operacted_action[batch_idx]
                    temp_obs[batch_idx][2][tuple(recipient_position[0])][1]=recipient_action[action_idx]

                all_possible_values.append(self.predict_values(temp_obs.reshape(*temp_obs.shape[:-2], -1)).reshape(-1))

            all_possible_values=torch.stack(all_possible_values, dim=1)

        return all_possible_values, selected_probs


    def get_action_and_prob(
        self, obs: PyTorchObs, deterministic: bool = False
    ) -> torch.Tensor:
        with torch.no_grad():
            act_features = self.extract_features(obs, features_type="pi")
            latent_pi = self.mlp_extractor.forward_actor(act_features)
            distribution = self._get_action_dist_from_latent(latent_pi)
            actions_prob = self._get_action_logits(latent_pi)
            actions = distribution.get_actions(deterministic=deterministic)
        return actions, actions_prob


class DilemmaCnnPolicy(DilemmaPolicy):
    def __init__(
        self,
        args,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = CnnGRUExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        vf_observation_space: Optional[spaces.Space] = None,
    ):
        super().__init__(
            args,
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            optimizer_class,
            optimizer_kwargs,
            vf_observation_space,
        )