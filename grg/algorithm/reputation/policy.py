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
from grg.algorithm.Extractor import CnnGRUExtractor


class ReputationPolicy(basePolicy):
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

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


        act_features = self.extract_features(obs, features_type="pi")

        latent_pi = self.mlp_extractor.forward_actor(act_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        actions = actions.reshape((-1, *self.action_space.shape))

        values = []

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
        self, obs: PyTorchObs, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if obs.ndim > 4:
            temp_obs = obs.clone()
            critic_obs = F.pad(temp_obs, (0, 1))

            for batch_idx in range(obs.shape[0]):
                agent_position=critic_obs[batch_idx][0]

                for idx in np.ndindex(agent_position.shape[:2]):
                    vec = agent_position[idx]
                    if vec[0] >0:
                        critic_obs[batch_idx][0][idx][-1] = actions[batch_idx]
            critic_obs= critic_obs.view(*critic_obs.shape[:-2], -1)
            obs = obs.view(*obs.shape[:-2], -1)
        else:
            critic_obs = torch.cat(
            [obs, actions.view(-1, 1)],
            dim=1,
        )       


        act_features = self.extract_features(obs, features_type="pi")

        latent_pi = self.mlp_extractor.forward_actor(act_features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        vf_features = self.extract_features(critic_obs, features_type="vf")
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)

        return values, log_prob, entropy, distribution

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def evaluate_baseline(self, obs: PyTorchObs) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            pi_features = self.extract_features(obs, features_type="pi")
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            distribution = self._get_action_dist_from_latent(latent_pi)
            actions_prob = self._get_action_logits(latent_pi)
            possible_action = distribution.get_actions()
            counter_actions = 1 - possible_action

            recipient_probs = actions_prob.gather(
                1, possible_action.unsqueeze(1)
            ).squeeze(
                1
            ) 
            counter_probs = actions_prob.gather(
                1, counter_actions.unsqueeze(1)
            ).squeeze(
                1
            ) 

            selected_probs = torch.stack(
                [recipient_probs, counter_probs], dim=1
            ) 

            possible_obs = torch.cat([obs, possible_action.unsqueeze(-1)], dim=-1).to(
                dtype=torch.float64
            )
            values = self.predict_values(obs=possible_obs)

            counterfacl_obs = torch.cat(
                [obs, (1 - possible_action).unsqueeze(-1)], dim=-1
            ).to(dtype=torch.float64)
            counterfacl_value = self.predict_values(counterfacl_obs)

            mean_baseline = (
                torch.cat([values, counterfacl_value], dim=1) * selected_probs
            ).sum(dim=1)

        return mean_baseline

    def _get_action_logits(self, latent_pi: torch.Tensor) -> Distribution:
        mean_actions = self.action_net(latent_pi)
        return F.softmax(mean_actions, dim=-1)

    def get_value_for_critic(
        self,
        repu_obs_for_act: PyTorchObs,
        repu_obs_after_dilemma: PyTorchObs,
        dilemma_policy: basePolicy,
        self_index: int,
        recipient_idx: int,
        repu_baseline_weight: float = 0.5,
        rewards: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N = repu_obs_after_dilemma.shape[:2]

        with torch.no_grad():
            obs_self = repu_obs_after_dilemma[:, self_index, :].view(B, -1, 4 + N)
            obs_predict_value = obs_self[..., :4].reshape(-1, 4)

            obs_opponent_act = obs_self[..., :2].reshape(-1, 2)[:, [1, 0]]

            recipient_act, dilemma_probs = dilemma_policy.get_action_and_prob(
                obs_opponent_act
            )
            counterfact_act = 1 - recipient_act

            dilemma_prob_actual = dilemma_probs.gather(
                1, recipient_act.unsqueeze(-1)
            ).squeeze(-1)
            dilemma_prob_counter = dilemma_probs.gather(
                1, counterfact_act.unsqueeze(-1)
            ).squeeze(-1)

            selected_dilemma_probs = torch.stack(
                [dilemma_prob_actual, dilemma_prob_counter], dim=1
            )
            selected_recipient_actions = torch.stack(
                [recipient_act, counterfact_act], dim=1
            )

            temp_obs = obs_predict_value.unsqueeze(1).repeat(1, 2, 1)
            temp_obs[..., -1] = selected_recipient_actions

            dilemma_values = dilemma_policy.predict_values(temp_obs.view(-1, 4)).view(-1, 2)

            baseline_value = (selected_dilemma_probs * dilemma_values).sum(dim=1)

            predict_value = dilemma_policy.predict_values(
                obs=obs_predict_value
            ).reshape(-1)

            final_value = predict_value - repu_baseline_weight * baseline_value

        final_value = (final_value - final_value.mean()) / (final_value.std() + 1e-8)
        return final_value


    def get_value_for_critic_2(
        self,
        repu_obs_for_act: PyTorchObs,
        repu_obs_after_dilemma: PyTorchObs,
        dilemma_policy: basePolicy,
        self_index: int,
        recipient_idx: int,
        repu_baseline_weight: float = 0.5,
        rewards: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():

            last_dilemma_value_obs=repu_obs_after_dilemma[:, self_index, :]
            obs_predict_value = last_dilemma_value_obs.reshape(*last_dilemma_value_obs.shape[:-2], -1)

            obs_opponent_act = last_dilemma_value_obs[..., 0][:, [2, 3, 0, 1], :, :]


            recipient_act, dilemma_probs = dilemma_policy.get_action_and_prob(
                obs_opponent_act
            )
            all_actions = torch.arange(5, device=recipient_act.device) 

            recipient_act_exp = recipient_act.unsqueeze(1) 

            mask = all_actions.unsqueeze(0) != recipient_act_exp 

            counterfact_act = all_actions.expand(recipient_act.shape[0], 5)[mask].reshape(recipient_act.shape[0], 4)  


            dilemma_prob_actual = dilemma_probs.gather(1, recipient_act.unsqueeze(1)).squeeze(1)  
            dilemma_prob_counter = dilemma_probs.gather(1, counterfact_act)  
            selected_dilemma_probs = torch.cat(
                [dilemma_prob_actual.unsqueeze(1), dilemma_prob_counter], dim=1
            )  

            selected_recipient_actions = torch.cat(
                [recipient_act.unsqueeze(1), counterfact_act], dim=1
            ) 
            max_actions =selected_recipient_actions.max()  
            selected_recipient_actions=selected_recipient_actions/max_actions


            all_possible_values=[]
            for action_idx in range(max_actions+1):
                temp_obs=last_dilemma_value_obs.clone()
                for batch_idx,recipient_action in enumerate(selected_recipient_actions):
                        recipient_position=torch.nonzero(
                            last_dilemma_value_obs[batch_idx][2].any(dim=-1), as_tuple=False
                        ) 
                        temp_obs[batch_idx][-2][tuple(recipient_position[0])][1]=recipient_action[action_idx]
                all_possible_values.append(dilemma_policy.predict_values(temp_obs.reshape(*temp_obs.shape[:-2], -1)).reshape(-1))
            all_possible_values=torch.stack(all_possible_values, dim=1)
            baseline_value = (selected_dilemma_probs * all_possible_values).sum(dim=1)

            predict_value = dilemma_policy.predict_values(
                obs=obs_predict_value
            ).reshape(-1)

            final_value = predict_value - repu_baseline_weight * baseline_value

        final_value = (final_value - final_value.mean()) / (final_value.std() + 1e-8)
        return final_value


class ReputationCnnPolicy(ReputationPolicy):
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