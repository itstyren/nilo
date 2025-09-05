from ...algorithm.basePolicy import basePolicy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.utils import get_device
from functools import partial

from grg.algorithm.Extractor import CnnGRUExtractor

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class IdentityExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        super().__init__(observation_space, features_dim=int(np.prod(observation_space.shape)))
        # You preserve the original input shape here.

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


class MultiHeadReputationAttention(nn.Module):
    """
    Multi-Head Attention module for computing agent reputations
    from observations and actions. Each agent queries others using
    attention to produce scalar reputation scores in [0, 1].
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[list[int], dict[str, list[int]]],
        activation_fn: type[nn.Module],
        num_heads=4,
        device: Union[torch.device, str] = "auto",
    ):
        """
        :param input_dim: Dimension of each agent's observation
        :param net_arch: specify the amount and size of the hidden layers
        :param num_heads: Number of attention heads
        :param activation_fn: The activation function to use for the networks.
        :param device: PyTorch device.
        """
        super().__init__()
        self.device = get_device(device)
        self.num_heads = num_heads

        # Handle net_arch dictionary or shared architecture
        if isinstance(net_arch, dict):
            embed_layers = net_arch.get("embed", [])
            query_layers = net_arch.get("query", [])
            key_layers = net_arch.get("key", [])
            value_layers = net_arch.get("value", [])
            latent_layers = net_arch.get("latent", [])
        else:
            embed_layers = query_layers = key_layers = value_layers = net_arch

        self.latent_dim = latent_layers[-1]

        # Build subnets
        self.embed = self._build_mlp(feature_dim, embed_layers, activation_fn).to(
            self.device
        )
        self.query = self._build_mlp(feature_dim, query_layers, activation_fn).to(
            self.device
        )
        # These two will use embed output dim as input
        embed_dim = embed_layers[-1] if embed_layers else feature_dim
        self.key = self._build_mlp(
            embed_dim, key_layers, activation_fn).to(self.device)
        self.value = self._build_mlp(embed_dim, value_layers, activation_fn).to(
            self.device
        )

        # Check multi-head setup
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        out_layers = []
        in_dim = feature_dim + embed_dim
        # embed_dim_copy = embed_dim
        for out_dim in latent_layers:
            out_layers.append(nn.Linear(in_dim, out_dim))
            out_layers.append(nn.ReLU())  # or another activation if you prefer
            in_dim = out_dim

        # scalar reputation per agent
        self.out_proj = nn.Sequential(*out_layers).to(self.device)

    def _build_mlp(
        self, input_dim: int, layer_dims: list[int], activation_fn: type[nn.Module]
    ) -> nn.Sequential:
        layers = []
        for dim in layer_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(activation_fn())
            input_dim = dim
        return nn.Sequential(*layers) if layers else nn.Identity()

    def split_heads(self, x):
        B, N, E = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, H, N, D)

    def merge_heads(self, x):
        B, H, N, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * D)  # (B, N, E)

    def forward(self, all_obs: torch.Tensor,attention_weights=False) -> torch.Tensor:
        """
        :param all_obs: Tensor of shape (B, N, obs_dim)
        :return: Tensor of shape (B, N) - scalar reputations
        """
        x_embed = self.embed(all_obs)  # (B, N, E)
        q = self.query(all_obs)  # (B, N, E)
        k = self.key(x_embed)  # (B, N, E)
        v = self.value(x_embed)  # (B, N, E)


        q = self.split_heads(q)  # (B, H, N, D)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # (B, H, N, N)
        weights = F.softmax(scores, dim=-1)  # (B, H, N, N)
        context = torch.matmul(weights, v)  # (B, H, N, D)
        context = self.merge_heads(context)  # (B, N, E)

        combined = torch.cat([all_obs, context], dim=-1)
        raw_scores=self.out_proj(combined)

        # raw_scores = self.out_proj(context)  # (B, N)
        observation_importance = weights.mean(dim=( 1, 2))  # [9]
        if attention_weights:
            return raw_scores, observation_importance
        else:
            return raw_scores



class MultiHeadReputationPolicy(basePolicy):
    """
    Reputation policy that uses multi-head attention to assign
    scalar reputation values to each agent based on their
    observation-action pairs.
    """

    def __init__(
        self,
        args,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class=IdentityExtractor,
        num_heads=4,
        hidden_dim=64,
        share_features_extractor: bool = True,
        ortho_init: bool = True,
        vf_observation_space: Optional[spaces.Space] = None,
        **kwargs: Any,
    ):
        """
        :param args: Global arguments (e.g., from argparse or config)
        :param observation_space: Gym-style space defining input shape
        :param action_space: Gym-style space defining action shape
        :param lr_schedule: Learning rate scheduler
        :param embed_dim: Embedding dimension for attention mechanism
        :param num_heads: Number of attention heads
        :param hidden_dim: Hidden layer size for actor and critic
        """

        self.args = args
        self.vf_observation_space = vf_observation_space

        super().__init__(observation_space, action_space,
                         features_extractor_class, **kwargs)

        self.is_discrete = isinstance(action_space, spaces.Discrete)
        self.action_dim = 1 if self.is_discrete else action_space.shape[0]

        if not self.is_discrete:
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.num_heads = num_heads
        self.ortho_init = ortho_init

        # wether share the features extractor between the policy and value network
        self.share_features_extractor = share_features_extractor
        # get features extractor for actor and critic
        self.make_features_extractor()

        self.log_std_init = log_std_init
        dist_kwargs = None

        assert not (
            squash_output and not use_sde
        ), "squash_output=True is only available when using gSDE (use_sde=True)"
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs
        # Action distribution
        self.action_dist = make_proba_distribution(
            action_space, use_sde=use_sde, dist_kwargs=dist_kwargs
        )

        self._build(lr_schedule)

        # self.critic = nn.Sequential(
        #     nn.Linear(self.obs_dim + self.action_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1),
        # )

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            [self.pi_features_dim, self.vf_features_dim],
            net_arch=dict(vf=self.net_arch.get("vf", [])),
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        # only create for critic
        self._build_mlp_extractor()
        self.attention = MultiHeadReputationAttention(
            feature_dim=self.pi_features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            num_heads=self.num_heads,
            device=self.device,
        )

        latent_dim_pi = self.attention.latent_dim

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

        self.value_net = nn.Linear(
            # self.mlp_extractor.latent_dim_vf, self.action_space.n,
            self.mlp_extractor.latent_dim_vf,
            1,
        )

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            from functools import partial

            module_gains = {
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1.0,
                self.attention: np.sqrt(2),
            }

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(
            1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def forward(self, obs: torch.Tensor, deterministic: bool = False,attention_weights=False) -> torch.Tensor:
        """
        :param obs: (B, N, obs_dim)
        :return: action (B, N, 1), log_prob (B, N, 1), entropy (B, N, 1)
        """
        if obs.ndim == 5:
           features = self.extract_features(obs.reshape(-1, *obs.shape[2:]), features_type="pi")    # Flatten batch and agent dimensions
           features=features.reshape(obs.shape[0], obs.shape[1], -1)  # Reshape back to (B, N, feature_dim)
        else:
            features = self.extract_features(obs, features_type="pi")
        
        

        # return self.attention(obs)
        if attention_weights:
            repu_mean, observation_importance = self.attention(features,attention_weights)  # (B, N,N)#
        else:
            repu_mean = self.attention(features)
        B, N = obs.shape[:2]

        # Flatten for distribution input
        repu_mean_flat = repu_mean.reshape(B * N, -1)
        # breakpoint()

        distribution = self._get_action_dist_from_latent(repu_mean_flat)
        actions_flat = distribution.get_actions(
            deterministic=deterministic)  # [B*N, 1]

        log_prob_flat = distribution.log_prob(actions_flat)  # [B*N]
        log_prob = log_prob_flat.view(B, N)                  # [B, N]

        actions = actions_flat.view(B, N, *self.action_space.shape)
        if self.action_space.shape == (1,):
            actions = actions.squeeze(-1)

        values = []
        # _, latent_vf = self.mlp_extractor(features)

        # values = self.value_net(latent_vf)
        if attention_weights:
            return actions, values,log_prob,observation_importance
        else:
            return actions, values, log_prob

    def _get_action_dist_from_latent(self, repu_mean: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(repu_mean)
        # F.softmax(mean_actions, dim=-1)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    # def evaluate_masked(self, obs, repu_actions):
    #     features = self.extract_features(obs)
    #     repu_mean = self.attention(features)  # (B, N)
    #     distribution = self._get_action_dist_from_latent(repu_mean)
    #     log_prob = distribution.log_prob(repu_actions)
    #     entropy = distribution.entropy()
    #     _, latent_vf = self.mlp_extractor(features)
    #     value_preds = self.value_net(latent_vf).squeeze(-1)
    #     return value_preds, log_prob, entropy

    def evaluate_unmasked(self, obs, repu_actions):
        """
        Evaluate log_probs, entropy, and value predictions for given observations and reputation actions.

        :param obs: Tensor of shape (B, N, obs_dim)
        :param repu_actions: Tensor of shape (B, N) for Box(1,) or (B, N, action_dim)
        :return: value_preds [B, N], log_prob [B, N], entropy [B, N]
        """
        B, N = obs.shape[:2]
        if obs.ndim == 5:
            temp_obs = obs.reshape(-1, *obs.shape[2:-1], obs.shape[-1]//2, 2)
            critic_obs=F.pad(temp_obs, (0, 1))
            for batch_idx in range(temp_obs.shape[0]):
                agent_position=critic_obs[batch_idx][0]
                actions=repu_actions.reshape(-1)  # Flatten batch and agent dimensions 
                # Loop through all vectors and modify
                for idx in np.ndindex(agent_position.shape[:2]):  # iterate over first two dims (3, 3)
                    vec = agent_position[idx]
                    if vec[0] >0:  # Check if the vector is [1, 0, 0]
                        critic_obs[batch_idx][0][idx][-1] = actions[batch_idx]# Set the last element to the action value
            critic_obs= critic_obs.view(*critic_obs.shape[:-2], -1)
            # critic_obs = critic_obs.reshape(-1, 2, *critic_obs.shape[1:])

            # obs=obs.view(*obs.shape[:-2], -1)
            act_features=self.extract_features(obs.reshape(-1, *obs.shape[2:]), features_type="pi") 

            act_features=act_features.reshape(obs.shape[0], obs.shape[1], -1)

        else:
            act_features = self.extract_features(
                obs, features_type="pi")                    # [B, N, obs_dim]
        
            critic_obs = torch.cat(
                [obs, repu_actions.unsqueeze(-1)], dim=-1

            )

        # [B, N, latent_dim]
        repu_mean = self.attention(act_features)

        # Flatten for distribution input
        repu_mean_flat = repu_mean.reshape(B * N, -1)

        distribution = self._get_action_dist_from_latent(repu_mean_flat)

        # sampe_flat_action=distribution.get_actions(deterministic=False)  # [B*N, 1]
        # sample_log_prob_flat = distribution.log_prob(sampe_flat_action)  # [B*N]
        # sample_log_prob = sample_log_prob_flat.view(B, N)

        # breakpoint()

        # Flatten actions and ensure shape is correct
        repu_actions_flat = repu_actions.reshape(B * N, -1)

        # Ensure action shape matches what the distribution expects
        if self.action_space.shape == ():
            # Discrete: flatten to [B*N]
            repu_actions_flat = repu_actions_flat.squeeze(-1)  # to [B*N]

        elif self.action_space.shape == (1,) and repu_actions_flat.shape[1] != 1:
            # ensure [B*N, 1]
            repu_actions_flat = repu_actions_flat.unsqueeze(-1)
        
        # distribution.log_prob(repu_actions_flat.squeeze(-1))

        log_prob_flat = distribution.log_prob(repu_actions_flat)     # [B*N]
        entropy_flat = distribution.entropy()                        # [B*N]

        log_prob = log_prob_flat.view(B, N)
        entropy = entropy_flat.view(B, N)

        vf_features = self.extract_features(critic_obs, features_type="vf")

        _, latent_vf = self.mlp_extractor(vf_features)
        value_preds = self.value_net(latent_vf).squeeze(-1)          # [B, N]
        if  obs.ndim == 5:
            value_preds = value_preds.reshape(B, N)
        return value_preds, log_prob, entropy, distribution

    def extract_features(  # type: ignore[override]
        self,
        obs: PyTorchObs,
        features_type: str = "pi",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if features_type == "pi":
            features_extractor = self.pi_features_extractor
            feature = super().extract_features(obs, features_extractor)
        else:
            features_extractor = self.vf_features_extractor
            feature = super().extract_features(obs, features_extractor)

        return feature

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)

        return self.value_net(latent_vf)

    def evaluate_baseline(
        self,
        obs: PyTorchObs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the baseline for given observations and actions.

        :param obs: Observation
        :param actions: Actions taken
        :return: The estimated values and log probabilities.
        """
        with torch.no_grad():

            pi_features = self.extract_features(obs, features_type="pi")
            latent_pi = self.attention(pi_features)

            # mean_actions = self.action_net(latent_pi)
            # log_prob=F.softmax(mean_actions, dim=-1)


            if isinstance(self.action_space, spaces.Box):
                B, N = obs.shape[:2]
                repu_mean_flat = latent_pi.reshape(B * N, -1)
                distribution = self._get_action_dist_from_latent(repu_mean_flat)
                baseline_value=[]
                for _ in range(10):
                    sample_action=distribution.sample()
                    sample_action = sample_action.view(B, N, *self.action_space.shape)
                    baseline_obs = torch.cat(
                        [obs, sample_action], dim=-1
                    )
                    value_preds = self.predict_values(baseline_obs)
                    baseline_value.append(value_preds)
                stack_value=torch.stack(baseline_value, dim=0)
                mean_baseline = stack_value.mean(dim=0).reshape(B, N)
            else:
                actions_prob, distribution = self._get_action_logits_and_dist(
                latent_pi)  # [B,N,Action_dim]
                possible_act = distribution.get_actions(
                deterministic=False)  # [B,N]
                counter_act = 1 - possible_act

                # Expand dims to match the shape needed for gather along the last dimension
                possible_act_exp = possible_act.unsqueeze(-1)  # [B, N, 1]
                counter_act_exp = counter_act.unsqueeze(-1)    # [B, N, 1]

                # Gather the probs of chosen and counter actions
                possible_act_probs = actions_prob.gather(
                    2, possible_act_exp).squeeze(-1)  # [B, N]
                counter_probs = actions_prob.gather(
                    2, counter_act_exp).squeeze(-1)    # [B, N]
                # Flatten for distribution input

                selected_probs = torch.stack(
                    [possible_act_probs, counter_probs], dim=2)

                # get the possiable obs
                possible_obs = torch.cat(
                    [obs, possible_act.unsqueeze(-1)], dim=-1
                )
                values = self.predict_values(obs=possible_obs)

                # get the conterfactal obs
                counterfacl_obs = torch.cat(
                    [obs, counter_act.unsqueeze(-1)], dim=-1
                )
                counterfacl_value = self.predict_values(counterfacl_obs)

                mean_baseline = (
                    torch.cat([values, counterfacl_value], dim=2) * selected_probs).sum(dim=2)
        
        return mean_baseline

    def eval_baseline_2(
        self,
        repu_obs_for_act: PyTorchObs,
        repu_obs_after_dilemma: PyTorchObs,
        dilemma_policy: basePolicy,
        self_index: int,
        recipient_idx: int,
        repu_baseline_weight: float = 0.5,
        rewards: Optional[torch.Tensor] = None,
        repu_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the baseline for given observations
        """
        B = repu_obs_after_dilemma.shape[0]  # infer batch size, e.g., 62
        N = repu_obs_after_dilemma.shape[1]  # infer agent count, e.g., 16

        payoff_matrix = torch.tensor([[1, -0.1], [1.1, 0]])

        # with torch.no_grad():
        #     self_index = torch.tensor(self_index, device=repu_obs_for_act.device).expand(B, 1)


        with torch.no_grad():
            obs_for_eval_dilemma_value=repu_obs_for_act.clone().reshape(-1, 4)


            obs_for_recipit_dilemma_act = repu_obs_for_act[:,:,:2].clone()
            obs_for_recipit_dilemma_act=obs_for_recipit_dilemma_act[...,[1,0]]

            recipient_dilemma_act, dilemma_act_prob_for_2 = dilemma_policy.get_action_and_prob(
                obs_for_recipit_dilemma_act.reshape(-1, 2))
            recipient_counterfact_dilemma_act = 1-recipient_dilemma_act
            recipient_dilemma_act_prob = dilemma_act_prob_for_2.gather(
                1, recipient_dilemma_act.unsqueeze(-1)).squeeze(-1)  # [B, N]
            rrecipient_counterfact_dilemma_act_prob = dilemma_act_prob_for_2.gather(
                1, recipient_counterfact_dilemma_act.unsqueeze(-1)).squeeze(-1)    # [B, N]
            # Flatten for distribution input

            selected_dilemma_probs = torch.stack(
                [recipient_dilemma_act_prob, rrecipient_counterfact_dilemma_act_prob], dim=1)
            selected_recipient_actions = torch.stack(
                [recipient_dilemma_act, recipient_counterfact_dilemma_act], dim=1)
            

            # iterate over the all possible actions
            new_repu_critic_values_set=[]
            for action_i in range(dilemma_policy.action_space.n):
                _obs=obs_for_eval_dilemma_value.clone()
                recipient_act=selected_recipient_actions[:, action_i]

                _obs[:,-1]=recipient_act
                
                critic_obs=torch.cat(
                    [_obs.reshape(B,N,4), repu_actions.unsqueeze(-1)], dim=-1
                )

                repu_critic_values=self.predict_values(
                    obs=critic_obs).reshape(-1)
                new_repu_critic_values_set.append(repu_critic_values)
            
            new_repu_critic_values_set=torch.stack(new_repu_critic_values_set, dim=1)
            baseline_value = (selected_dilemma_probs * new_repu_critic_values_set).sum(dim=1)

        return baseline_value




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
        """
        Evaluate the baseline for given observations
        """
        B, N = repu_obs_after_dilemma.shape[:2]

        with torch.no_grad():
            # get dilemma round result for myself
            obs_self=repu_obs_after_dilemma[:, self_index, :].view(B, -1, 4 + N)

            obs_predict_value =obs_self[..., :4].reshape(-1, 4)

            # Extract opponent's dilemma action observations
            obs_opponent_act = obs_self[..., :2].reshape(-1, 2)[:, [1, 0]]


            # Get recipient actions and action probabilities
            recipient_act, dilemma_probs = dilemma_policy.get_action_and_prob(obs_opponent_act)
            counterfact_act = 1 - recipient_act

            # Gather probabilities for actual and counterfactual actions
            dilemma_prob_actual = dilemma_probs.gather(1, recipient_act.unsqueeze(-1)).squeeze(-1)
            dilemma_prob_counter = dilemma_probs.gather(1, counterfact_act.unsqueeze(-1)).squeeze(-1)
            # Flatten for distribution input

            selected_dilemma_probs = torch.stack([dilemma_prob_actual, dilemma_prob_counter], dim=1)
            selected_recipient_actions = torch.stack([recipient_act, counterfact_act], dim=1)
            

            # Prepare batch for value predictions under both actions
            temp_obs = obs_predict_value.unsqueeze(1).repeat(1, 2, 1)
            temp_obs[..., -1] = selected_recipient_actions

            # Flatten batch for efficient value prediction
            temp_obs_flat = temp_obs.view(-1, 4)
            dilemma_values_flat = dilemma_policy.predict_values(obs=temp_obs_flat).reshape(-1, 2)


            # Compute baseline value
            baseline_value = (selected_dilemma_probs * dilemma_values_flat).sum(dim=1)    
               

            # Predict original value
            predict_value = dilemma_policy.predict_values(obs=obs_predict_value).reshape(-1)

            # Final adjusted value
            final_value = predict_value - repu_baseline_weight * baseline_value

        # Normalize
        final_value=(final_value - final_value.mean()) / (final_value.std() + 1e-8)

        # Reshape to match the original expectation
        reshape_dim = obs_self[..., :4].shape[-2]
        final_value = final_value.view(-1, reshape_dim)

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
        """
        Evaluate the baseline for given observations
        """
        

        with torch.no_grad():
            # Extract current agent's dilemma-related observations
            last_dilemma_value_obs=repu_obs_after_dilemma[:, self_index, :]
            obs_predict_value = last_dilemma_value_obs.reshape(*last_dilemma_value_obs.shape[:-2], -1)

            # Extract opponent's dilemma action observations
            obs_opponent_act = last_dilemma_value_obs[..., 0][:, [2, 3, 0, 1], :, :]


            # Get recipient actions and action probabilities
            recipient_act, dilemma_probs = dilemma_policy.get_action_and_prob(
                obs_opponent_act
            )
            # Create a tensor of all possible actions [0, 1, 2, 3, 4]
            all_actions = torch.arange(5, device=recipient_act.device)  # shape [5]

            # Expand recipient actions to compare with all actions
            recipient_act_exp = recipient_act.unsqueeze(1)  # [batch_size, 1]

            # Broadcast and compare to create mask for all other actions
            mask = all_actions.unsqueeze(0) != recipient_act_exp  # [batch_size, 5]

            # Use the mask to select all actions except the actual one
            counterfact_act = all_actions.expand(recipient_act.shape[0], 5)[mask].reshape(recipient_act.shape[0], 4)  # [batch_size, 4]


            dilemma_prob_actual = dilemma_probs.gather(1, recipient_act.unsqueeze(1)).squeeze(1)  # [batch_size]
            dilemma_prob_counter = dilemma_probs.gather(1, counterfact_act)  # [batch_size, 4]
            # Flatten for distribution input

            # Optional: stack for further processing
            selected_dilemma_probs = torch.cat(
                [dilemma_prob_actual.unsqueeze(1), dilemma_prob_counter], dim=1
            )  # [batch_size, 5]

            selected_recipient_actions = torch.cat(
                [recipient_act.unsqueeze(1), counterfact_act], dim=1
            )  # [batch_size, 5]

            max_actions =selected_recipient_actions.max()  # [batch_size, 1]
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

            # Predict original value
            predict_value = dilemma_policy.predict_values(
                obs=obs_predict_value
            ).reshape(-1)

            # Final adjusted value
            final_value = predict_value - repu_baseline_weight * baseline_value

        # # Normalize
        final_value = (final_value - final_value.mean()) / (final_value.std() + 1e-8)
        return final_value


    def _get_action_logits_and_dist(self, repu_mean: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(repu_mean)
        # F.softmax(mean_actions, dim=-1)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution = self.action_dist.proba_distribution(
                mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            distribution = self.action_dist.proba_distribution(
                action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            distribution = self.action_dist.proba_distribution(
                action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            distribution = self.action_dist.proba_distribution(
                action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")
        return F.softmax(mean_actions, dim=-1), distribution




class CnnGRUMultiHeadReputationPolicy(MultiHeadReputationPolicy):
    def __init__(
        self,
        args,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,
        use_sde=False,
        log_std_init=0.0,
        full_std=True,
        use_expln=False,
        squash_output=False,
        num_heads=4,
        hidden_dim=64,
        share_features_extractor=True,
        ortho_init=True,
        vf_observation_space=None,
        features_extractor_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            args=args,
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            share_features_extractor=share_features_extractor,
            ortho_init=ortho_init,
            vf_observation_space=vf_observation_space,
            features_extractor_class=CnnGRUExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs,
        )