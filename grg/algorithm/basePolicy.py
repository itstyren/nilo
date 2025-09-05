from stable_baselines3.common.policies import BaseModel
from torch import nn
import torch
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
import numpy as np
from grg.utils import tools as utils
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.preprocessing import (
    get_action_dim,
    is_image_space,
    maybe_transpose,
    preprocess_obs,
)
from stable_baselines3.common.utils import (
    get_device,
    is_vectorized_observation,
    obs_as_tensor,
)
import copy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from typing_extensions import override

class baseMode(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def obs_to_tensor(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        observation_type=None,
    ) -> tuple[PyTorchObs, bool]:
        vectorized_env = False
        observation_space = (
            self.vf_observation_space
            if observation_type == "value_pred"
            else self.observation_space
        )

        if isinstance(observation, dict):
            assert isinstance(
                observation_space, spaces.Dict
            ), f"The observation provided is a dict but the obs space is {observation_space}"

            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(
                    obs_, obs_space
                )
                observation[key] = obs_.reshape((-1, *observation_space[key].shape))

        elif is_image_space(observation_space):
            observation = maybe_transpose(observation, observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            vectorized_env = is_vectorized_observation(observation, observation_space)
            observation = observation.reshape((-1, *observation_space.shape))

        obs_tensor = obs_as_tensor(observation, self.device)
        return obs_tensor, vectorized_env

    def create_features_extractor(self) -> BaseFeaturesExtractor:
        return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)

    def create_critic_extractor(self) -> BaseFeaturesExtractor:
        return self.features_extractor_class(
            self.vf_observation_space, **self.features_extractor_kwargs
        )

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        if self.share_features_extractor:
            self.features_extractor=self.create_features_extractor()
            self.features_dim = self.features_extractor.features_dim
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.create_features_extractor()
            self.pi_features_dim = self.pi_features_extractor.features_dim

            self.vf_features_extractor = self.create_critic_extractor()
            self.vf_features_dim = self.vf_features_extractor.features_dim


    def extract_features(
        self, obs: torch.Tensor, features_extractor: BaseFeaturesExtractor
    ) -> torch.Tensor:
        preprocessed_obs = preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )
        return features_extractor(preprocessed_obs)


class basePolicy(baseMode):
    def __init__(self, *args, squash_output: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._squash_output = squash_output

    def forward(self, obs, deterministic: bool = True):
        raise NotImplementedError

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)