import gymnasium as gym
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np


class CnnGRUExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 16,
        gru_hidden_size: int = 16,
        gru_num_layers: int = 1,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "CoinsCnnGRUExtractor must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            cnn_output = self.cnn(sample_input)
            self.gru_input_size = cnn_output.view(cnn_output.size(0), -1).shape[1]

        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True
        )

        self.linear = nn.Sequential(
            nn.Linear(gru_hidden_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.size(0)
        cnn_out = self.cnn(observations)
        flat = cnn_out.view(batch_size, -1)
        gru_input = flat.unsqueeze(1)
        gru_out, _ = self.gru(gru_input)
        return self.linear(gru_out[:, -1, :])