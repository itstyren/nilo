from torch import nn
from stable_baselines3.common import torch_layers
import gymnasium as gym
import torch.nn.functional as F
import torch


class CustomCNN(torch_layers.BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim=128,
        num_frames=6,
        fcnet_hiddens=(1024, 128),
    ):
        super().__init__(observation_space, features_dim)


        self.conv = nn.Sequential(
            nn.Conv2d(
                num_frames * 3, num_frames * 3, kernel_size=8, stride=4, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                num_frames * 3, num_frames * 6, kernel_size=5, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                num_frames * 6, num_frames * 6, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
        flat_out = num_frames * 6 * 7 * 7
        self.fc1 = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
        self.fc2 = nn.Linear(
            in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1]
        )

    def forward(self, observations) -> torch.Tensor:


        observations = observations.permute(0, 3, 1, 2)
        features = self.conv(observations)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return features