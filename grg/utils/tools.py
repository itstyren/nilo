

import math
from typing import Callable,Any, Dict, Optional, Tuple, Type, Union
from gymnasium import spaces
from torch.nn import functional as F
import torch

def extract_scenarios(input_string):
    scenarios= ['harvest', 'prisoner','matrix']
    for scenario in scenarios:
        if scenario in input_string.lower():
            return scenario
    return None

def round_up(number: float, decimals: int = 2) -> float:
    if not isinstance(decimals, int):
        raise TypeError("Decimal places must be an integer")
    if decimals < 0:
        raise ValueError("Decimal places must be 0 or more")
    if decimals == 0:
        return math.ceil(number)
    else:
        factor = 10**decimals
        return math.ceil(number * factor) / factor

def linear_schedule_to_0(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def linear_schedule_to_1(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return initial_value + (1.0 - initial_value) * (1 - progress_remaining)
    return func

def periodic_linear_schedule(initial_value: float, final_value: float, cycle_length: float = 0.1) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        elapsed = 1 - progress_remaining
        cycle_pos = (elapsed % cycle_length) / cycle_length
        return initial_value + (final_value - initial_value) * cycle_pos
    return func

def linear_schedule_from_to_int(initial_value: int, final_value: int) -> Callable[[float], int]:
    def func(progress_remaining: float) -> int:
        return int(round((1 - progress_remaining) * (final_value - initial_value) + initial_value))
    return func

def delayed_linear_schedule(
    initial_value: float,
    final_value: float,
    delay_fraction: float = 0.3,
    reach_fraction: float = 1.0
) -> Callable[[float], float]:
    assert 0.0 <= delay_fraction < reach_fraction <= 1.0
    def func(progress_remaining: float) -> float:
        elapsed = 1.0 - progress_remaining
        if elapsed < delay_fraction:
            return initial_value
        elif elapsed >= reach_fraction:
            return final_value
        else:
            linear_elapsed = (elapsed - delay_fraction) / (reach_fraction - delay_fraction)
            return initial_value + (final_value - initial_value) * linear_elapsed
    return func

def linear_countdown_to_1(initial_value: int) -> Callable[[float], int]:
    def func(progress_remaining: float) -> int:
        return max(1, int(round(progress_remaining * (initial_value - 1) + 1)))
    return func

def t2n(x,reshape_dim_1=None,reshape_dim_2=None,reshape_dim_3=None):
    x=x.detach().cpu().numpy()
    if reshape_dim_3 is None:
        if reshape_dim_1 is not None and reshape_dim_2 is not None:
            x=x.reshape(reshape_dim_1,reshape_dim_2)
    else:
            x=x.reshape(reshape_dim_1,reshape_dim_2,reshape_dim_3)
    return x

def preprocess_obs(
    obs: torch.Tensor,
    observation_space: spaces.Space,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(observation_space, spaces.Box):
        return obs.float()
    if isinstance(observation_space, spaces.MultiDiscrete):
        return torch.cat(
            [
                F.one_hot(obs[_].long(), num_classes=dim).float()
                for _, dim in enumerate(observation_space.nvec)
            ],
            dim=-1,
        ).view(obs.shape[0], observation_space.nvec[0])
    elif isinstance(observation_space, spaces.Dict):
        assert isinstance(obs, Dict)
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs(_obs, observation_space[key])
        return preprocessed_obs
    elif isinstance(observation_space, spaces.Discrete):
        return F.one_hot(obs.long(), num_classes=int(observation_space.n)).float()
    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.float()
    else:
        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")

def category_attention_weight_to_norm(binary_array, value_array):
    combinations = torch.tensor([[0, 1], [0, 0], [1, 1], [1, 0]])
    result = torch.zeros(4)
    num_rows = binary_array.shape[0]
    for i, combo in enumerate(combinations):
        mask = (binary_array[:, :, 0] == combo[0]) & (binary_array[:, :, 1] == combo[1])
        row_sums = torch.zeros(num_rows)
        for row_idx in range(num_rows):
            row_mask = mask[row_idx]
            if row_mask.any():
                row_sums[row_idx] = value_array[row_idx][row_mask].sum()
            else:
                row_sums[row_idx] = 0
        result[i] = row_sums.mean()
    return result.cpu().numpy()