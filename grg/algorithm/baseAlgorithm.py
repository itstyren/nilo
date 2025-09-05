from abc import ABC
from typing import List, Type, Union
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from .basePolicy import basePolicy
import torch
from stable_baselines3.common.utils import (
    get_schedule_fn,
    update_learning_rate,
    get_device,
)
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union


class baseAlgorithm(ABC):
    def __init__(
        self,
        all_args,
        logger,
        env: Union[GymEnv, str],
        sub_env: str,
        policy_class: Union[str, Type[basePolicy]],
        learning_rate: Union[float, Schedule],
        n_epochs: int,
        max_grad_norm: float,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        device: Union[torch.device, str] = "auto",
    ):
        self.all_args = all_args
        self.logger = logger
        self.policy_class = policy_class
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.device = get_device(device)
        self._current_progress_remaining = 1.0

        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self._n_updates = 0 


        if env is not None:
            self.observation_space = env.observation_space[sub_env]
            self.action_space =env.action_space[sub_env]
            self.env = env
            self.n_envs = env.num_envs


    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self._setup_updates_schedule()

        self.policy = self.policy_class(
            args=self.all_args,
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy.to(self.device)


    def _setup_lr_schedule(self) -> None:
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _update_learning_rate(self, optimizers: Union[list[torch.optim.Optimizer], torch.optim.Optimizer]) -> None:
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))
        return self.lr_schedule(self._current_progress_remaining)

    def _setup_updates_schedule(self) -> None:
        self.updates_schedule = get_schedule_fn(self.n_epochs)

    def _update_n_epochs(self) -> None:
        n_epochs=int(self.updates_schedule(self._current_progress_remaining))
        return n_epochs