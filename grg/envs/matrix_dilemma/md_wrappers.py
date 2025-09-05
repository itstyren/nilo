from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
from collections import OrderedDict
from gymnasium import spaces
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)


class CloudpickleWrapper(object):

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions,action_type,move_step=True):
        pass

    @abstractmethod
    def step_wait(self):
        pass


    def close_extras(self):
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions, action_type,move_step=True):
        self.step_async(actions, action_type,move_step)
        return self.step_wait()

    def update_repu(self, repus, update_env=True):
        self.repu_async(repus, update_env)
        return self.repu_wait()


class DummyVecEnv(ShareVecEnv):

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            self.env.observation_space("agent"),
            self.env.action_space("agent"),
        )
        self.actions = None
        self.action_type = None
        self.repu_access = None

    def step_async(self, actions,action_type,move_step=True):
        self.actions = actions[0]
        self.action_type = action_type
        self.move_step = move_step

    def step_wait(self):
        termination = False
        truncation = False
        for single_action in self.actions:
            result = self.env.step(single_action, self.action_type,self.move_step)
            if result is None:
                pass
            else:
                obs, rews, termination, truncation, infos = result


        if (
            "bool" in truncation.__class__.__name__
            or "bool" in termination.__class__.__name__
        ):
            if termination:
                obs, cl = self.env.reset(options="termination")

            elif truncation:
                obs, cl = self.env.reset()

        self.actions = None

        return (
            np.array([obs]),
            np.array([rews]),
            np.array([termination]),
            np.array([truncation]),
            np.array([infos]),
        )

    def reset(
        self,
        seed=None,
    ):
        results = [env.reset(seed) for env in self.envs]
        obs_all, coop_l = zip(*results)
        return _flatten_obs(obs_all, self.observation_space), _flatten_obs(coop_l, None)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="rgb_array", step=0):
        if mode == "train":
            frame = self.env.render(mode=mode, step=step)
            return np.array([frame])
        else:
            raise NotImplementedError


class SubprocVecEnv(ShareVecEnv):

    def __init__(self, env_fns, spaces=None):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn))
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.daemon = (
                True
            )
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("get_spaces", None))
        observation_spaces, action_spaces = self.remotes[0].recv()

        ShareVecEnv.__init__(
            self, len(env_fns), observation_spaces, action_spaces
        )

    def step_async(self, actions, action_type,move_step=True):
        for remote, action in zip(self.remotes, actions):
            remote.send(("action_step", [action, action_type,move_step]))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, termination,truncation,infos = zip(*results)
        return (
            np.stack(obs),
            np.stack(rews),
            np.stack(termination),
            np.stack(truncation),
            np.stack(infos),
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, coop_l = zip(*results)
        return _flatten_obs(obs, self.observation_space), _flatten_obs(coop_l, None)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array", step=0):
        for remote in self.remotes:
            remote.send(("render", (mode, step)))
        if mode == "train":
            results = [remote.recv() for remote in self.remotes]

            frame, action_array, repu_array = zip(*results)
            return np.stack(frame), np.stack(action_array), np.stack(repu_array)

    def get_actions(self):
        for remote in self.remotes:
            remote.send(("get_actions", None))
        results = [remote.recv() for remote in self.remotes]
        return results


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "action_step":
            for _ in range(len(data[0])):
                result = env.step(data[0][_], data[1],data[2])
                if result is None:
                    pass
                else:
                    obs, rews, termination, truncation, infos = result

            if (
                "bool" in truncation.__class__.__name__
                or "bool" in termination.__class__.__name__
            ):
                if termination:

                    obs, cl = env.reset(options="termination")
                elif truncation:

                    obs, cl = env.reset()
            remote.send(
                (
                    obs,
                    rews,
                    termination,
                    truncation,
                    infos,
                )
            )
        elif cmd == "reset":
            ob, coop_l = env.reset()
            remote.send((ob, coop_l))
        elif cmd == "render":
            if data[0] == "train":
                fr, action_arr, repu_arr = env.render(mode=data[0], step=data[1])
                remote.send((fr, action_arr, repu_arr))
            elif data == "human":
                env.render(mode=data)
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send(
                (env.observation_space("agent"), env.action_space("agent"))
            )
        elif cmd == "get_actions":
            actions = []
            for agent in env.world.agents:
                actions.append(agent.action.s)

            remote.send(actions)
        else:
            raise NotImplementedError


def _flatten_obs(
    obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: spaces.Space
) -> VecEnvObs:


    assert isinstance(
        obs, (list, tuple)
    ), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"


    if isinstance(space, spaces.Dict) and isinstance(obs[0], dict):
        assert isinstance(space.spaces, dict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return {key: np.stack([single_obs[key] for single_obs in obs]) for key in space.spaces.keys()}
    elif isinstance(space, spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([single_obs[i] for single_obs in obs]) for i in range(obs_len))
    else:
        return np.stack(obs)


def combine_nested_dicts(nested_dict):
    if not should_combine_nested_dicts(nested_dict):
        raise ValueError("Input is not a nested dictionary suitable for combining.")

    combined_dict = {}
    for outer_key, inner_dict in nested_dict.items():
        if isinstance(inner_dict, Dict):
            combined_dict.update(inner_dict)
        else:
            combined_dict[outer_key] = inner_dict

    return Dict(combined_dict)


def should_combine_nested_dicts(nested_dict):
    if isinstance(nested_dict, Dict):

        return any(isinstance(value, Dict) for value in nested_dict.values())
    return False