from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
from collections import OrderedDict
from gymnasium import spaces
from typing import List,  Tuple, Union

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnvObs,
)


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions,action_type,move_step=True):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    # @abstractmethod
    # def calcu_async(self, actions):
    #     pass

    # @abstractmethod
    # def calcu_wait(self):
    #     pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions, action_type,move_step=True):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.

        :param action: the action to take
        :param action_type: the type of action (asssessment means only update self assessment)        
        """
        self.step_async(actions, action_type,move_step)
        return self.step_wait()

    def update_repu(self, repus, update_env=True):
        self.repu_async(repus, update_env)
        return self.repu_wait()

    # def calculate_reward(self):
    #     self.calcu_async()
    #     return self.calcu_wait()


class DummyVecEnv(ShareVecEnv):
    """
    Sing Env
    """

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

            # checking whether a variable named done is of type bool or a NumPy array
        if (
            "bool" in truncation.__class__.__name__
            or "bool" in termination.__class__.__name__
        ):
            if termination:
                obs = self.env.reset(options="termination")
            elif truncation:
                obs = self.env.reset()
                # print('reset')

        self.actions = None
        # compatibility with the multi env version
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
        
        return _flatten_obs(results, self.observation_space)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "human":
            # Let the first environment handle live window rendering
            self.env.render(mode="human")
        elif mode == "rgb_array":
            # Return a list of RGB images (or a batch) for gif/saving
            frame = self.env.render(mode="rgb_array")
            return np.array(frame)
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported in DummyVecEnv.")
        

class SubprocVecEnv(ShareVecEnv):
    """
    Multiple Env
 
    """
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
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
                True  # if the main process crashes, we should not cause things to hang
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
        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs, self.observation_space)


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
            # breakpoint()
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
            # checking whether a variable named done is of type bool or a NumPy array
            if (
                "bool" in truncation.__class__.__name__
                or "bool" in termination.__class__.__name__
            ):
                if termination:
                    # print("Termination")
                    obs = env.reset(options="termination")
                elif truncation:
                    # print("Truncation")
                    obs = env.reset()
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
            ob = env.reset()
            remote.send(ob)
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
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    # if should_combine_nested_dicts(space):
    #     space=combine_nested_dicts(space)


    assert isinstance(
        obs, (list, tuple)
    ), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"


    if isinstance(space, spaces.Dict) and isinstance(obs[0], dict):
        assert isinstance(space.spaces, dict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return {key: np.stack([single_obs[key] for single_obs in obs]) for key in space.spaces.keys()}  # type: ignore[call-overload]
    elif isinstance(space, spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([single_obs[i] for single_obs in obs]) for i in range(obs_len))  # type: ignore[index]
    else:
        return np.stack(obs)  # type: ignore[arg-type]