import numpy as np
from .._cg_utils.core import World, Agent
from .._cg_utils.coin_env import CoinGameEnv
from .._cg_utils.scenario import BaseScenario
from .._cg_utils import utils


class raw_env(CoinGameEnv):
    def __init__(self, args, max_cycles=1, render_mode=None, seed=None):
        scenario = Scenario()
        world = scenario.make_world(args, seed=seed)
        CoinGameEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            max_cycles=max_cycles,
            render_mode=render_mode,
            args=args,
        )
        self.metadata["name"] = "coinRepu_v0"


env = utils.make_env(raw_env)
parallel_env = utils.parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()

    def make_world(self, args, seed=None):
        agent_num = args.env_dim
        grid_size = args.grid_size

        world = World(agent_num, grid_size)
        if args.baseline_type != "None":
            world.set_repu = False

        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"

        return world

    def reset_world(self, world, action_space, np_rng, _rng_seed=None):
        world.reset()

        agent_num = len(world.agents)

        repu_sample = utils.sample_repu_for_agents(
            action_space["reputation"], agent_num, np_rng
        )
        for agent in world.agents:
            agent.reputation_view = repu_sample.copy()

    def observation(self, self_agent, world):
        grid_size = world.grid_size
        num_agents = world.num_agents

        state = np.zeros((num_agents * 2, grid_size * grid_size), dtype=np.float32)

        agent_positions = world.get_agent_positions()
        flat_agent_positions = agent_positions[:, 0] * grid_size + agent_positions[:, 1]
        flat_coin_positions = world.coins[:, 0] * grid_size + world.coins[:, 1]

        for i in range(num_agents):
            state[2 * i, flat_agent_positions[i]] = (
                self_agent.reputation_view[i]
                if self_agent.reputation_view[i] > 0
                else 0.3 if world.set_repu else 1
            )
            state[2 * i + 1, flat_coin_positions[i]] = 1

        state = state.reshape(num_agents * 2, grid_size, grid_size)

        agent_idx = world.agents.index(self_agent)
        rolled_state = np.roll(state, shift=-2 * agent_idx, axis=0)

        for _agent in world.agents:
            if agent_idx != _agent.index:
                recipient_act = _agent.action.s

        previous_state = (
            self_agent.previous_state.copy()
            if self_agent.previous_state is not None
            else rolled_state.copy()
        )
        self_obs = previous_state[[0, 1, 2, 3]].copy()
        recipient_obs = previous_state[[2, 3, 0, 1]].copy()

        normalized_agent_actions = self_agent.action.s / (
            len(self_agent.action.MOVES) - 1
        )
        normalized_recipient_actions = recipient_act / (
            len(self_agent.action.MOVES) - 1
        )
        self_obs_expand = np.repeat(self_obs[..., np.newaxis], 2, axis=-1)
        recipient_obs_expand = np.repeat(recipient_obs[..., np.newaxis], 2, axis=-1)

        agent_position = np.argwhere(self_obs[0] != 0)
        self_obs_expand[0][tuple(agent_position[0])][1] = normalized_agent_actions
        recipient_postion = np.argwhere(self_obs[2] != 0)
        self_obs_expand[2][tuple(recipient_postion[0])][
            1
        ] = normalized_recipient_actions

        agent_position = np.argwhere(recipient_obs[0] != 0)
        recipient_obs_expand[0][tuple(agent_position[0])][
            1
        ] = normalized_recipient_actions
        recipient_postion = np.argwhere(recipient_obs[2] != 0)
        recipient_obs_expand[2][tuple(recipient_postion[0])][
            1
        ] = normalized_agent_actions

        repu_obs = np.array([self_obs_expand, recipient_obs_expand])
        repu_obs = np.roll(repu_obs, shift=1, axis=0) if agent_idx == 1 else repu_obs

        world.agents[agent_idx].previous_state = rolled_state.copy()

        obs = {
            "dilemma_obs": rolled_state,
            "repu_obs": repu_obs,
        }

        return obs

    def update_repu_view(self, agent, repu_action):
        for i, assign_repu in enumerate(repu_action):
            agent.reputation_view[i] = repu_action[assign_repu]