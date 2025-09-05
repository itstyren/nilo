import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
import functools
from gymnasium import spaces
from grg.envs.matrix_dilemma._md_utils import utils
from stable_baselines3.common.utils import safe_mean


class MatrixEnv(AECEnv):
    """
    A Doner game environment has gym API for soical dilemma
    """

    metadata = {"render.modes": ["rgb_array"], "name": "matrix_v0"}

    def __init__(
        self,
        scenario,
        world,
        max_cycles,
        render_mode=None,
        args=None,
    ):
        super().__init__()
        self.render_mode = render_mode

        self.args = args
        #  game terminates after the number of cycles
        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]

        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }
        self._agent_selector = agent_selector(self.agents)

        # current dilemma actions for all agents
        self.current_dilemma_action = [agent.action.s for agent in self.world.agents]
        self.current_reputation_action = [
            [0.0 for _ in agent.reputation_view] for agent in self.world.agents
        ]

        # flag indicating whether the game is just reset
        self._just_reset = False

        # obs_repu,react_action==>assign repu
        self.norm_pattern = {
            "SJ": [[0.0, 1.0], [1.0, 0.0]],
            "IS": [[1.0, 0.0], [1.0, 0.0]],
            "SH": [[0.0, 0.0],[1.0, 0.0]],
            "SS": [[1.0, 1.0],[1.0, 0.0]], 
        }
        self.norm_type = self.args.norm_type
        self.dilemma_obs_shape = 2 if self.args.baseline_type == "None" else 1


    def _seed(self, seed=None):
        if seed is None:
            seed = 1
        self._rng_seed = seed  # Store the seed for later reference
        self.rng = np.random.default_rng(seed)

    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Set the action and observation spaces
        """
        observation_space = spaces.Dict(
            {
                "reputation": spaces.Box(
                    low=0, high=1, shape=(self.args.num_recipients, 4+self.num_agents), dtype=np.float32
                ),
                "reputation_vf": spaces.Box(low=0, high=1, shape=((4+self.num_agents)*self.args.num_recipients+1,), dtype=np.float32),  # include self assignment
                "dilemma": spaces.Box(low=0, high=1, shape=(self.dilemma_obs_shape,), dtype=np.float32),
                "dilemma_vf": spaces.Box(low=0, high=1, shape=(self.dilemma_obs_shape+2,), dtype=np.float32),   # include self and recipient actions
            }
        )

        return observation_space

    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        action_space = spaces.Dict(
            {
                "dilemma": spaces.Discrete(2),
                # "reputation": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "reputation": spaces.Discrete(2),
            }
        )

        return action_space

    def reset(self, seed=None, options="truncation"):
        """
        Reset the environment
        """
        # reset world
        if seed is not None:
            self._seed(seed=seed)
        else:
            self._seed(seed=self._rng_seed)

        self.scenario.reset_world(
            self.world,
            self.args,
            self.action_space(self.agents[0]),
            self.rng,
            self._rng_seed,
        )


        # reset agents
        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()

        # get current actions
        self.current_dilemma_action = [agent.action.s for agent in self.world.agents]
        self.current_reputation_action = [
            [0.0 for _ in agent.reputation_view] for agent in self.world.agents
        ]

        if options == "truncation":
            self.steps = 0

        obs_all = self.observe_all()
        # print(obs_all[0]['dilemma_obs'])

        # Get initial cooperative level
        coop_level, overall_reputation = self.state()
        return obs_all, coop_level

    def observe(self, agent):
        """
        observe repu info for one sepcific agent
        """
        agent_actions = (
            []
        )  # store all action since it will be used in everyone observation
        agent_recipients_idx = []
        for agent in self.world.agents:
            agent_actions.append(agent.action.s)
            # the first is self index
            agent_recipients_idx.append([agent.index] + agent.recipients)

        return self.scenario.observation(
            agent, self.world, agent_actions, agent_recipients_idx
        )

    def observe_single(self, agent_idx, agent_actions, agent_recipients_idx):
        """
        observe repu info for one sepcific agent
        """
        # print(self.world.agents[self._index_map[agent_idx]].action.s)
        return self.scenario.observation(
            self.world.agents[self._index_map[agent_idx]],
            self.world,
            agent_actions,
            agent_recipients_idx,
        )

    def observe_all(self):
        """
        observe_single current scenario info for all agents
        :return: list of observations
        """
        # agent_actions = np.zeros((len(self.possible_agents), len(self.world.agents[0].action.s)))  # store all action since it will be used in everyone observation
        agent_actions = []
        agent_recipients_idx = []

        for agent in self.world.agents:
            agent_actions.append(agent.action.s)
            # the first is self index
            agent_recipients_idx.append([agent.index] + agent.recipients)

        observations = []  # Create an empty list to store observations.
        for agent in self.world.agents:
            observation = self.observe_single(
                agent.name, np.array(agent_actions), np.array(agent_recipients_idx)
            )  # Get the observation for each agent.
            # Append the observation to the list.
            observations.append(observation)
        return observations

    def state(self) -> np.ndarray:
        """
        Return the current state of the environment
        including the current actions and reputation view of all agents
        """
        coop_level = np.mean(self.current_dilemma_action)
        overall_reputation = np.zeros(self.num_agents)
        for agent in self.world.agents:
            overall_reputation += agent.reputation_view
        overall_reputation = overall_reputation / self.num_agents
        return 1 - coop_level, overall_reputation

    def _execute_dilemma_world_step(self):
        """
        Apply the actions of all agents in this round to the environment
        """
        rewards_n = []
        recipient_actions_n = []

        # set current actions to world agents
        self.world.step(self.current_dilemma_action)

        repu_based_action = [
            [],
            [],
        ]  # measure the average reputation on cooperation and defection

        for agent in self.world.agents:
            # calculate the reward for current agent
            # each interaction pair will have a independent reward
            agent_reward, recipient_actions = self.scenario.calculate_reward(
                agent, self.world
            )
            # if agent.index ==0:
            for idx, recipient_idx in enumerate(agent.recipients):
                repu_based_action[agent.action.s[idx]].append(
                    agent.reputation_view[recipient_idx]
                )

            #     print(agent.recipients)
            self.rewards[agent.name] = agent_reward
            rewards_n.append(agent_reward)
            recipient_actions_n.append(recipient_actions)

        termination = False
        coop_level, overall_reputation = self.state()

        # get current recipient idx for each agent
        agent_recipients_idx = self.world.get_agent_recipients_idx()
        infos = {
            "coop_level": coop_level,
            "average_reputation": overall_reputation,
            "average_rewards": safe_mean(rewards_n),
            "agent_recipients_idx": agent_recipients_idx,
            "repu_based_action": [
                safe_mean(repu_based_action[0]) if repu_based_action[0] else 0,
                safe_mean(repu_based_action[1]) if repu_based_action[1] else 0,
            ],
            "recipient_actions": recipient_actions_n,
        }

        obs_n = self.observe_all()
        return obs_n, rewards_n, termination, infos

    def _execute_reputation_world_step(self):
        """
        Apply the reputation action of all agents in this round to the environment
        """
        group_observable_idx = utils.categorize_interactions(
            self.world.agents, self.args.group_num
        )
        rewards_n = []
        for agent_idx, agent in enumerate(self.world.agents):
            if (self.norm_type != "RL" and self.norm_type != "attention"):
                for obs_idx in group_observable_idx.get(agent.group_idx):
                    for j, recipient_idx in enumerate(
                        self.world.agents[obs_idx].recipients
                    ):
                        repu_action = 0.0
                        if self.norm_type == "random":
                            repu_action += (
                                np.random.choice([0, 1])
                            )
                        else:
                            repu_action += self.norm_pattern[self.norm_type][
                                int(agent.reputation_view[recipient_idx])
                            ][self.world.agents[obs_idx].action.s[j]]
                    self.current_reputation_action[agent_idx][obs_idx] = repu_action/len(
                        self.world.agents[obs_idx].recipients)

            # update the reputation view of each agent
            self.scenario.update_repu_view(
                agent,
                group_observable_idx.get(agent.group_idx),
                self.current_reputation_action[agent_idx],
            )


        strategy_reputation = [[], []]
        # coop_level, overall_reputation = self.state()
        for agent_idx, agent in enumerate(self.world.agents):
            # iterate the reputation view of each agent
            for traget_idx in group_observable_idx.get(agent.group_idx):
                if 0 in self.world.agents[traget_idx].action.s:
                    strategy_reputation[0].append(agent.reputation_view[traget_idx])
                else:
                    strategy_reputation[1].append(agent.reputation_view[traget_idx])

        self_loop_times = self.world.get_new_recipient(self._rng_seed)
        obs_n = self.observe_all()

        infos = {
            "self_loop_fraction": self_loop_times / self.num_agents,
            "strategy_reputation": [
                safe_mean(strategy_reputation[0]),
                safe_mean(strategy_reputation[1]),
            ],
        }

        return obs_n, rewards_n, infos

    def step(self, action, action_type="reputaiton", move_step=True):
        """
        Take a step in the environment
        Automatically switches control to the next agent.
        :param action: the action to take
        :param action_type: the type of action (reputaiton means only update self assessment)
        """
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        # set agent_selection to next agent
        self.agent_selection = self._agent_selector.next()
        if action_type == "reputaiton":
            self.current_reputation_action[current_idx] = action
        else:
            self.current_dilemma_action[current_idx] = action



        # do _execute_dilemma_world_step only when all agents have gone through a step once
        if next_idx == 0:
            truncation = False
            termination = False
            infos = {}
            if action_type == "reputaiton":
                obs_n, reward_n, infos = self._execute_reputation_world_step()
                if self.steps == self.max_cycles:
                    infos["epsidoe_end"] = True
                else:
                    infos["epsidoe_end"] = False
            elif action_type == "dilemma":

                obs_n, reward_n, termination, infos = self._execute_dilemma_world_step()
                if move_step:
                    self.steps += 1
                if self.steps > self.max_cycles:
                    truncation = True
                    infos["terminal_observation"] = (
                        obs_n  # store the terminal observation
                    )
                    infos["cumulative_payoffs"] = list(
                        self._cumulative_rewards.values()
                    )

            return obs_n, reward_n, termination, truncation, infos
        elif next_idx + 1 == self.num_agents:
            self._clear_rewards()
