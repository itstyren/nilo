import random
import numpy as np
from .._md_utils import utils


class Action:
    """
    A class for the actions in the donation game
    """

    def __init__(self):
        # dilemma strategy action (0-->cooperate, 1-->defection)
        self.s = np.array([], dtype=np.int16)


class Agent:
    """
    A class for the agents in the donation game
    """

    def __init__(self, args):
        # name & index
        self.name = ""
        self.index = 0
        self.group_idx = None

        # dilemma action and reputation assignment
        self.action = Action()

        self.rewards = 0.0
        self.recipients = None
        self.num_recipients = args.num_recipients
        # Agent have reputation view of all other agents (including itself)
        self.reputation_view = np.array([], dtype=np.float32)


class World:
    """
    A world class for the donation game
    """

    def __init__(self, initial_ratio, game_parameter):
        self.initial_ratio = initial_ratio

        self.agents = []
        self.payoff_matrix = game_parameter
        self._pair_time = 0

    def initialize_network(self, args,seed):
        """
        Initialize the network of agents
        """
        self.num_recipients = args.num_recipients
        self.in_group_prob = args.in_group_prob
        if args.network_type == "mixed":
            utils.assign_group_indices(self.agents, args.group_num)
            # utils.gen_random_donor_recipient_pairs(self.agents, num_recipients=args.num_recipients, in_group_prob=args.in_group_prob)
            utils.gen_connected_interaction_pairs(
                self.agents,
                num_recipients=self.num_recipients,
                in_group_prob=self.in_group_prob,
                seed=seed,
            )
            utils.display_agent_connections(self.agents)
            utils.categorize_interactions(self.agents, args.group_num)

    def step(self, actions):
        """
        update agent actions
        """
        for idx, agent in enumerate(self.agents):
            agent.action.s = actions[idx]

    def get_new_recipient(self,_rng_seed):
        """
        Get new recipient for each agent

        :param np_rng: numpy random number generator
        :return: new recipient for each agent
        """
        self._pair_time += 1
        return utils.gen_connected_interaction_pairs(
            self.agents,
            num_recipients=self.num_recipients,
            in_group_prob=self.in_group_prob,
            call_times=self._pair_time,
            seed=_rng_seed,
            
        )

    def get_agent_recipients_idx(self):
        """
        Get the recipient index for each agent
        """
        agent_recipients_idx = []
        for agent in self.agents:
            agent_recipients_idx.append(agent.recipients)

        return np.array(agent_recipients_idx)