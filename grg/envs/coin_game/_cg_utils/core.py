import random
import numpy as np
from .._cg_utils import utils 


class CoinGameAction:
    MOVE_NAMES = ["RIGHT", "LEFT", "DOWN", "UP", "STAY"]
    MOVES = np.array(
        [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]], dtype=np.int32
    )
    def __init__(self):
        self.s = -1


    @classmethod
    def get_move(cls, action_idx):
        return cls.MOVES[action_idx]

    @classmethod
    def action_space(cls):
        return len(cls.MOVE_NAMES)


class Agent:

    def __init__(self,idx,grid_size):

        self.name = ""
        self.index = idx
        self.grid_size = grid_size
        self.position = None

        self.action = CoinGameAction()
        self.rewards = 0.0
        self.reputation_view = np.array([], dtype=np.float32)
        self.previous_state=None
        self.collected_coins = 0
        self.collected_own_coins = 0
        self.coin_taken_by_others = 0


    def reset(self):
        flat_pos = np.random.randint(self.grid_size**2)
        self.position = np.array([flat_pos // self.grid_size, flat_pos % self.grid_size], dtype=np.int32)
        self.previous_state = None
        self.collected_coins = 0
        self.collected_own_coins = 0
        self.coin_taken_by_others = 0


    def move(self, move_delta):
        self.position = (self.position + move_delta) % self.grid_size


class World:

    def __init__(self,num_agents, grid_size):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.agents = [Agent(i, grid_size) for i in range(num_agents)]
        self.coins = np.zeros((num_agents, 2), dtype=np.int32)
        self.set_repu=True


    def reset(self):
        for agent in self.agents:
            agent.reset()
        for i in range(self.num_agents):
            flat_pos = np.random.randint(self.grid_size**2)
            self.coins[i] = np.array([flat_pos // self.grid_size, flat_pos % self.grid_size], dtype=np.int32)

    def get_agent_positions(self):
        return np.array([agent.position for agent in self.agents], dtype=np.int32)

    def step(self, actions):
        for idx, agent in enumerate(self.agents):
            agent.action.s = actions[idx]
            move = CoinGameAction.get_move(agent.action.s)
            agent.move(move)

    def get_agent_recipients_idx(self):
        recipients = []
        for i, agent in enumerate(self.agents):
            recipient_idx = (i + 1) % self.num_agents
            recipients.append(recipient_idx)
        return recipients