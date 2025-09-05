import gymnasium
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
import functools
from gymnasium import spaces
import pygame
from stable_baselines3.common.utils import safe_mean
from .core import CoinGameAction
import time

MOVE_NAMES = ["RIGHT", "LEFT", "DOWN", "UP", "STAY"]
MOVES = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]], dtype=np.int32)


DEFAULT_AGENT_COLORS = [
    (230, 80, 80, 128),
    (80, 130, 230, 128),
    (0, 200, 100, 128),
    (255, 215, 0, 128),
    (200, 100, 200, 128),
    (100, 220, 220, 128),
]

COINS_FEEDBACK_COLORS = [
    (50, 180, 100),
    (240, 180, 60),
    (200, 70, 70)
]

class CoinGameEnv(AECEnv):
    metadata = {"render_modes": ["human",'rgb_array'], "name": "coin_game_v0"}

    def __init__(self, scenario, world, max_cycles, render_mode=None, args=None):
        super().__init__()

        self.render_mode = render_mode
        self.args = args
        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }
        self.steps = 0

        self.COIN_MISMATCH_PUNISHMENT = -self.num_agents / (self.num_agents - 1)

        self.COIN_REWARD =1.0

        self._seed()


        if args.use_render:
            if render_mode == "human" or render_mode == "rgb_array":
                num_agents = len(self.world.agents)

                self.agent_colors = DEFAULT_AGENT_COLORS[:num_agents]
                self.screen_size = (500, 500+100)
                self.grid_width = self.screen_size[0] // self.world.grid_size
                self.grid_height = (self.screen_size[1]-50) // self.world.grid_size
                self.coin_radius = self.grid_width // 4

                self.agent_surfaces = [
                    self._generate_robot_surface(color[:3]) for color in self.agent_colors
                ] 

                pygame.init()
                if render_mode == 'rgb_array':
                    pygame.display.set_mode(self.screen_size, flags=pygame.HIDDEN)
                    self.screen = pygame.display.get_surface()
                else:
                    self.screen = pygame.display.set_mode(self.screen_size)

                self.font = pygame.font.Font("freesansbold.ttf", 24)

    def _seed(self, seed=None):
        if seed is None:
            seed = 1
        self._rng_seed = seed
        self.rng = np.random.default_rng(seed)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        observation_space = spaces.Dict(
            {
                "dilemma": spaces.Box(
                    0,
                    1,
                    shape=(
                        self.num_agents * 2,
                        self.world.grid_size,
                        self.world.grid_size,
                    ),
                    dtype=np.float32,
                ),
                "reputation": spaces.Box(
                    0,
                    1,
                    shape=(
                        self.num_agents* 2 ,
                        self.world.grid_size,
                        self.world.grid_size*2,
                    ),
                    dtype=np.float32,
                ),
                "dilemma_vf": spaces.Box(
                    0,
                    1,
                    shape=(
                        self.num_agents * 2,
                        self.world.grid_size,
                        self.world.grid_size*2,
                    ),
                    dtype=np.float32,
                ),
                "reputation_vf": spaces.Box(
                    0,
                    1,
                    shape=(
                        self.num_agents* 2 ,
                        self.world.grid_size,
                        self.world.grid_size*3,
                    ),
                    dtype=np.float32,
                ),

            }
        )
        return observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        action_space = spaces.Dict(
            {
                "dilemma": spaces.Discrete(CoinGameAction.action_space()),
                "reputation": spaces.Discrete(2),
            }
        )
        return action_space

    def reset(self, seed=None, options='truncation'):
        if seed is not None:
            self._seed(seed)

        self.scenario.reset_world(
            self.world,
            self.action_space(self.agents[0]),
            self.rng,
            _rng_seed=self._rng_seed,
        )

        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()

        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.last_round_collected_coins = np.zeros(
            (self.num_agents,), dtype=int
        )
        self.last_round_own_coin_collected = np.zeros(
            (self.num_agents,), dtype=int
        )
        self.last_round_coin_taken_by_others = np.zeros(
            (self.num_agents,), dtype=int
        )


        self.current_dilemma_action = [agent.action.s for agent in self.world.agents]
        self.current_reputation_action = [
            [0.0 for _ in agent.reputation_view] for agent in self.world.agents
        ]

        if options == "truncation":
            self.steps = 0

        obs_all = self.observe_all()

        return obs_all

    def observe_single(self, agent_name):
        agent = self.world.agents[self._index_map[agent_name]]
        return self.scenario.observation(agent, self.world)

    def observe_all(self):
        agent_actions = []
        agent_recipients_idx = []

        for agent in self.world.agents:
            agent_actions.append(agent.action)
            agent_recipients_idx.append([self._index_map[agent.name]])

        agent_actions = np.array(agent_actions)
        agent_recipients_idx = np.array(agent_recipients_idx)

        observations = []
        for agent in self.world.agents:
            obs = self.observe_single(agent.name)
            observations.append(obs)
        return observations

    def _execute_reputation_world_step(self):


        for agent_idx, agent in enumerate(self.world.agents):
            self.scenario.update_repu_view(
                agent, self.current_reputation_action[agent_idx]
            )
        own_coin_collected_repu=[]
        collected_other_coin_repu=[]
        for i, coin in enumerate(self.own_coin_collected):
            if coin > 0:
                own_coin_collected_repu.append(np.array(self.current_reputation_action)[:,i])
        for i, coin in enumerate(self.coin_taken_by_others):
            if coin > 0:
                collected_other_coin_repu.append(np.array(self.current_reputation_action)[:,1-i])


        obs_n = self.observe_all()
        infos = {
            'own_coin_collected_repu':safe_mean(own_coin_collected_repu),
            'collected_other_coin_repu':safe_mean(collected_other_coin_repu),
        }
        rewards_n = []
        return obs_n, rewards_n, infos

    def _execute_dilemma_world_step(self):
        self.last_round_coin_poistion = self.world.coins.copy()


        self.world.step(self.current_dilemma_action)


        self.collected_coins = np.zeros(
            (self.num_agents,), dtype=int
        )
        self.own_coin_collected = np.zeros((self.num_agents,), dtype=int)
        self.coin_taken_by_others = np.zeros((self.num_agents,), dtype=int)

        agent_positions = self.world.get_agent_positions()


        for coin_idx in range(self.num_agents):
            coin_pos = self.world.coins[coin_idx]
            for agent_idx in range(self.num_agents):
                agent_pos = agent_positions[agent_idx]
                if np.all(agent_pos == coin_pos):
                    self.collected_coins[agent_idx] += 1
                    if agent_idx == coin_idx:
                        self.own_coin_collected[agent_idx] += 1
                    else:


                        self.coin_taken_by_others[coin_idx] += 1


        rewards = []

        for agent_idx, agent in enumerate(self.agents):
            reward = 0.0


            reward += self.collected_coins[agent_idx] * self.COIN_REWARD 

            reward += self.coin_taken_by_others[agent_idx] * self.COIN_MISMATCH_PUNISHMENT

            rewards.append(reward)
            self.rewards[agent] = reward
        self._accumulate_rewards()


        for coin_idx in range(self.num_agents):
            coin_collected = False
            for agent_idx in range(self.num_agents):
                if np.all(agent_positions[agent_idx] == self.world.coins[coin_idx]):
                    coin_collected = True
                    break

            if coin_collected:

                occupied_positions = {tuple(pos) for pos in agent_positions}
                for i, coin in enumerate(self.world.coins):
                    if i != coin_idx:
                        occupied_positions.add(tuple(coin))


                all_positions = [
                    (i, j) for i in range(self.world.grid_size) for j in range(self.world.grid_size)
                ]

                available_positions = [pos for pos in all_positions if pos not in occupied_positions]


                if available_positions:
                    new_pos = self.rng.choice(len(available_positions))
                    self.world.coins[coin_idx] = np.array(available_positions[new_pos], dtype=np.int32)
                else:

                    print("Warning: No available position to respawn coin.")

        agent_recipients_idx = self.world.get_agent_recipients_idx()
        obs_n = self.observe_all()
        overall_reputation = self.get_average_reputation()

        termination = False


        for agent_idx, agent in enumerate(self.world.agents):
            agent.collected_coins += self.collected_coins[agent_idx]
            agent.collected_own_coins += self.own_coin_collected[agent_idx]
            agent.coin_taken_by_others += self.coin_taken_by_others[agent_idx]


        infos = {
            "own_coin_collected": self.own_coin_collected.tolist(),
            "coin_taken_by_others": self.coin_taken_by_others.tolist(),
            "total_coins_collected": self.collected_coins.tolist(),
            "agent_recipients_idx": agent_recipients_idx,
            "step_rewards":np.array(rewards),
            "average_reputation": overall_reputation,
        }


        return obs_n, rewards, termination, infos

    def get_average_reputation(self):
        overall_reputation = np.array(
            [agent.reputation_view for agent in self.world.agents]
        )
        average_reputation = np.mean(overall_reputation, axis=0)
        return average_reputation

    def step(self, action, action_type="reputaiton", move_step=True):
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents

        self.agent_selection = self._agent_selector.next()
        if action_type == "reputaiton":
            self.current_reputation_action[current_idx] = action
        else:
            self.current_dilemma_action[current_idx] = action[0]


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
                        obs_n
                    )
                    infos["cumulative_payoffs"] = list(
                        self._cumulative_rewards.values()
                    )

            return obs_n, reward_n, termination, truncation, infos
        elif next_idx + 1 == self.num_agents:
            self._clear_rewards()


    def render(self, mode=None):
        render_mode = mode if mode is not None else self.render_mode

        agent_positions = self.world.get_agent_positions()
        coin_positions = self.last_round_coin_poistion if hasattr(self, "last_round_coin_poistion") else self.world.coins

        if render_mode == "human":
            self._draw_screen(agent_positions, coin_positions)
            pygame.display.flip()
            time.sleep(0.1)

        elif render_mode == "rgb_array":
            self._draw_screen(agent_positions, coin_positions)
            return self._get_rgb_array()

        else:
            raise NotImplementedError(f"Render mode '{render_mode}' is not supported.")


    def _draw_screen(self, agent_positions, coin_positions):
        import math

        margin_top = 70
        self.screen.fill((30, 30, 30))
        pygame.draw.rect(self.screen, (70, 70, 70), (0, 0, self.screen_size[0], margin_top))


        total_reward = sum(self._cumulative_rewards.values())
        info_text = self.font.render(f"Step: {self.steps} | Total R: {int(total_reward)}", True, (255, 255, 255))
        info_rect = info_text.get_rect(center=(self.screen.get_width() // 2, 10 + info_text.get_height() // 2))
        self.screen.blit(info_text, info_rect)


        agent_texts = []
        for agent_idx, agent in enumerate(self.world.agents):
            color = self.agent_colors[agent_idx][:3]
            reward = self._cumulative_rewards.get(agent.name, 0)
            text = self.font.render(
                f"A_{agent_idx} {agent.collected_own_coins}/{agent.collected_coins} | R: {int(reward)}",
                True,
                color,
            )
            agent_texts.append((text, color))


        spacing = 45
        total_width = sum(text.get_width() for text, _ in agent_texts) + spacing * (len(agent_texts) - 1)
        start_x = (self.screen.get_width() - total_width) // 2


        x = start_x
        for text, color in agent_texts:
            self.screen.blit(text, (x, 40))
            x += text.get_width() + spacing


        for i in range(self.world.grid_size):
            for j in range(self.world.grid_size):
                x = j * self.grid_width
                y = i * self.grid_height + margin_top
                rect = pygame.Rect(x, y, self.grid_width, self.grid_height)
                cx, cy = rect.center


                pygame.draw.rect(self.screen, (50, 50, 50), rect)
                pygame.draw.rect(self.screen, (120, 120, 120), rect, width=2)

                coords = np.array([i, j])


                coin_matches = np.where((coin_positions == coords).all(axis=1))[0]
                agent_matches = np.where((agent_positions == coords).all(axis=1))[0]


                for k, coin_idx in enumerate(coin_matches):
                    angle = 2 * math.pi * k / max(1, len(coin_matches))
                    offset_x = int(math.cos(angle) * 10)
                    offset_y = int(math.sin(angle) * 10)
                    pygame.draw.circle(
                        self.screen,
                        self.agent_colors[coin_idx][:3],
                        (cx + offset_x, cy + offset_y),
                        self.coin_radius // 1.5
                    )


                for k, agent_idx in enumerate(agent_matches):
                    robot_surface = self.agent_surfaces[agent_idx]
                    shrink_factor = 0.6
                    robot_w = int(self.grid_width * shrink_factor)
                    robot_h = int(self.grid_height * shrink_factor)
                    robot_surface_small = pygame.transform.scale(robot_surface, (robot_w, robot_h))

                    angle = 2 * math.pi * k / max(1, len(agent_matches))
                    offset_radius = 8
                    offset_x = int(math.cos(angle) * offset_radius)
                    offset_y = int(math.sin(angle) * offset_radius)

                    robot_center = (cx + offset_x, cy + offset_y + 8)
                    robot_rect = robot_surface_small.get_rect(center=robot_center)
                    self.screen.blit(robot_surface_small, robot_rect.topleft)


                    if self.last_round_collected_coins[agent_idx] < self.world.agents[agent_idx].collected_coins:
                        if self.last_round_own_coin_collected[agent_idx] < self.world.agents[agent_idx].collected_own_coins:
                            text = self.font.render(f"+{self.COIN_REWARD}", True, COINS_FEEDBACK_COLORS[0])
                            self.last_round_own_coin_collected[agent_idx] = self.world.agents[agent_idx].collected_own_coins
                        else:
                            text = self.font.render(f"+{self.COIN_REWARD}", True, COINS_FEEDBACK_COLORS[1])

                        self.last_round_collected_coins[agent_idx] = self.world.agents[agent_idx].collected_coins
                        self.screen.blit(text, (robot_center[0] - 10, robot_center[1] - self.grid_height // 2))

                    elif self.last_round_coin_taken_by_others[agent_idx] < self.world.agents[agent_idx].coin_taken_by_others:
                        text = self.font.render(f"{self.COIN_MISMATCH_PUNISHMENT}", True, COINS_FEEDBACK_COLORS[2])
                        self.last_round_coin_taken_by_others[agent_idx] = self.world.agents[agent_idx].coin_taken_by_others
                        self.screen.blit(text, (robot_center[0] - 10, robot_center[1] - self.grid_height // 2))


                    if hasattr(self, "current_dilemma_action"):
                        move = self.current_dilemma_action[agent_idx]
                        if isinstance(move, (tuple, list)) and isinstance(move[0], int):
                            move = move[0]
                        if isinstance(move, np.integer):
                            dx, dy = MOVES[move]
                            if move != 4:
                                arrow_len = 14
                                end_x = robot_center[0] + dx * arrow_len
                                end_y = robot_center[1] + dy * arrow_len
                                pygame.draw.line(
                                    self.screen, self.agent_colors[agent_idx][:3],
                                    robot_center, (end_x, end_y), width=2
                                )

                                angle = math.atan2(dy, dx)
                                head_len = 6
                                left = (
                                    end_x - head_len * math.cos(angle - math.pi / 6),
                                    end_y - head_len * math.sin(angle - math.pi / 6)
                                )
                                right = (
                                    end_x - head_len * math.cos(angle + math.pi / 6),
                                    end_y - head_len * math.sin(angle + math.pi / 6)
                                )
                                pygame.draw.polygon(self.screen, self.agent_colors[agent_idx][:3],
                                                    [left, right, (end_x, end_y)])

    def _get_rgb_array(self):
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _generate_robot_surface(self, color, size=None):
        size = size or (self.grid_width, self.grid_height)
        surf = pygame.Surface(size, pygame.SRCALPHA)
        w, h = size


        body_rect = pygame.Rect(w * 0.2, h * 0.2, w * 0.6, h * 0.6)
        pygame.draw.rect(surf, color, body_rect)


        eye_size = w // 6
        eye_x1 = int(w * 0.3)
        eye_x2 = int(w * 0.6)
        eye_y = int(h * 0.3)
        pygame.draw.rect(surf, (0, 0, 0), (eye_x1, eye_y, eye_size, eye_size))
        pygame.draw.rect(surf, (0, 0, 0), (eye_x2, eye_y, eye_size, eye_size))


        arm_w = w * 0.15
        arm_h = h * 0.3
        pygame.draw.rect(surf, color, (0, h * 0.35, arm_w, arm_h))
        pygame.draw.rect(surf, color, (w - arm_w, h * 0.35, arm_w, arm_h))


        leg_w = w * 0.2
        leg_h = h * 0.2
        pygame.draw.rect(surf, color, (w * 0.25, h - leg_h, leg_w, leg_h))
        pygame.draw.rect(surf, color, (w * 0.55, h - leg_h, leg_w, leg_h))

        return surf