import time

import torch
import pygame

from grg.envs.coin_game._cg_utils.utils import same_coords
from grg.envs.coin_game._cg_utils.render_utils import draw_circle_alpha
from grg.envs.coin_game.vectorized_env import VectorizedEnv


class CoinGameStats:
    def __init__(self, num_players: int, device: torch.device):
        self.episode_count = 0
        self.num_players = num_players
        self.device = device
        self.batch_size = None
        self.running_rewards_per_batch = torch.zeros(num_players, device=device)
        self.own_coin_count = torch.zeros(num_players, device=device)
        self.other_coin_count = torch.zeros(num_players, device=device)
        self.total_coin_count = torch.zeros(num_players, device=device)
        self.running_info = torch.zeros((num_players, num_players), device=device)
        self.logs = []

    def update(self, rewards: torch.Tensor, info: dict):
        if self.batch_size is None:
            self.batch_size = rewards.size(0)

        self.running_info += info

        for agent_idx in range(self.num_players):
            self.own_coin_count[agent_idx] += info[agent_idx, agent_idx].item()
            self.other_coin_count[agent_idx] += (
                info[agent_idx, 0:agent_idx].sum()
                + info[agent_idx, agent_idx + 1 :].sum()
            )
            self.total_coin_count[agent_idx] += info[agent_idx, :].sum()
            self.running_rewards_per_batch[agent_idx] += (
                rewards[:, agent_idx].detach().mean()
            )

    def _log_episode_agent(self, idx):
        agent_logs = {
            f"coins_taken_by_{i}": self.running_info[i, idx].item() / self.batch_size
            for i in range(self.num_players)
        }
        agent_logs.update(
            {
                f"reward": self.running_rewards_per_batch[idx].item(),
                "own_coin_count": self.own_coin_count[idx].item() / self.batch_size,
                "other_coin_count": self.other_coin_count[idx].item() / self.batch_size,
                "total_coin_count": self.total_coin_count[idx].item() / self.batch_size,
            }
        )
        return agent_logs

    def log_episode(self, extra_logs: list or dict = None, verbose: bool = False):
        log = {
            f"player_{idx}": self._log_episode_agent(idx)
            for idx in range(self.num_players)
        }
        if extra_logs is not None:
            if isinstance(extra_logs, list):
                for idx, extra_log in enumerate(extra_logs):
                    log[f"player_{idx}"].update(extra_log)
            elif isinstance(extra_logs, dict):
                log.update(extra_logs)
            else:
                raise ValueError("Extra logs must be a list or a dictionary.")
        self.logs.append(log)
        print(f"EPISODE {self.episode_count}:", self.logs[-1]) if verbose else None
        self._reset()
        self.episode_count += 1

    def _reset(self):
        self.running_rewards_per_batch = torch.zeros(
            self.num_players, device=self.device
        )
        self.own_coin_count = torch.zeros(self.num_players, device=self.device)
        self.other_coin_count = torch.zeros(self.num_players, device=self.device)
        self.total_coin_count = torch.zeros(self.num_players, device=self.device)
        self.running_info = torch.zeros(
            (self.num_players, self.num_players), device=self.device
        )


class CoinGame(VectorizedEnv):

    MOVE_NAMES = ["RIGHT", "LEFT", "DOWN", "UP", "STAY"]
    MOVES = torch.stack(
        [
            torch.LongTensor([0, 1]),
            torch.LongTensor([0, -1]),
            torch.LongTensor([1, 0]),
            torch.LongTensor([-1, 0]),
            torch.LongTensor([0, 0]),
        ],
        dim=0,
    )
    COIN_REWARD = 1.0
    # COIN_MISMATCH_PUNISHMENT = -2

    # Class-level constant for agent colors (RGBA)
    DEFAULT_AGENT_COLORS = [
        (255, 0, 0, 128),     # Red
        (0, 0, 255, 128),     # Blue
        (0, 255, 0, 128),     # Green
        (255, 255, 0, 128),   # Yellow
        (255, 0, 255, 128),   # Magenta
        (0, 255, 255, 128),   # Cyan
    ]

    def __init__(
        self,
        batch_size: int,
        max_steps: int,
        device: torch.device,
        num_agents: int = 2,
        grid_size: int = 3,
        enable_render: bool = False,
    ):
        super().__init__(batch_size, max_steps, device, num_agents, enable_render)
        self.grid_size = grid_size
        self.COIN_MISMATCH_PUNISHMENT = -num_agents / (num_agents - 1)
        self.agent_colors = self.DEFAULT_AGENT_COLORS[:num_agents]
        self.agent_positions = torch.zeros(
            (self.batch_size, self.num_agents, 2), dtype=torch.long, device=self.device
        )
        self.coin_positions = torch.zeros(
            (self.batch_size, self.num_agents, 2), dtype=torch.long, device=self.device
        )
        self.moves = self.MOVES.to(self.device)
        self.reset()

        if enable_render:
            pygame.init()
            self.last_actions = torch.zeros((self.batch_size, self.num_agents), dtype=torch.long, device=self.device)

            self.screen_size = (500, 500 + 50)
            self.grid_width = self.screen_size[0] // self.grid_size
            self.grid_height = (self.screen_size[1]-20) // self.grid_size
            self.coin_radius = self.grid_width // 4
            self.screen = pygame.display.set_mode(self.screen_size)
            self.font = pygame.font.Font("freesansbold.ttf", 24)

            if self.num_agents > len(self.DEFAULT_AGENT_COLORS):
                raise ValueError(
                    f"Too many agents ({self.num_agents}) for the available colors ({len(self.DEFAULT_AGENT_COLORS)})."
                )

            # Generate agent surfaces using procedurally generated robot avatars
            self.agent_surfaces = [
                self._generate_robot_surface(color[:3]) for color in self.agent_colors
            ]

    def reset(self):
        self.step_count = 0

        flat_pos = torch.randint(
            self.grid_size**2, size=(self.batch_size, self.num_agents * 2)
        ).to(self.device)
        self.agent_positions = self._get_coords(flat_pos[:, : self.num_agents]).to(
            self.device
        )
        self.coin_positions = self._get_coords(flat_pos[:, self.num_agents :]).to(
            self.device
        )
        return self._generate_observations()

    def _get_coords(self, position_val: torch.Tensor) -> torch.Tensor:

        return torch.stack(
            [position_val // self.grid_size, position_val % self.grid_size], dim=-1
        )

    def _flatten_coords(self, coords: torch.Tensor) -> torch.Tensor:

        return (coords[..., 0] * self.grid_size + coords[..., 1]).to(self.device)

    def _generate_coins(self, collected_coins: torch.Tensor):

        for coin_idx in range(self.num_agents):
            mask = collected_coins[:, coin_idx]
            new_coin_pos_flat = torch.randint(
                self.grid_size**2, size=(self.batch_size,)
            ).to(self.device)[mask]
            new_coin_pos = self._get_coords(new_coin_pos_flat)
            self.coin_positions[mask, coin_idx, :] = new_coin_pos

    def _generate_robot_surface(self, color, size=None):
        """Create a pixel-style robot avatar surface."""
        size = size or (self.grid_width, self.grid_height)
        surf = pygame.Surface(size, pygame.SRCALPHA)
        w, h = size

        # Robot body
        body_rect = pygame.Rect(w * 0.2, h * 0.2, w * 0.6, h * 0.6)
        pygame.draw.rect(surf, color, body_rect)

        # Eyes (black squares)
        eye_size = w // 6
        eye_x1 = int(w * 0.3)
        eye_x2 = int(w * 0.6)
        eye_y = int(h * 0.3)
        pygame.draw.rect(surf, (0, 0, 0), (eye_x1, eye_y, eye_size, eye_size))
        pygame.draw.rect(surf, (0, 0, 0), (eye_x2, eye_y, eye_size, eye_size))

        # Arms
        arm_w = w * 0.15
        arm_h = h * 0.3
        pygame.draw.rect(surf, color, (0, h * 0.35, arm_w, arm_h))
        pygame.draw.rect(surf, color, (w - arm_w, h * 0.35, arm_w, arm_h))

        # Legs
        leg_w = w * 0.2
        leg_h = h * 0.2
        pygame.draw.rect(surf, color, (w * 0.25, h - leg_h, leg_w, leg_h))
        pygame.draw.rect(surf, color, (w * 0.55, h - leg_h, leg_w, leg_h))

        return surf




    def _generate_observations(self):
        if self.device == torch.device("mps"):
            device = "cpu"
        else:
            device = self.device

        state = torch.zeros(
            (self.batch_size, self.num_agents * 2, self.grid_size * self.grid_size),
            dtype=torch.float,
            device=device,
        )

        flat_agent_positions = self._flatten_coords(self.agent_positions).to(device)
        flat_coin_positions = self._flatten_coords(self.coin_positions).to(device)

        for i in range(self.num_agents):
            state[:, i].scatter_(1, flat_agent_positions[:, i : i + 1], 1)  
            state[:, i + self.num_agents].scatter_(
                1, flat_coin_positions[:, i : i + 1], 1                    
            )

        state = state.view(
            self.batch_size, self.num_agents * 2, self.grid_size, self.grid_size
        ).to(self.device)
        return [state.roll(-i * 2, dims=1) for i in range(self.num_agents)]

    def step(self, actions: torch.Tensor):
        self.last_actions = actions.clone()

        collected_coins = torch.zeros(
            (self.batch_size, self.num_agents), dtype=torch.bool, device=self.device
        )
        total_coin_counts = torch.zeros(
            (self.num_agents, self.num_agents), dtype=torch.long, device=self.device
        )
        rewards = torch.zeros(
            (self.batch_size, self.num_agents), dtype=torch.float, device=self.device
        )
        moves = torch.index_select(self.moves, 0, actions.view(-1)).view(
            self.batch_size, self.num_agents, 2
        )
        # self.agent_positions = (self.agent_positions + moves).clamp(0, self.grid_size - 1)
        self.agent_positions = (self.agent_positions + moves) % self.grid_size

        # Check coin collections and compute rewards
        for coin_idx in range(self.num_agents):
            coin_pos = self.coin_positions[:, coin_idx, :]
            collected_coins[:, coin_idx] = (
                same_coords(self.agent_positions.transpose(0, 1), coin_pos).sum(dim=0)
                > 0
            )
            coin_count_per_batch = same_coords(
                self.agent_positions.transpose(0, 1), coin_pos
            ).T
            total_coin_counts[:, coin_idx] = coin_count_per_batch.sum(
                dim=0
            )  # Sum across batches
            rewards += coin_count_per_batch * self.COIN_REWARD

            mismatched_coin_count = coin_count_per_batch[:, 0:coin_idx].sum(
                dim=1
            ) + coin_count_per_batch[:, coin_idx + 1 : self.num_agents].sum(dim=1)
            rewards[:, coin_idx] += (
                mismatched_coin_count * self.COIN_MISMATCH_PUNISHMENT
            )
            # if mismatched_coin_count!=0:
            #     breakpoint()

        self._generate_coins(collected_coins)
        # observations = [self._generate_observations(agent_idx) for agent_idx in range(self.num_agents)]
        observations = self._generate_observations()
        self.step_count += 1
        done = (self.step_count >= self.max_steps) * torch.ones(
            self.batch_size, dtype=torch.bool, device=self.device
        )
        return observations, rewards, done, total_coin_counts

    def render(self, batch_idx, delay: float = 1.0):
        if not self.enable_render:
            raise RuntimeError(
                "Cannot render environment that has not been initialized with enable_render=True."
            )
        agent_positions = self.agent_positions[batch_idx]
        coin_positions = self.coin_positions[batch_idx]
        self._draw_grid(agent_positions, coin_positions)
        self._update_screen()
        time.sleep(delay)

    def _draw_grid(self, agent_positions, coin_positions):
        import math
        self.screen.fill((30, 30, 30))  # Dark background

        # --- Scoreboard ---
        margin_top = 40
        pygame.draw.rect(self.screen, (40, 40, 40), (0, 0, self.screen_size[0], margin_top))

        for agent_idx in range(self.num_agents):
            color = self.agent_colors[agent_idx][:3]
            coins = int(self.stats.total_coin_count[agent_idx].item()) if hasattr(self, "stats") else 0
            text = self.font.render(f"Agent {agent_idx}: {coins}", True, color)
            self.screen.blit(text, (20 + agent_idx * 150, 10))

        # --- Grid ---
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                coords = torch.LongTensor([i, j]).to(self.device)
                coin_matches = torch.nonzero(same_coords(coin_positions, coords))
                agent_matches = torch.nonzero(same_coords(agent_positions, coords))

                grid_pos = (j * self.grid_width, i * self.grid_height + margin_top)
                rect = pygame.Rect(*grid_pos, self.grid_width, self.grid_height)
                cx, cy = rect.center

                # Draw tile background
                pygame.draw.rect(self.screen, (50, 50, 50), rect)
                pygame.draw.rect(self.screen, (120, 120, 120), rect, width=2)

                # Draw coins with offset if multiple in same cell
                for k, match in enumerate(coin_matches):
                    coin_idx = match.item()
                    angle = 2 * math.pi * k / max(1, len(coin_matches))
                    offset_x = int(math.cos(angle) * 10)
                    offset_y = int(math.sin(angle) * 10)
                    pygame.draw.circle(
                        self.screen,
                        self.agent_colors[coin_idx][:3],
                        (cx + offset_x, cy + offset_y),
                        self.coin_radius // 1.5
                    )



                # --- Draw agents (if present) ---
                if agent_matches.size(0) > 0:
                    for k, match in enumerate(agent_matches):
                        agent_idx = match.item()
                        robot_surface = self.agent_surfaces[agent_idx]

                        # Shrink the robot sprite
                        shrink_factor = 0.6
                        robot_w = int(self.grid_width * shrink_factor)
                        robot_h = int(self.grid_height * shrink_factor)
                        robot_surface_small = pygame.transform.scale(robot_surface, (robot_w, robot_h))

                        # Offset position if multiple agents on same tile
                        angle = 2 * math.pi * k / max(1, len(agent_matches))
                        offset_radius = 8
                        offset_x = int(math.cos(angle) * offset_radius)
                        offset_y = int(math.sin(angle) * offset_radius)

                        # Apply margin offset and position
                        robot_center = (cx + offset_x, cy + offset_y + 8)  # slightly lower
                        robot_rect = robot_surface_small.get_rect(center=robot_center)
                        self.screen.blit(robot_surface_small, robot_rect.topleft)

                        # Optional: show feedback text for this agent
                        if coin_matches.size(0) > 0:
                            coin_indices = [m.item() for m in coin_matches]
                            if agent_idx in coin_indices:
                                text = self.font.render("+1", True, (0, 255, 0))
                                self.screen.blit(text, (robot_center[0] - 10, robot_center[1] - self.grid_height // 2))
                            elif len(coin_matches) > 0:
                                text = self.font.render("-1", True, (255, 0, 0))
                                self.screen.blit(text, (robot_center[0] - 10, robot_center[1] - self.grid_height // 2))

                        # --- Draw move direction arrow ---
                        if hasattr(self, "last_actions"):
                            move = self.last_actions[0, agent_idx].item()
                            arrow_len = 14
                            arrow_color = self.agent_colors[agent_idx][:3]
                            if move in [0, 1, 2, 3]:  # Not STAY
                                dx, dy = self.MOVES[move].cpu().tolist()

                                end_x = robot_center[0] + dx * arrow_len
                                end_y = robot_center[1] + dy * arrow_len

                                pygame.draw.line(self.screen, arrow_color, robot_center, (end_x, end_y), width=2)
                                
                                
                                # Arrowhead
                                if dx != 0 or dy != 0:
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
                                    pygame.draw.polygon(self.screen, arrow_color, [left, right, (end_x, end_y)])






    @staticmethod
    def _update_screen():
        pygame.display.flip()


if __name__ == "__main__":
    device = torch.device("cpu")
    num_players = 2
    grid_size = 3
    bsz = 1
    max_steps = 100

    test_env = CoinGame(
        bsz,
        max_steps,
        device,
        num_agents=num_players,
        grid_size=grid_size,
        enable_render=True,
    )
    test_env.stats = CoinGameStats(num_players, device)
    total_rews = 0
    for _ in range(max_steps):
        actions = torch.randint(
            0,
            test_env.MOVES.shape[0],
            (bsz, num_players),
            dtype=torch.long,
            device=device,
        )
        obs, rew, _, info = test_env.step(actions)
        test_env.stats.update(rew, info)
        test_env.render(0, delay=1.0)

        pygame.event.pump()
