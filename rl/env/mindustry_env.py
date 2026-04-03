"""
MindustryEnv — Gymnasium environment wrapping the Mimi Gateway TCP mod.

Observation: Dict{"grid": (4,31,31), "features": (47,)}
Action:      MultiDiscrete([7, 9]) — [action_type, arg]

action_type:
  0 = WAIT
  1 = MOVE (arg = direction 0-7)
  2 = BUILD_TURRET  (arg = slot 0-8)
  3 = BUILD_WALL    (arg = slot 0-8)
  4 = BUILD_POWER   (arg = slot 0-8)
  5 = BUILD_DRILL   (arg = slot 0-8)
  6 = REPAIR        (arg = slot 0-8)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

from rl.env.mimi_client import MimiClient
from rl.env.spaces import (
    make_obs_space, make_action_space, parse_observation,
    BLOCK_TURRET, BLOCK_WALL, BLOCK_POWER, BLOCK_DRILL,
)
from rl.rewards.multi_objective import compute_reward


DEFAULT_TRAINING_MAPS = [
    "Ancient Caldera", "Archipelago", "Debris Field", "Domain", "Fork", "Fortress",
    "Glacier", "Islands", "Labyrinth", "Maze", "Molten Lake", "Mud Flats",
    "Passage", "Shattered", "Tendrils", "Triad", "Veins", "Wasteland",
]


class MindustryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        max_steps: int = 5000,
        client: Optional[MimiClient] = None,
        maps: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.observation_space = make_obs_space()
        self.action_space = make_action_space()
        self.max_steps = max_steps

        self._host = host
        self._port = port
        self._client: Optional[MimiClient] = client
        self._maps: list[str] = maps if maps is not None else DEFAULT_TRAINING_MAPS
        self._map_index: int = 0

        self._prev_state: Optional[Dict[str, Any]] = None
        self._step_count: int = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        if self._client is None:
            self._client = MimiClient(self._host, self._port)

        map_name = self._maps[self._map_index % len(self._maps)]
        self._map_index += 1

        self._client.send_command(f"RESET;{map_name}")

        state = self._client.receive_state()
        if state is None:
            raise RuntimeError("Failed to receive initial state from Mindustry server")
        self._prev_state = state
        self._step_count = 0

        obs = parse_observation(state)
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("Must call reset() before step()")

        action_type = int(action[0])
        arg = int(action[1])

        self._execute_action(action_type, arg)

        state = self._client.receive_state()
        if state is None:
            raise RuntimeError("Lost connection to Mindustry server during step")
        self._step_count += 1

        obs = parse_observation(state)
        core_hp = float(state.get("core", {}).get("hp", 0.0))
        player_alive = bool(state.get("player", {}).get("alive", False))
        terminated = core_hp <= 0.0 or not player_alive
        truncated = self._step_count >= self.max_steps

        reward = compute_reward(self._prev_state, state, done=terminated)
        self._prev_state = state

        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _execute_action(self, action_type: int, arg: int) -> None:
        if action_type == 0:
            pass
        elif action_type == 1:
            self._client.send_command(f"PLAYER_MOVE;{arg % 8}")
        elif action_type == 2:
            self._client.send_command(f"PLAYER_BUILD;{BLOCK_TURRET};{arg}")
        elif action_type == 3:
            self._client.send_command(f"PLAYER_BUILD;{BLOCK_WALL};{arg}")
        elif action_type == 4:
            self._client.send_command(f"PLAYER_BUILD;{BLOCK_POWER};{arg}")
        elif action_type == 5:
            self._client.send_command(f"PLAYER_BUILD;{BLOCK_DRILL};{arg}")
        elif action_type == 6:
            self._client.send_command(f"REPAIR_SLOT;{arg}")
        else:
            raise ValueError(f"Invalid action_type: {action_type}. Must be 0-6")
