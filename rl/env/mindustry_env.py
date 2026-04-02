"""
MindustryEnv — Gymnasium environment wrapping the Mimi Gateway TCP mod.

Observation: Dict{"grid": (4,31,31), "features": (43,)}
Action:      Dict{"action_type": Discrete(8), "x": Box(1,), "y": Box(1,)}
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

from rl.env.mimi_client import MimiClient
from rl.env.spaces import make_obs_space, make_action_space, parse_observation
from rl.rewards.multi_objective import compute_reward


class MindustryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        max_steps: int = 5000,
        client: Optional[MimiClient] = None,
    ) -> None:
        super().__init__()
        self.observation_space = make_obs_space()
        self.action_space = make_action_space()
        self.max_steps = max_steps

        self._host = host
        self._port = port
        self._client: Optional[MimiClient] = client

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

        self._client.message("RESET")
        state = self._client.receive_state()
        self._prev_state = state
        self._step_count = 0

        obs = parse_observation(state)
        return obs, {}

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action_type = int(action["action_type"])
        x = int(action["x"][0])
        y = int(action["y"][0])

        self._execute_action(action_type, x, y)

        state = self._client.receive_state()
        self._step_count += 1

        obs = parse_observation(state)
        core_hp = float(state.get("core", {}).get("hp", 0.0))
        terminated = core_hp <= 0.0
        truncated = self._step_count >= self.max_steps

        reward = compute_reward(self._prev_state, state, done=terminated)
        self._prev_state = state

        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _execute_action(self, action_type: int, x: int, y: int) -> None:
        if action_type == 0:
            self._client.message("WAIT")
        elif action_type == 1:
            self._client.build("duo", x, y, rotation=0)
        elif action_type == 2:
            self._client.build("wall", x, y, rotation=0)
        elif action_type == 3:
            self._client.build("solar-panel", x, y, rotation=0)
        elif action_type == 4:
            self._client.repair(x, y)
        elif action_type == 5:
            unit_id = self._get_first_friendly_id()
            if unit_id is not None:
                self._client.move_unit(unit_id, x, y)
        elif action_type == 6:
            unit_id = self._get_first_friendly_id()
            if unit_id is not None:
                self._client.attack(unit_id, x, y)
        elif action_type == 7:
            self._client.spawn_unit(x, y, unit_type="poly")

    def _get_first_friendly_id(self) -> Optional[int]:
        if self._prev_state is None:
            return None
        units = self._prev_state.get("friendlyUnits", [])
        if units:
            return int(units[0].get("id", 0))
        return None
