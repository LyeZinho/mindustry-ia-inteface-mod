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

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

_log = logging.getLogger(__name__)

from rl.env.mimi_client import MimiClient
from rl.env.spaces import (
    make_obs_space, make_action_space, parse_observation,
    compute_action_mask, NUM_ACTION_TYPES, NUM_SLOTS,
    BLOCK_TURRET, BLOCK_WALL, BLOCK_POWER, BLOCK_DRILL,
)
from rl.rewards.multi_objective import compute_reward, _detect_new_drills


DEFAULT_TRAINING_MAPS = [
    "Ancient Caldera", "Archipelago", "Debris Field", "Domain", "Fork", "Fortress",
    "Glacier", "Islands", "Labyrinth", "Maze", "Molten Lake", "Mud Flats",
    "Passage", "Shattered", "Tendrils", "Triad", "Veins", "Wasteland",
]


_MAX_RESET_RETRIES = 5
_RESET_BACKOFF = 2.0


class MindustryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        tcp_port: Optional[int] = None,
        max_steps: int = 5000,
        client: Optional[MimiClient] = None,
        maps: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.observation_space = make_obs_space()
        self.action_space = make_action_space()
        self.max_steps = max_steps

        self._host = host
        self._port = tcp_port if tcp_port is not None else port
        self._client: Optional[MimiClient] = client
        self._maps: list[str] = maps if maps is not None else DEFAULT_TRAINING_MAPS
        self._map_index: int = 0

        self._prev_state: Optional[Dict[str, Any]] = None
        self._step_count: int = 0
        self._action_history: list[int] = []  # Track last N actions for inactivity detection

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        map_name = self._maps[self._map_index % len(self._maps)]
        self._map_index += 1

        backoff = _RESET_BACKOFF
        last_exc: Optional[Exception] = None

        for attempt in range(1, _MAX_RESET_RETRIES + 1):
            try:
                if self._client is None:
                    self._client = MimiClient(self._host, self._port)

                self._client.send_command(f"RESET;{map_name}")
                state = self._client.receive_state()

                if state is None:
                    raise OSError("Server returned EOF after RESET")

                self._prev_state = state
                self._step_count = 0
                self._action_history = []
                return parse_observation(state), {}

            except OSError as exc:
                last_exc = exc
                _log.warning(
                    "reset() attempt %d/%d failed: %s",
                    attempt, _MAX_RESET_RETRIES, exc,
                )
                try:
                    if self._client is not None:
                        self._client.close()
                except OSError:
                    pass
                self._client = None

                if attempt < _MAX_RESET_RETRIES:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)

        raise RuntimeError(
            f"Failed to reset after {_MAX_RESET_RETRIES} attempts: {last_exc}"
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("Must call reset() before step()")

        action_type = int(action[0])
        arg = int(action[1])

        t0 = time.perf_counter()
        try:
            self._execute_action(action_type, arg)
            state = self._client.receive_state()
        except OSError as exc:
            _log.warning("Connection lost during step (%s); ending episode.", exc)
            self._client = None
            empty_obs = parse_observation(self._prev_state or {})
            return empty_obs, -1.0, True, False, {"connection_error": str(exc)}

        step_latency_ms = (time.perf_counter() - t0) * 1000.0

        if state is None:
            _log.warning("Server closed connection during step; ending episode.")
            self._client = None
            empty_obs = parse_observation(self._prev_state or {})
            return empty_obs, -1.0, True, False, {"connection_error": "EOF"}

        self._step_count += 1

        obs = parse_observation(state)
        core_hp = float(state.get("core", {}).get("hp", 0.0))
        player_alive = bool(state.get("player", {}).get("alive", False))
        terminated = core_hp <= 0.0 or not player_alive
        truncated = self._step_count >= self.max_steps

        # Update action history BEFORE reward computation so current action
        # is included in streak checks (fixes off-by-one timing bug)
        self._action_history.append(action_type)
        if len(self._action_history) > 10:
            self._action_history.pop(0)

        reward = compute_reward(
            self._prev_state,
            state,
            done=terminated,
            action_taken=action_type,
            action_history=self._action_history,
        )

        penalty_a_triggered, penalty_b_triggered = self._compute_penalties(
            self._prev_state, state
        )
        drills_built = _detect_new_drills(self._prev_state, state)

        self._prev_state = state

        info: Dict[str, Any] = {
            "step_latency_ms": step_latency_ms,
            "resources": state.get("resources", {}),
            "power": state.get("power", {}),
            "buildings": len(state.get("buildings", [])),
            "units": len(state.get("friendlyUnits", [])),
            "build_failed": bool(state.get("actionFailed", False)),
            "drills_built_this_step": drills_built,
            "penalty_a_triggered": penalty_a_triggered,
            "penalty_b_triggered": penalty_b_triggered,
            "action_taken_index": action_type,
            "step_count": self._step_count,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def action_masks(self) -> np.ndarray:
        """Return action mask for MaskablePPO. Shape: (16,) = 7 action_types + 9 slots."""
        if self._prev_state is None:
            return np.ones(NUM_ACTION_TYPES + NUM_SLOTS, dtype=np.bool_)
        return compute_action_mask(self._prev_state)

    def _compute_penalties(
        self, prev_state: Dict[str, Any], curr_state: Dict[str, Any]
    ) -> Tuple[int, int]:
        from rl.rewards.multi_objective import _detect_action_repetition_penalty, _detect_resource_bleeding_penalty

        def _total_resources(state: Dict[str, Any]) -> float:
            return sum(float(v) for v in state.get("resources", {}).values())

        resources_delta = _total_resources(curr_state) - _total_resources(prev_state)
        penalty_a = _detect_action_repetition_penalty(self._action_history, resources_delta)
        penalty_b = _detect_resource_bleeding_penalty(prev_state, curr_state)
        return int(penalty_a != 0.0), int(penalty_b != 0.0)

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
