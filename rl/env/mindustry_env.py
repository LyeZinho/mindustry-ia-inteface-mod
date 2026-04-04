"""
MindustryEnv — Gymnasium environment wrapping the Mimi Gateway TCP mod.

Observation: Dict{"grid": (4,31,31), "features": (83,)}
Action:      MultiDiscrete([12, 9]) — [action_type, arg]

action_type:
  0  = WAIT
  1  = MOVE              (arg = direction 0-7)
  2  = BUILD_TURRET      (arg = slot 0-8)
  3  = BUILD_WALL        (arg = slot 0-8)
  4  = BUILD_POWER       (arg = slot 0-8)
  5  = BUILD_DRILL       (arg = slot 0-8)
  6  = REPAIR            (arg = slot 0-8)
  7  = BUILD_CONVEYOR    (arg = slot 0-8)
  8  = BUILD_GRAPHITE_PRESS (arg = slot 0-8)
  9  = BUILD_SILICON_SMELTER (arg = slot 0-8)
  10 = BUILD_COMBUSTION_GEN (arg = slot 0-8)
  11 = BUILD_PNEUMATIC_DRILL (arg = slot 0-8)
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
    ACTION_REGISTRY, ACTION_WAIT, ACTION_MOVE, ACTION_REPAIR,
    BLOCK_TURRET, BLOCK_WALL, BLOCK_POWER, BLOCK_DRILL,
    SLOT_DX, SLOT_DY, GRID_SIZE,
)
from rl.rewards.multi_objective import compute_reward, _detect_new_drills
from rl.optimization.worker import OptimizationWorker


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
        self._action_history: list[int] = []
        self._global_timestep: int = 0

        self._opt_worker = OptimizationWorker(update_every=5)
        self._opt_worker.start()

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
                opt_signals = self._opt_worker.get_result()
                threat_map, power_map, build_map = self._compute_spatial_maps(state, opt_signals)
                return parse_observation(state, opt_signals=opt_signals, threat_map=threat_map, power_map=power_map, build_map=build_map), {}

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
        except Exception as exc:
            _log.error("Unexpected error during step (%s); ending episode.", exc)
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
        self._global_timestep += 1

        self._opt_worker.update(state)
        opt_signals = self._opt_worker.get_result()
        threat_map, power_map, build_map = self._compute_spatial_maps(state, opt_signals)
        obs = parse_observation(state, opt_signals=opt_signals, threat_map=threat_map, power_map=power_map, build_map=build_map)
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
            timestep=self._global_timestep,
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
        self._opt_worker.stop()

    def action_masks(self) -> np.ndarray:
        """Return action mask for MaskablePPO. Shape: (21,) = 12 action_types + 9 slots."""
        if self._prev_state is None:
            return np.ones(NUM_ACTION_TYPES + NUM_SLOTS, dtype=np.bool_)

        resource_mask = compute_action_mask(self._prev_state)

        from rl.rewards.multi_objective import apply_curriculum_action_mask, CURRICULUM_ENABLED
        if CURRICULUM_ENABLED:
            curriculum_mask = apply_curriculum_action_mask(self._global_timestep)
            for i in range(NUM_ACTION_TYPES):
                if not curriculum_mask[i]:
                    resource_mask[i] = False

        return resource_mask

    def _compute_spatial_maps(
        self, state: Dict[str, Any], opt_signals: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        threat_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        power_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        build_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        for enemy in state.get("enemies", []):
            ex = int(enemy.get("x", 0))
            ey = int(enemy.get("y", 0))
            ehp = float(enemy.get("hp", 0.0))
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    dist = max(1.0, ((x - ex) ** 2 + (y - ey) ** 2) ** 0.5)
                    threat_map[y, x] = min(1.0, threat_map[y, x] + ehp / dist)

        power_nodes = state.get("powerNodes", [])
        for node in power_nodes:
            nx = int(node.get("x", 0))
            ny = int(node.get("y", 0))
            nr = int(node.get("range", 10))
            for y in range(max(0, ny - nr), min(GRID_SIZE, ny + nr + 1)):
                for x in range(max(0, nx - nr), min(GRID_SIZE, nx + nr + 1)):
                    if ((x - nx) ** 2 + (y - ny) ** 2) <= nr * nr:
                        power_map[y, x] = 1.0

        for tile in state.get("oreGrid", []):
            if isinstance(tile, (list, tuple)) and len(tile) >= 2:
                ox, oy = int(tile[0]), int(tile[1])
            else:
                continue
            if 0 <= ox < GRID_SIZE and 0 <= oy < GRID_SIZE:
                build_map[oy, ox] = min(1.0, build_map[oy, ox] + 0.8)

        core = state.get("core", {})
        cx = int(core.get("x", GRID_SIZE // 2))
        cy = int(core.get("y", GRID_SIZE // 2))
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                dist = max(1.0, ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5)
                build_map[y, x] = min(1.0, build_map[y, x] + 0.2 / dist)

        return threat_map, power_map, build_map

    def _compute_penalties(
        self, prev_state: Dict[str, Any], curr_state: Dict[str, Any]
    ) -> Tuple[int, int]:
        from rl.rewards.multi_objective import _detect_action_repetition_penalty, _detect_resource_bleeding_penalty

        new_buildings = max(
            0,
            len(curr_state.get("buildings", [])) - len(prev_state.get("buildings", []))
        )
        penalty_a = _detect_action_repetition_penalty(
            self._action_history, new_buildings=new_buildings
        )
        penalty_b = _detect_resource_bleeding_penalty(prev_state, curr_state)
        return int(penalty_a != 0.0), int(penalty_b != 0.0)

    def _execute_action(self, action_type: int, arg: int) -> None:
        if action_type == ACTION_WAIT:
            return
        if action_type == ACTION_MOVE:
            self._client.send_command(f"PLAYER_MOVE;{arg % 8}")
            return
        if action_type == ACTION_REPAIR:
            self._client.send_command(f"REPAIR_SLOT;{arg}")
            return
        if 0 <= action_type < len(ACTION_REGISTRY):
            block = ACTION_REGISTRY[action_type].block
            if block is not None:
                # Validate target tile is free before sending BUILD command
                if not self._is_build_target_free(arg):
                    _log.debug(
                        "Skipping BUILD %s at slot %d: target tile occupied",
                        block, arg
                    )
                    return
                self._client.send_command(f"PLAYER_BUILD;{block};{arg}")
                return
        raise ValueError(f"Invalid action_type: {action_type}")
    
    def _is_build_target_free(self, slot: int) -> bool:
        """Check if target tile for build slot is free (block == 'air' and no building)."""
        if self._prev_state is None:
            return True
        
        player = self._prev_state.get("player", {})
        if not player.get("alive", False):
            return False
        
        player_x = int(player.get("x", 0))
        player_y = int(player.get("y", 0))
        target_x = player_x + SLOT_DX[slot % NUM_SLOTS]
        target_y = player_y + SLOT_DY[slot % NUM_SLOTS]
        
        # Check if target tile has a block (not air)
        grid = self._prev_state.get("grid", [])
        for tile in grid:
            if int(tile.get("x", 0)) == target_x and int(tile.get("y", 0)) == target_y:
                block = tile.get("block", "air")
                if block != "air":
                    return False
        
        # Check if any building occupies the target position
        buildings = self._prev_state.get("buildings", [])
        for building in buildings:
            bx = int(building.get("x", 0))
            by = int(building.get("y", 0))
            if bx == target_x and by == target_y:
                return False
        
        # Tile not in grid and no building = free to build
        return True
