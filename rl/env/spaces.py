"""
Observation and action space definitions for the Mindustry RL environment.

Observation: Dict with two tensors:
  "grid":     float32 (4, 31, 31)  — CNN input
  "features": float32 (43,)        — MLP input

Action: MultiDiscrete([8, 31, 31])
  action[0]: action_type  ∈ {0..7}
  action[1]: x            ∈ {0..30}
  action[2]: y            ∈ {0..30}
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces

# ------------------------------------------------------------------ #
# Constants                                                             #
# ------------------------------------------------------------------ #

GRID_SIZE = 31
OBS_FEATURES_DIM = 43
NUM_ACTIONS = 8
MAX_ENEMIES = 5
MAX_FRIENDLY = 3
ENEMY_FEATURES = 4   # hp, x, y, type_enc
FRIENDLY_FEATURES = 3  # hp, x, y

ALLY_TEAMS = {"sharded", "player"}
ENEMY_TEAMS = {"crux"}

BLOCK_TURRET = "duo"
BLOCK_WALL = "wall"
BLOCK_POWER = "solar-panel"


# ------------------------------------------------------------------ #
# Space constructors                                                   #
# ------------------------------------------------------------------ #

def make_obs_space() -> spaces.Dict:
    return spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, shape=(4, GRID_SIZE, GRID_SIZE), dtype=np.float32),
        "features": spaces.Box(-np.inf, np.inf, shape=(OBS_FEATURES_DIM,), dtype=np.float32),
    })


def make_action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([NUM_ACTIONS, GRID_SIZE, GRID_SIZE])


# ------------------------------------------------------------------ #
# Observation parser                                                   #
# ------------------------------------------------------------------ #

def _encode_team(team: str) -> float:
    if team in ALLY_TEAMS:
        return 0.5
    if team in ENEMY_TEAMS:
        return 1.0
    return 0.0


def _encode_block(block: str) -> float:
    return (abs(hash(block)) % 50) / 50.0


def parse_observation(state: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert raw Mimi Gateway state dict into gym-compatible observation."""
    grid_arr = _parse_grid(state.get("grid", []))
    features = _parse_features(state)
    return {
        "grid": grid_arr.astype(np.float32),
        "features": features.astype(np.float32),
    }


def _parse_grid(grid: List[Dict[str, Any]]) -> np.ndarray:
    arr = np.zeros((4, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for tile in grid:
        x = int(tile.get("x", 0))
        y = int(tile.get("y", 0))
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            continue
        arr[0, y, x] = _encode_block(tile.get("block", "air"))
        arr[1, y, x] = float(tile.get("hp", 0.0))
        arr[2, y, x] = _encode_team(tile.get("team", "neutral"))
        arr[3, y, x] = float(tile.get("rotation", 0)) / 3.0
    return arr


def _parse_features(state: Dict[str, Any]) -> np.ndarray:
    feat = np.zeros(OBS_FEATURES_DIM, dtype=np.float32)

    core = state.get("core", {})
    feat[0] = float(core.get("hp", 0.0))

    res = state.get("resources", {})
    feat[1] = res.get("copper", 0.0) / 1000.0
    feat[2] = res.get("lead", 0.0) / 1000.0
    feat[3] = res.get("graphite", 0.0) / 1000.0
    feat[4] = res.get("titanium", 0.0) / 1000.0
    feat[5] = res.get("thorium", 0.0) / 1000.0

    power = state.get("power", {})
    max_power = max(float(power.get("capacity", 1.0)), 1.0)
    feat[6] = float(power.get("produced", 0.0)) / max_power
    feat[7] = float(power.get("consumed", 0.0)) / max_power
    feat[8] = float(power.get("stored", 0.0)) / max_power
    feat[9] = 1.0  # capacity normalized = 1

    feat[10] = float(state.get("wave", 0)) / 100.0

    # Enemies (top MAX_ENEMIES, zero-padded)
    enemies = state.get("enemies", [])[:MAX_ENEMIES]
    base = 11
    for i, e in enumerate(enemies):
        offset = base + i * ENEMY_FEATURES
        feat[offset] = float(e.get("hp", 0.0))
        feat[offset + 1] = float(e.get("x", 0)) / (GRID_SIZE - 1)
        feat[offset + 2] = float(e.get("y", 0)) / (GRID_SIZE - 1)
        feat[offset + 3] = (abs(hash(e.get("type", ""))) % 20) / 20.0

    # Friendly units (top MAX_FRIENDLY, zero-padded)
    friendly = state.get("friendlyUnits", [])[:MAX_FRIENDLY]
    base = 31
    for i, u in enumerate(friendly):
        offset = base + i * FRIENDLY_FEATURES
        feat[offset] = float(u.get("hp", 0.0))
        feat[offset + 1] = float(u.get("x", 0)) / (GRID_SIZE - 1)
        feat[offset + 2] = float(u.get("y", 0)) / (GRID_SIZE - 1)

    feat[40] = float(state.get("waveTime", 0)) / 3600.0
    feat[41] = float(core.get("x", 0)) / (GRID_SIZE - 1)
    feat[42] = float(core.get("y", 0)) / (GRID_SIZE - 1)

    return feat
