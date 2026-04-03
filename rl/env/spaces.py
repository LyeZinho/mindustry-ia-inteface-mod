"""
Observation and action space definitions for the Mindustry RL environment.

Observation: Dict with two tensors:
  "grid":     float32 (4, 31, 31)  — CNN input
  "features": float32 (79,)        — MLP input (resources: copper,lead,graphite,titanium,thorium,coal,sand; power; wave; enemies; friendly; player)

Action: MultiDiscrete([7, 9])
  action[0]: action_type  ∈ {0..6}  (WAIT, MOVE, BUILD_TURRET, BUILD_WALL, BUILD_POWER, BUILD_DRILL, REPAIR)
  action[1]: arg          ∈ {0..8}  (direction 0-7 for MOVE; relative slot 0-8 for build/repair; ignored for WAIT)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple, Optional

import numpy as np
from gymnasium import spaces

# ------------------------------------------------------------------ #
# Constants                                                             #
# ------------------------------------------------------------------ #

GRID_SIZE = 31
OBS_FEATURES_DIM = 83   # 1(core_hp)+7(res)+4(power)+1(wave)+20(enemies)+9(friendly)+7(player/core)+15(nearby_ores)+15(nearby_enemies)+4(ext_resources)=83
EXTENDED_RESOURCES: list[str] = ["silicon", "oil", "water", "metaglass"]
NUM_SLOTS = 9           # 3x3 relative grid around unit (also covers 8 directions + 0 for WAIT)
MAX_ENEMIES = 5
MAX_FRIENDLY = 3
ENEMY_FEATURES = 4      # hp, x, y, type_enc
FRIENDLY_FEATURES = 3   # hp, x, y

ALLY_TEAMS = {"sharded", "player"}
ENEMY_TEAMS = {"crux"}

class ActionDef(NamedTuple):
    name: str
    block: str | None
    mask_fn: Callable[[dict], bool]

ACTION_REGISTRY: list[ActionDef] = [
    ActionDef("WAIT",                None,                     lambda r: True),
    ActionDef("MOVE",                None,                     lambda r: True),
    ActionDef("BUILD_TURRET",        "duo",                    lambda r: r.get("copper", 0) >= 35),
    ActionDef("BUILD_WALL",          "copper-wall",            lambda r: r.get("copper", 0) >= 6),
    ActionDef("BUILD_POWER",         "solar-panel",            lambda r: r.get("copper", 0) >= 40 and r.get("lead", 0) >= 35),
    ActionDef("BUILD_DRILL",         "mechanical-drill",       lambda r: r.get("copper", 0) >= 12),
    ActionDef("REPAIR",              None,                     lambda r: True),
    ActionDef("BUILD_CONVEYOR",      "conveyor",               lambda r: r.get("copper", 0) >= 1),
    ActionDef("BUILD_GRAPHITE_PRESS","graphite-press",         lambda r: r.get("copper", 0) >= 75),
    ActionDef("BUILD_SILICON_SMELTER","silicon-smelter",       lambda r: r.get("copper", 0) >= 30 and r.get("lead", 0) >= 30),
    ActionDef("BUILD_COMBUSTION_GEN","combustion-generator",   lambda r: r.get("copper", 0) >= 25 and r.get("lead", 0) >= 15),
    ActionDef("BUILD_PNEUMATIC_DRILL","pneumatic-drill",       lambda r: r.get("copper", 0) >= 12 and r.get("graphite", 0) >= 10),
]

NUM_ACTION_TYPES: int = len(ACTION_REGISTRY)
ACTION_NAMES: list[str] = [a.name for a in ACTION_REGISTRY]

def _action_idx(name: str) -> int:
    for i, a in enumerate(ACTION_REGISTRY):
        if a.name == name:
            return i
    raise ValueError(f"Unknown action: {name}")

ACTION_WAIT   = _action_idx("WAIT")
ACTION_MOVE   = _action_idx("MOVE")
ACTION_REPAIR = _action_idx("REPAIR")

BLOCK_TURRET = ACTION_REGISTRY[_action_idx("BUILD_TURRET")].block
BLOCK_WALL   = ACTION_REGISTRY[_action_idx("BUILD_WALL")].block
BLOCK_POWER  = ACTION_REGISTRY[_action_idx("BUILD_POWER")].block
BLOCK_DRILL  = ACTION_REGISTRY[_action_idx("BUILD_DRILL")].block

# Deterministic block encoding — no hash collisions
BLOCK_IDS: dict[str, int] = {
    "air": 0,
    "copper-wall": 1,
    "copper-wall-large": 2,
    "duo": 3,
    "scatter": 4,
    "hail": 5,
    "lancer": 6,
    "wave": 7,
    "swarmer": 8,
    "mechanical-drill": 9,
    "pneumatic-drill": 10,
    "conveyor": 11,
    "titanium-conveyor": 12,
    "router": 13,
    "junction": 14,
    "overflow-gate": 15,
    "sorter": 16,
    "solar-panel": 17,
    "solar-panel-large": 18,
    "battery": 19,
    "battery-large": 20,
    "power-node": 21,
    "power-node-large": 22,
    "thermal-generator": 23,
    "core-shard": 24,
    "vault": 25,
    "container": 26,
    "mender": 27,
    "mend-projector": 28,
    "overdrive-projector": 29,
    "force-projector": 30,
    "graphite-press": 31,
    "silicon-smelter": 32,
    "combustion-generator": 33,
}
_NUM_KNOWN_BLOCKS = len(BLOCK_IDS) + 1  # +1 for "unknown"

# Slot (0-8) → (dx, dy) in tiles, relative to player unit
# Row-major 3×3: top-left=0, top-center=1, ... bottom-right=8
# y+ = north on Mindustry map
SLOT_DX = [-1,  0,  1, -1,  0,  1, -1,  0,  1]
SLOT_DY = [ 1,  1,  1,  0,  0,  0, -1, -1, -1]

# Direction (0-7) for MOVE: N/NE/E/SE/S/SW/W/NW
MOVE_DIR_DX = [0, 1, 1,  1,  0, -1, -1, -1]
MOVE_DIR_DY = [1, 1, 0, -1, -1, -1,  0,  1]


# ------------------------------------------------------------------ #
# Space constructors                                                   #
# ------------------------------------------------------------------ #

def make_obs_space() -> spaces.Dict:
    return spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, shape=(4, GRID_SIZE, GRID_SIZE), dtype=np.float32),
        "features": spaces.Box(-np.inf, np.inf, shape=(OBS_FEATURES_DIM,), dtype=np.float32),
    })


def make_action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([NUM_ACTION_TYPES, NUM_SLOTS])


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
    idx = BLOCK_IDS.get(block, _NUM_KNOWN_BLOCKS - 1)
    return idx / _NUM_KNOWN_BLOCKS


def parse_observation(state: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert raw Mimi Gateway state dict into gym-compatible observation."""
    grid_arr = _parse_grid(state.get("grid", []))
    features = _parse_features(state)
    return {
        "grid": grid_arr.astype(np.float32),
        "features": features.astype(np.float32),
    }


def _parse_grid(grid: List[Dict[str, Any]]) -> np.ndarray:
    """
    Parse sparse grid format (31×31 matrix now empty to reduce JSON size from 50KB to 500B).
    Sparse features (nearby ores/enemies) are extracted separately in _parse_features.
    Backward compatible: returns (4, 31, 31) zeros if grid is empty array.
    """
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
    feat[6] = res.get("coal", 0.0) / 1000.0
    feat[7] = res.get("sand", 0.0) / 1000.0

    for i, rname in enumerate(EXTENDED_RESOURCES):
        feat[79 + i] = res.get(rname, 0.0) / 1000.0

    power = state.get("power", {})
    max_power = max(float(power.get("capacity", 1.0)), 1.0)
    feat[8] = float(power.get("produced", 0.0)) / max_power
    feat[9] = float(power.get("consumed", 0.0)) / max_power
    feat[10] = float(power.get("stored", 0.0)) / max_power
    feat[11] = 1.0  # capacity normalized = 1

    feat[12] = float(state.get("wave", 0)) / 100.0

    # Enemies (top MAX_ENEMIES, zero-padded)
    enemies = state.get("enemies", [])[:MAX_ENEMIES]
    base = 13
    for i, e in enumerate(enemies):
        offset = base + i * ENEMY_FEATURES
        feat[offset] = float(e.get("hp", 0.0))
        feat[offset + 1] = float(e.get("x", 0)) / (GRID_SIZE - 1)
        feat[offset + 2] = float(e.get("y", 0)) / (GRID_SIZE - 1)
        feat[offset + 3] = (abs(hash(e.get("type", ""))) % 20) / 20.0

    # Friendly units (top MAX_FRIENDLY, zero-padded)
    friendly = state.get("friendlyUnits", [])[:MAX_FRIENDLY]
    base = 33
    for i, u in enumerate(friendly):
        offset = base + i * FRIENDLY_FEATURES
        feat[offset] = float(u.get("hp", 0.0))
        feat[offset + 1] = float(u.get("x", 0)) / (GRID_SIZE - 1)
        feat[offset + 2] = float(u.get("y", 0)) / (GRID_SIZE - 1)

    feat[42] = float(state.get("waveTime", 0)) / 3600.0
    feat[43] = float(core.get("x", 0)) / (GRID_SIZE - 1)
    feat[44] = float(core.get("y", 0)) / (GRID_SIZE - 1)

    # Player unit position relative to core (feat[45..48])
    player = state.get("player", {})
    core_x = float(core.get("x", 0))
    core_y = float(core.get("y", 0))
    player_alive = bool(player.get("alive", False))
    if player_alive:
        feat[45] = (float(player.get("x", core_x)) - core_x) / 15.0
        feat[46] = (float(player.get("y", core_y)) - core_y) / 15.0
        feat[47] = 1.0
        feat[48] = float(player.get("hp", 0.0))
    # else stays 0.0

    # PHASE 2: Sparse ore/enemy features (30 dims, indices 49-78)
    # Top 5 nearest ores: 5 × (distance + angle + block_id) = 15 dims (indices 49-63)
    # Top 5 nearest enemies: 5 × (distance + angle + hp) = 15 dims (indices 64-78)

    nearby_ores = state.get("nearbyOres", [])
    for i in range(min(5, len(nearby_ores))):
        ore = nearby_ores[i]
        offset = 49 + i * 3
        feat[offset] = float(ore.get("distance", 0.0)) / 50.0
        feat[offset + 1] = float(ore.get("angle", 0.0)) / 180.0
        feat[offset + 2] = float(ore.get("block_id", 0)) / 32.0

    nearby_enemies = state.get("nearbyEnemies", [])
    for i in range(min(5, len(nearby_enemies))):
        enemy = nearby_enemies[i]
        offset = 64 + i * 3
        feat[offset] = float(enemy.get("distance", 0.0)) / 50.0
        feat[offset + 1] = float(enemy.get("angle", 0.0)) / 180.0
        feat[offset + 2] = float(enemy.get("hp", 0.0))

    return feat


# ------------------------------------------------------------------ #
# Action masking (for MaskablePPO)                                     #
# ------------------------------------------------------------------ #

def compute_action_mask(state: Dict[str, Any]) -> np.ndarray:
    """Return boolean mask of shape (NUM_ACTION_TYPES + NUM_SLOTS,).

    First NUM_ACTION_TYPES entries: valid action types.
    Next NUM_SLOTS entries: valid slots (always True).
    """
    mask = np.ones(NUM_ACTION_TYPES + NUM_SLOTS, dtype=np.bool_)

    player = state.get("player", {})
    if not player.get("alive", False):
        mask[1:NUM_ACTION_TYPES] = False
        return mask

    resources = state.get("resources", {})
    buildings = state.get("buildings", [])

    for i, action in enumerate(ACTION_REGISTRY):
        if i in (ACTION_WAIT, ACTION_MOVE):
            continue
        if i == ACTION_REPAIR:
            if len(buildings) == 0:
                mask[i] = False
            continue
        if not action.mask_fn(resources):
            mask[i] = False

    return mask
