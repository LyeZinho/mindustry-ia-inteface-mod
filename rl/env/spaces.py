"""
Observation and action space definitions for the Mindustry RL environment.

Observation: Dict with two tensors:
  "grid":     float32 (8, 31, 31)  — CNN input
    ch0: block type (encoded)
    ch1: block hp
    ch2: team (ally/enemy/neutral)
    ch3: ore type (from oreGrid)
    ch4: threat score heatmap
    ch5: power coverage heatmap
    ch6: build score heatmap
    ch7: rotation
    "features": float32 (122,) — MLP input
    0:      core_hp
    1-7:    resources (copper,lead,graphite,titanium,thorium,coal,sand) / 1000
    8-11:   power (produced,consumed,stored,capacity normalized)
    12:     wave / 100
    13-32:  enemies (5 × 4: hp,x,y,type_enc)
    33-41:  friendly units (3 × 3: hp,x,y)
    42:     waveTime / 3600
    43-44:  core x,y normalized
    45-48:  player (dx,dy,alive,hp)
    49-63:  nearby ores (5 × 3: distance,angle,block_id)
    64-78:  nearby enemies (5 × 3: distance,angle,hp)
    79-82:  extended resources (silicon,oil,water,metaglass) / 1000
    83-91:  ore_in_slot[0..8] / 7
    92-100: placement_scores[0..8] (tanh normalized, from opt worker)
    101:    power_deficit (from opt worker)
    102:    defense_gap (from opt worker)
    103:    wave_threat_index (from opt worker)
    104-108: build_order_priority[0..4] (from opt worker)
    109-121: lookahead_scores[0..12] (from opt worker)

Action: MultiDiscrete([13, 9])
  action[0]: action_type  ∈ {0..12} (WAIT, MOVE, BUILD_TURRET, BUILD_WALL, BUILD_POWER, BUILD_DRILL, REPAIR, BUILD_CONVEYOR, BUILD_GRAPHITE_PRESS, BUILD_SILICON_SMELTER, BUILD_COMBUSTION_GEN, BUILD_PNEUMATIC_DRILL, DELETE)
  action[1]: arg          ∈ {0..8}  (direction 0-7 for MOVE; relative slot 0-8 for build/repair/delete; ignored for WAIT)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple, Optional

import numpy as np
from gymnasium import spaces

# ------------------------------------------------------------------ #
# Constants                                                             #
# ------------------------------------------------------------------ #

GRID_SIZE = 31
GRID_CHANNELS = 8       # ch0:block ch1:hp ch2:team ch3:ore ch4:threat ch5:power ch6:build_score ch7:rotation
OBS_FEATURES_DIM = 122  # 92 existing + 9(placement) + 1(power_deficit) + 1(defense_gap) + 1(wave_threat) + 5(build_order) + 13(lookahead) = 122
EXTENDED_RESOURCES: list[str] = ["silicon", "oil", "water", "metaglass"]
NUM_SLOTS = 9           # 3x3 relative grid around unit (also covers 8 directions + 0 for WAIT)
MAX_ENEMIES = 5
MAX_FRIENDLY = 3
ENEMY_FEATURES = 4      # hp, x, y, type_enc
FRIENDLY_FEATURES = 3   # hp, x, y

ENEMY_TYPE_IDS: dict[str, int] = {
    "dagger": 1,
    "crawler": 2,
    "nova": 3,
    "pulsar": 4,
    "quasar": 5,
    "atrax": 6,
    "spiroct": 7,
    "arkyid": 8,
    "toxopid": 9,
    "flare": 10,
    "horizon": 11,
    "zenith": 12,
    "antumbra": 13,
    "eclipse": 14,
    "mono": 15,
    "poly": 16,
    "mega": 17,
    "quad": 18,
    "oct": 19,
}
_NUM_KNOWN_ENEMY_TYPES = 20

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
    ActionDef("DELETE",              None,                     lambda r: True),
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
ACTION_DELETE = _action_idx("DELETE")

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
        "grid": spaces.Box(0.0, 1.0, shape=(GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32),
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


def parse_observation(
    state: Dict[str, Any],
    opt_signals: Optional[Dict[str, Any]] = None,
    threat_map: Optional[np.ndarray] = None,
    power_map: Optional[np.ndarray] = None,
    build_map: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Convert raw Mimi Gateway state dict into gym-compatible observation."""
    ore_grid = state.get("oreGrid", [])
    core = state.get("core", {})
    core_x = int(core.get("x", GRID_SIZE // 2))
    core_y = int(core.get("y", GRID_SIZE // 2))
    grid_arr = _parse_grid(
        state.get("grid", []),
        ore_grid,
        threat_map,
        power_map,
        build_map,
        buildings=state.get("buildings", []),
        core_x=core_x,
        core_y=core_y,
    )
    features = _parse_features(state, opt_signals)
    return {
        "grid": grid_arr.astype(np.float32),
        "features": features.astype(np.float32),
    }


def _parse_grid(
    grid: List[Dict[str, Any]],
    ore_grid: Optional[List] = None,
    threat_map: Optional[np.ndarray] = None,
    power_map: Optional[np.ndarray] = None,
    build_map: Optional[np.ndarray] = None,
    buildings: Optional[List] = None,
    core_x: Optional[int] = None,
    core_y: Optional[int] = None,
) -> np.ndarray:
    arr = np.zeros((GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    if core_x is None:
        core_x = GRID_SIZE // 2
    if core_y is None:
        core_y = GRID_SIZE // 2
    origin_x = core_x - GRID_SIZE // 2
    origin_y = core_y - GRID_SIZE // 2

    for tile in grid:
        x = int(tile.get("x", 0))
        y = int(tile.get("y", 0))
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            continue
        arr[0, y, x] = _encode_block(tile.get("block", "air"))
        arr[1, y, x] = float(tile.get("hp", 0.0))
        arr[2, y, x] = _encode_team(tile.get("team", "neutral"))
        arr[7, y, x] = float(tile.get("rotation", 0)) / 3.0

    if buildings:
        for b in buildings:
            gx = int(b.get("x", 0)) - origin_x
            gy = int(b.get("y", 0)) - origin_y
            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                arr[0, gy, gx] = _encode_block(b.get("block", "air"))
                arr[1, gy, gx] = float(b.get("hp", 0.0))
                arr[2, gy, gx] = _encode_team(b.get("team", "neutral"))
                arr[7, gy, gx] = float(b.get("rotation", 0)) / 3.0

    if ore_grid:
        for entry in ore_grid:
            if len(entry) >= 3:
                ox = int(entry[0]) - origin_x
                oy = int(entry[1]) - origin_y
                ore_id = int(entry[2])
                if 0 <= ox < GRID_SIZE and 0 <= oy < GRID_SIZE:
                    arr[3, oy, ox] = ore_id / 7.0

    if threat_map is not None:
        arr[4] = np.clip(threat_map, 0.0, 1.0)

    if power_map is not None:
        arr[5] = np.clip(power_map, 0.0, 1.0)

    if build_map is not None:
        arr[6] = np.clip(build_map, 0.0, 1.0)

    return arr


def _parse_features(
    state: Dict[str, Any],
    opt_signals: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
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
    core_x = float(core.get("x", 0))
    core_y = float(core.get("y", 0))
    for i, e in enumerate(enemies):
        offset = base + i * ENEMY_FEATURES
        feat[offset] = float(e.get("hp", 0.0))
        feat[offset + 1] = (float(e.get("x", 0)) - core_x) / GRID_SIZE
        feat[offset + 2] = (float(e.get("y", 0)) - core_y) / GRID_SIZE
        feat[offset + 3] = ENEMY_TYPE_IDS.get(e.get("type", ""), 0) / _NUM_KNOWN_ENEMY_TYPES

    # Friendly units (top MAX_FRIENDLY, zero-padded)
    friendly = state.get("friendlyUnits", [])[:MAX_FRIENDLY]
    base = 33
    for i, u in enumerate(friendly):
        offset = base + i * FRIENDLY_FEATURES
        feat[offset] = float(u.get("hp", 0.0))
        feat[offset + 1] = (float(u.get("x", 0)) - core_x) / GRID_SIZE
        feat[offset + 2] = (float(u.get("y", 0)) - core_y) / GRID_SIZE

    feat[42] = float(state.get("waveTime", 0)) / 3600.0
    feat[43] = 0.5
    feat[44] = 0.5

    # Player unit position relative to core (feat[45..48])
    player = state.get("player", {})
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

    slots_ore = state.get("slotsOreType", [])
    for i in range(9):
        ore_id = int(slots_ore[i]) if i < len(slots_ore) else 0
        feat[83 + i] = ore_id / 7.0

    if opt_signals is not None:
        placement = opt_signals.get("placement_scores", np.zeros(9, dtype=np.float32))
        feat[92:101] = np.clip(placement, -1.0, 1.0)
        feat[101] = float(np.clip(opt_signals.get("power_deficit", 0.0), 0.0, 1.0))
        feat[102] = float(np.clip(opt_signals.get("defense_gap", 0.0), 0.0, 1.0))
        feat[103] = float(np.clip(opt_signals.get("wave_threat_index", 0.0), 0.0, 1.0))
        bop = opt_signals.get("build_order_priority", np.zeros(5, dtype=np.float32))
        feat[104:109] = np.clip(bop, 0.0, 1.0)
        lookahead = opt_signals.get("lookahead_scores", np.zeros(NUM_ACTION_TYPES, dtype=np.float32))
        feat[109:122] = np.clip(lookahead, 0.0, 1.0)

    return feat


# ------------------------------------------------------------------ #
# Action masking (for MaskablePPO)                                     #
# ------------------------------------------------------------------ #

def compute_action_mask(state: Dict[str, Any]) -> np.ndarray:
    """Return boolean mask of shape (NUM_ACTION_TYPES + NUM_SLOTS,).

    First NUM_ACTION_TYPES entries: valid action types.
    Next NUM_SLOTS entries: valid slots per action (masked if target occupied).
    
    Grid occupation check: For BUILD actions, mask slots where the target tile
    already has a block (block != "air").
    """
    mask = np.ones(NUM_ACTION_TYPES + NUM_SLOTS, dtype=np.bool_)

    player = state.get("player", {})
    if not player.get("alive", False):
        mask[1:NUM_ACTION_TYPES] = False
        return mask

    resources = state.get("resources", {})
    buildings = state.get("buildings", [])
    
    # Parse grid to check tile occupation
    grid = state.get("grid", [])
    grid_dict = {}  # Map (x, y) -> block_name
    for tile in grid:
        x = int(tile.get("x", 0))
        y = int(tile.get("y", 0))
        block = tile.get("block", "air")
        grid_dict[(x, y)] = block

    # Parse buildings to check occupation (buildings not in grid snapshot)
    buildings_set = set()  # Set of occupied (x, y) positions
    for building in buildings:
        bx = int(building.get("x", 0))
        by = int(building.get("y", 0))
        buildings_set.add((bx, by))

    blocked_set = set()
    for entry in state.get("blockedTiles", []):
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            blocked_set.add((int(entry[0]), int(entry[1])))

    # Get player position to compute slot coordinates
    player_x = int(player.get("x", 0)) if player else 0
    player_y = int(player.get("y", 0)) if player else 0

    for i, action in enumerate(ACTION_REGISTRY):
        if i in (ACTION_WAIT, ACTION_MOVE):
            continue
        if i == ACTION_REPAIR:
            if len(buildings) == 0:
                mask[i] = False
            # Don't mask slots for REPAIR — any friendly building can be repaired
            continue
        
        # Check resource availability for this action type
        if not action.mask_fn(resources):
            mask[i] = False
            continue
        
        # For BUILD actions, mask individual slots based on grid and buildings occupation
        if action.block is not None:
            for slot in range(NUM_SLOTS):
                target_x = player_x + SLOT_DX[slot]
                target_y = player_y + SLOT_DY[slot]
                block_at_target = grid_dict.get((target_x, target_y), "air")
                is_building_at_target = (target_x, target_y) in buildings_set
                
                # Mask slot if target tile is occupied (not air in grid OR has building OR in blockedTiles)
                if block_at_target != "air" or is_building_at_target or (target_x, target_y) in blocked_set:
                    slot_mask_idx = NUM_ACTION_TYPES + slot
                    mask[slot_mask_idx] = False

    return mask
