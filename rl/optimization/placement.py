import math
import numpy as np
from rl.env.spaces import SLOT_DX, SLOT_DY, NUM_SLOTS

_ORE_WEIGHT = 2.0
_CORE_WEIGHT = 0.5
_THREAT_WEIGHT = 1.0

ORE_VALUES = {
    1: 1.0,
    2: 0.9,
    3: 0.8,
    4: 0.7,
    5: 1.2,
    6: 1.5,
    7: 0.3,
}

def compute_placement_scores(state: dict) -> np.ndarray:
    player = state.get("player", {})
    if not player.get("alive", False):
        return np.zeros(NUM_SLOTS, dtype=np.float32)
    
    player_x = player.get("x", 0)
    player_y = player.get("y", 0)
    
    core = state.get("core", {})
    core_x = core.get("x", 0)
    core_y = core.get("y", 0)
    
    enemies = state.get("enemies", [])
    
    blocked_tiles = set()
    for tile in state.get("blockedTiles", []):
        if isinstance(tile, list) and len(tile) >= 2:
            blocked_tiles.add((tile[0], tile[1]))
    
    for building in state.get("buildings", []):
        bx = building.get("x")
        by = building.get("y")
        if bx is not None and by is not None:
            blocked_tiles.add((bx, by))
    
    ore_grid_dict = {}
    for entry in state.get("oreGrid", []):
        if isinstance(entry, list) and len(entry) >= 3:
            x, y, ore_id = entry[0], entry[1], entry[2]
            ore_grid_dict[(x, y)] = ore_id
    
    scores = np.zeros(NUM_SLOTS, dtype=np.float32)
    
    for slot in range(NUM_SLOTS):
        slot_x = player_x + SLOT_DX[slot]
        slot_y = player_y + SLOT_DY[slot]
        
        if (slot_x, slot_y) in blocked_tiles:
            scores[slot] = -0.5
            continue
        
        ore_id = ore_grid_dict.get((slot_x, slot_y), 0)
        ore_value = ORE_VALUES.get(ore_id, 0.0)
        
        dist_to_core = math.sqrt((slot_x - core_x)**2 + (slot_y - core_y)**2)
        core_score = max(0.0, 1.0 - dist_to_core / 20.0)
        
        if enemies:
            min_enemy_dist = min(
                math.sqrt((slot_x - e.get("x", 0))**2 + (slot_y - e.get("y", 0))**2)
                for e in enemies
            )
            threat_score = min(1.0, min_enemy_dist / 15.0)
        else:
            threat_score = 1.0
        
        raw_score = (
            _ORE_WEIGHT * ore_value +
            _CORE_WEIGHT * core_score +
            _THREAT_WEIGHT * threat_score
        )
        
        scores[slot] = math.tanh(raw_score / 2.0)
    
    return scores
