import math
import copy
import numpy as np
from rl.env.spaces import ACTION_REGISTRY, NUM_ACTION_TYPES, SLOT_DX, SLOT_DY, NUM_SLOTS

BUILD_COSTS = {
    "duo": {"copper": 35},
    "copper-wall": {"copper": 6},
    "solar-panel": {"copper": 40, "lead": 35},
    "mechanical-drill": {"copper": 12},
    "conveyor": {"copper": 1},
    "graphite-press": {"copper": 75},
    "silicon-smelter": {"copper": 30, "lead": 30},
    "combustion-generator": {"copper": 25, "lead": 15},
    "pneumatic-drill": {"copper": 12, "graphite": 10},
}

BUILDING_INCOME = {
    "mechanical-drill": {"copper": 2.0},
    "pneumatic-drill": {"copper": 1.5, "graphite": 0.5},
    "graphite-press": {"graphite": 1.0},
    "silicon-smelter": {"silicon": 0.5},
    "combustion-generator": {"power_delta": 60.0},
}

def compute_lookahead_scores(state: dict) -> np.ndarray:
    scores = np.zeros(NUM_ACTION_TYPES, dtype=np.float32)
    
    player = state.get("player", {})
    if not player.get("alive", False):
        return scores
    
    for action_idx, action_def in enumerate(ACTION_REGISTRY):
        sim_state = copy.deepcopy(state)
        total_score = 0.0
        
        for step in range(3):
            if action_def.block is None:
                pass
            else:
                block_name = action_def.block
                cost = BUILD_COSTS.get(block_name, {})
                
                resources = sim_state.get("resources", {})
                can_afford = all(
                    resources.get(res_name, 0) >= res_cost
                    for res_name, res_cost in cost.items()
                )
                
                if can_afford:
                    total_score += 1.0
                    
                    for res_name, res_cost in cost.items():
                        resources[res_name] = resources.get(res_name, 0) - res_cost
                    
                    income = BUILDING_INCOME.get(block_name, {})
                    for income_res, income_val in income.items():
                        if income_res == "power_delta":
                            power = sim_state.get("power", {})
                            power["produced"] = power.get("produced", 0.0) + income_val
                            total_score += 0.1 * income_val / 60.0
                        else:
                            resources[income_res] = resources.get(income_res, 0) + income_val
                            total_score += 0.1 * income_val
        
        power = sim_state.get("power", {})
        if power.get("produced", 0.0) > power.get("consumed", 0.0):
            total_score += 0.2
        
        scores[action_idx] = math.log(1 + total_score) / math.log(1 + 10.0)
    
    return scores
