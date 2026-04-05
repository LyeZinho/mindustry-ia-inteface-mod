from enum import IntEnum
from typing import Dict, List, Any


class Action(IntEnum):
    PLACE_DRILL = 0
    PLACE_CONVEYOR = 1
    PLACE_GENERATOR = 2
    PLACE_TURRET = 3
    UPGRADE_BLOCK = 4
    DEMOLISH_BLOCK = 5
    WAIT = 6


class BehaviorTree:
    def __init__(self):
        self.action_names = {v.value: v.name for v in Action}
    
    def get_feasible_actions(self, state: Dict[str, Any]) -> List[int]:
        feasible = []
        
        # 1. Threat Assessment (highest priority)
        enemies_nearby = state["threat"]["enemies_nearby"]
        if enemies_nearby > 0:
            turrets = state["infrastructure"]["turrets_count"]
            if turrets < 3:
                feasible.append(Action.PLACE_TURRET)
            return feasible
        
        # 2. Energy Management
        power_ratio = state["power"]["current"] / state["power"]["capacity"]
        if power_ratio < 0.8:
            feasible.append(Action.PLACE_GENERATOR)
        
        # 3. Resource Production
        copper_ratio = state["resources"]["copper"] / 200.0
        lead_ratio = state["resources"]["lead"] / 100.0
        
        if copper_ratio < 0.5 or lead_ratio < 0.5:
            feasible.append(Action.PLACE_DRILL)
        
        # 4. Optimization
        if len(feasible) == 0:
            feasible.append(Action.PLACE_CONVEYOR)
            feasible.append(Action.WAIT)
        
        return feasible
