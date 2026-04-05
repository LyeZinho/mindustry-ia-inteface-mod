from dataclasses import dataclass
from typing import Tuple, Dict, Optional


@dataclass
class GameState:
    structures: dict
    resources: dict
    map_width: int
    map_height: int


class ActionValidator:
    RESOURCE_COSTS = {
        "PLACE_DRILL": {"copper": 20},
        "PLACE_CONVEYOR": {"copper": 10},
        "PLACE_CONTAINER": {"copper": 50},
    }
    
    def validate(self, action, state: GameState) -> Tuple[bool, str]:
        if not self._is_in_bounds(action.position, state):
            return False, f"Position {action.position} out of bounds"
        
        if not self._is_cell_empty(action.position, state):
            return False, f"Position {action.position} already occupied"
        
        if not self._has_resources(action.type, state):
            return False, f"Insufficient resources for {action.type}"
        
        return True, "OK"
    
    def detect_conflict(self, action1, action2) -> bool:
        return action1.position == action2.position
    
    def _is_in_bounds(self, position: Tuple[int, int], state: GameState) -> bool:
        x, y = position
        return 0 <= x < state.map_width and 0 <= y < state.map_height
    
    def _is_cell_empty(self, position: Tuple[int, int], state: GameState) -> bool:
        return position not in state.structures
    
    def _has_resources(self, action_type: str, state: GameState) -> bool:
        if action_type not in self.RESOURCE_COSTS:
            return True
        
        required = self.RESOURCE_COSTS[action_type]
        for resource, amount in required.items():
            if state.resources.get(resource, 0) < amount:
                return False
        
        return True
