from enum import IntEnum
from typing import Dict, Any


class GameState(IntEnum):
    MINING = 0
    CRAFTING = 1
    ENERGY = 2
    DEFENSE = 3
    IDLE = 4


class StateMachine:
    def __init__(self):
        self.current_state = GameState.MINING
    
    def update(self, state: Dict[str, Any]) -> GameState:
        # Priority: Defense overrides all
        if state["threat"]["enemies_nearby"] > 0:
            self.current_state = GameState.DEFENSE
            return self.current_state
        
        # Check resource thresholds for normal progression
        copper = state["resources"]["copper"]
        graphite = state["resources"]["graphite"]
        power_ratio = state["power"]["current"] / state["power"]["capacity"]
        
        # State transitions
        if self.current_state == GameState.MINING:
            if copper > 200:
                self.current_state = GameState.CRAFTING
        
        elif self.current_state == GameState.CRAFTING:
            if graphite > 50:
                self.current_state = GameState.ENERGY
        
        elif self.current_state == GameState.ENERGY:
            if power_ratio > 0.8:
                self.current_state = GameState.IDLE
        
        elif self.current_state == GameState.IDLE:
            if copper < 100:
                self.current_state = GameState.MINING
        
        return self.current_state
