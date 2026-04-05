import time
from enum import IntEnum


class Action(IntEnum):
    PLACE_DRILL = 0
    PLACE_CONVEYOR = 1
    PLACE_GENERATOR = 2
    PLACE_TURRET = 3
    UPGRADE_BLOCK = 4
    DEMOLISH_BLOCK = 5
    WAIT = 6


class ActionExecutor:
    def __init__(self, safe_mode: bool = True):
        self.safe_mode = safe_mode
        self.action_names = {v.value: v.name for v in Action}
    
    def execute(self, action: int, x: float = 0, y: float = 0, rotation: int = 0) -> bool:
        try:
            if action == Action.PLACE_DRILL:
                return self._place_drill(x, y, rotation)
            elif action == Action.PLACE_CONVEYOR:
                return self._place_conveyor(x, y, rotation)
            elif action == Action.PLACE_GENERATOR:
                return self._place_generator(x, y, rotation)
            elif action == Action.PLACE_TURRET:
                return self._place_turret(x, y, rotation)
            elif action == Action.UPGRADE_BLOCK:
                return self._upgrade_block(x, y)
            elif action == Action.DEMOLISH_BLOCK:
                return self._demolish_block(x, y)
            elif action == Action.WAIT:
                return self._wait()
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as e:
            print(f"Error executing action {self.action_names.get(action, 'UNKNOWN')}: {e}")
            return False
    
    def _place_drill(self, x: float, y: float, rotation: int) -> bool:
        if self.safe_mode:
            time.sleep(0.1)
        return True
    
    def _place_conveyor(self, x: float, y: float, rotation: int) -> bool:
        if self.safe_mode:
            time.sleep(0.1)
        return True
    
    def _place_generator(self, x: float, y: float, rotation: int) -> bool:
        if self.safe_mode:
            time.sleep(0.1)
        return True
    
    def _place_turret(self, x: float, y: float, rotation: int) -> bool:
        if self.safe_mode:
            time.sleep(0.1)
        return True
    
    def _upgrade_block(self, x: float, y: float) -> bool:
        if self.safe_mode:
            time.sleep(0.1)
        return True
    
    def _demolish_block(self, x: float, y: float) -> bool:
        if self.safe_mode:
            time.sleep(0.1)
        return True
    
    def _wait(self) -> bool:
        return True
