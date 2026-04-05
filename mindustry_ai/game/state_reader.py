import numpy as np
from typing import Dict, Any


class GameStateReader:
    def __init__(self, map_width: int = 32, map_height: int = 32):
        self.map_width = map_width
        self.map_height = map_height
    
    def read_state(self) -> Dict[str, Any]:
        return {
            "resources": {
                "copper": 100,
                "lead": 50,
                "coal": 30,
                "graphite": 20,
                "titanium": 10,
            },
            "power": {
                "current": 500,
                "capacity": 1000,
                "production": 300,
                "consumption": 200,
            },
            "threat": {
                "enemies_nearby": 0,
                "wave_number": 1,
                "time_to_wave": 600,
            },
            "infrastructure": {
                "drills_count": 2,
                "turrets_count": 1,
                "conveyors_count": 5,
            },
            "status": {
                "core_health": 1.0,
                "recent_damage": 0,
                "game_time": 0,
            },
        }
    
    def to_flat_vector(self, state: Dict[str, Any]) -> np.ndarray:
        return np.array([
            state["resources"]["copper"],
            state["resources"]["lead"],
            state["resources"]["coal"],
            state["resources"]["graphite"],
            state["resources"]["titanium"],
            state["power"]["current"],
            state["power"]["capacity"],
            state["power"]["production"],
            state["power"]["consumption"],
            state["threat"]["enemies_nearby"],
            state["threat"]["wave_number"],
            state["threat"]["time_to_wave"],
            state["infrastructure"]["drills_count"],
            state["infrastructure"]["turrets_count"],
            state["status"]["core_health"],
        ], dtype=np.float32)
    
    def to_spatial_map(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return {
            "blocks": np.zeros((self.map_width, self.map_height), dtype=np.int32),
            "resources": np.zeros((self.map_width, self.map_height), dtype=np.float32),
            "enemies": np.zeros((self.map_width, self.map_height), dtype=np.float32),
        }
