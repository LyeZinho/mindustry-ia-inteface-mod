import numpy as np
from typing import Dict, Any


class GameStateReader:
    def __init__(self, map_width: int = 32, map_height: int = 32):
        self.map_width = map_width
        self.map_height = map_height
        self._reset_simulation()
    
    def _reset_simulation(self):
        self.sim_step = 0
        self.sim_resources = {"copper": 100, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10}
        self.sim_power = 500
        self.sim_health = 1.0
        self.sim_enemies = 0
        self.sim_wave = 1
        self.sim_infrastructure = {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5}
    
    def read_state(self) -> Dict[str, Any]:
        self._update_simulation()
        
        return {
            "resources": self.sim_resources.copy(),
            "power": {
                "current": self.sim_power,
                "capacity": 1000,
                "production": 300,
                "consumption": 200,
            },
            "threat": {
                "enemies_nearby": self.sim_enemies,
                "wave_number": self.sim_wave,
                "time_to_wave": max(0, 600 - (self.sim_step % 600)),
            },
            "infrastructure": self.sim_infrastructure.copy(),
            "status": {
                "core_health": self.sim_health,
                "recent_damage": 0,
                "game_time": self.sim_step,
            },
        }
    
    def _update_simulation(self):
        self.sim_step += 1
        
        if self.sim_step % 5 == 0:
            for resource in self.sim_resources:
                self.sim_resources[resource] += np.random.randint(1, 5)
        
        power_prod = 300 + np.random.randint(-20, 30)
        power_cons = 200 + np.random.randint(-15, 25)
        self.sim_power = min(1000, max(0, self.sim_power + power_prod - power_cons))
        
        if self.sim_step % 100 == 0 and np.random.rand() < 0.3:
            damage = np.random.uniform(0.01, 0.05)
            self.sim_health = max(0.0, self.sim_health - damage)
        
        if self.sim_step % 600 == 0:
            self.sim_wave += 1
            self.sim_enemies = np.random.randint(1, 5)
    
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
