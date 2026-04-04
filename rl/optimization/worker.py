import threading
import time
import copy
import numpy as np
from rl.optimization.placement import compute_placement_scores
from rl.optimization.defense import compute_defense_gap
from rl.optimization.power import compute_power_deficit
from rl.optimization.lookahead import compute_lookahead_scores

def _compute_build_order_priority(state: dict) -> np.ndarray:
    resources = state.get("resources", {})
    copper = resources.get("copper", 0)
    lead = resources.get("lead", 0)
    graphite = resources.get("graphite", 0)
    
    priorities = np.zeros(5, dtype=np.float32)
    
    priorities[0] = 1.0 if copper >= 12 else 0.0
    priorities[1] = 1.0 if copper >= 35 else 0.0
    priorities[2] = 1.0 if copper >= 40 and lead >= 35 else 0.0
    priorities[3] = 1.0 if copper >= 75 else 0.0
    priorities[4] = 1.0 if copper >= 12 and graphite >= 10 else 0.0
    
    return priorities

def _compute_wave_threat(state: dict) -> float:
    wave = state.get("wave", 0)
    enemies = state.get("enemies", [])
    
    wave_factor = min(1.0, wave / 20.0)
    
    enemy_factor = min(1.0, len(enemies) / 10.0)
    
    total_threat = (wave_factor + enemy_factor) / 2.0
    
    return total_threat

class OptimizationWorker(threading.Thread):
    def __init__(self, update_every=5):
        super().__init__(daemon=True)
        self.update_every = update_every
        self._lock = threading.Lock()
        self._pending_state = None
        self._result = {
            "placement_scores": np.zeros(9, dtype=np.float32),
            "lookahead_scores": np.zeros(12, dtype=np.float32),
            "defense_gap": 1.0,
            "power_deficit": 0.0,
            "build_order_priority": np.zeros(5, dtype=np.float32),
            "wave_threat_index": 0.0,
        }
        self._stop_event = threading.Event()
        self._step_counter = 0
    
    def update(self, state: dict):
        self._step_counter += 1
        if self._step_counter % self.update_every == 0:
            with self._lock:
                self._pending_state = copy.deepcopy(state)
    
    def get_result(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._result)
    
    def run(self):
        while not self._stop_event.is_set():
            with self._lock:
                pending = self._pending_state
                self._pending_state = None
            
            if pending:
                placement_scores = compute_placement_scores(pending)
                lookahead_scores = compute_lookahead_scores(pending)
                defense_gap = compute_defense_gap(pending)
                power_deficit = compute_power_deficit(pending)
                build_order_priority = _compute_build_order_priority(pending)
                wave_threat = _compute_wave_threat(pending)
                
                with self._lock:
                    self._result = {
                        "placement_scores": placement_scores,
                        "lookahead_scores": lookahead_scores,
                        "defense_gap": defense_gap,
                        "power_deficit": power_deficit,
                        "build_order_priority": build_order_priority,
                        "wave_threat_index": wave_threat,
                    }
            
            time.sleep(0.005)
    
    def stop(self):
        self._stop_event.set()
        self.join(timeout=1.0)
