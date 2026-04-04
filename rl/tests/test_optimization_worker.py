import time
import pytest
from rl.optimization.worker import OptimizationWorker

BASE_STATE = {
    "player": {"x": 10, "y": 10, "alive": True},
    "core": {"x": 10, "y": 10, "size": 3, "hp": 1.0},
    "resources": {"copper": 200, "lead": 50},
    "power": {"produced": 0.0, "consumed": 0.0, "capacity": 100},
    "buildings": [],
    "enemies": [],
    "wave": 1,
    "oreGrid": [],
    "blockedTiles": [],
}

def test_worker_starts_and_stops():
    w = OptimizationWorker()
    w.start()
    w.stop()
    assert not w.is_alive()

def test_worker_produces_results_after_update():
    w = OptimizationWorker(update_every=1)
    w.start()
    w.update(BASE_STATE)
    time.sleep(0.1)
    result = w.get_result()
    w.stop()
    assert "placement_scores" in result
    assert "lookahead_scores" in result
    assert "defense_gap" in result
    assert "power_deficit" in result

def test_worker_result_shapes():
    w = OptimizationWorker(update_every=1)
    w.start()
    w.update(BASE_STATE)
    time.sleep(0.1)
    result = w.get_result()
    w.stop()
    assert len(result["placement_scores"]) == 9
    assert len(result["lookahead_scores"]) == 12
