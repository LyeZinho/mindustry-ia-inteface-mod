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
    assert len(result["lookahead_scores"]) == 13


def test_placement_blocked_tile_is_negative():
    from rl.optimization.placement import compute_placement_scores

    state = {
        "player": {"x": 10, "y": 10, "alive": True},
        "core": {"x": 5, "y": 5},
        "enemies": [],
        "blockedTiles": [[10, 10]],
        "buildings": [],
        "oreGrid": [],
    }
    scores = compute_placement_scores(state)
    assert scores[4] < 0.0


def test_placement_ore_slot_higher_than_empty_slot():
    from rl.optimization.placement import compute_placement_scores
    from rl.env.spaces import SLOT_DX, SLOT_DY

    px, py = 10, 10
    ore_slot = 0
    ore_x = px + SLOT_DX[ore_slot]
    ore_y = py + SLOT_DY[ore_slot]

    state = {
        "player": {"x": px, "y": py, "alive": True},
        "core": {"x": 5, "y": 5},
        "enemies": [],
        "blockedTiles": [],
        "buildings": [],
        "oreGrid": [[ore_x, ore_y, 1]],
    }
    scores = compute_placement_scores(state)
    assert scores[ore_slot] > scores[1]


def test_placement_scores_range():
    from rl.optimization.placement import compute_placement_scores

    state = {
        "player": {"x": 10, "y": 10, "alive": True},
        "core": {"x": 5, "y": 5},
        "enemies": [],
        "blockedTiles": [[10, 10]],
        "buildings": [],
        "oreGrid": [[11, 11, 1]],
    }
    scores = compute_placement_scores(state)
    assert all(-1.0 <= s <= 1.0 for s in scores)
