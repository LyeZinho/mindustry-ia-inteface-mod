import numpy as np
import pytest
from rl.optimization.lookahead import compute_lookahead_scores

BASE_STATE = {
    "player": {"x": 10, "y": 10, "alive": True},
    "core": {"x": 10, "y": 10, "size": 3, "hp": 1.0},
    "resources": {"copper": 200, "lead": 50, "graphite": 0},
    "power": {"produced": 0.0, "consumed": 0.0, "capacity": 100},
    "buildings": [],
    "enemies": [],
    "wave": 1,
    "oreGrid": [[10, 11, 1]],
    "blockedTiles": [],
}

def test_lookahead_scores_shape():
    scores = compute_lookahead_scores(BASE_STATE)
    assert scores.shape == (12,)

def test_lookahead_scores_non_negative():
    scores = compute_lookahead_scores(BASE_STATE)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)

def test_build_drill_scores_positive():
    scores = compute_lookahead_scores(BASE_STATE)
    BUILD_DRILL_IDX = 5
    assert scores[BUILD_DRILL_IDX] > 0.0

def test_wait_scores_lower_than_build():
    scores = compute_lookahead_scores(BASE_STATE)
    WAIT_IDX = 0
    BUILD_DRILL_IDX = 5
    assert scores[WAIT_IDX] < scores[BUILD_DRILL_IDX]
