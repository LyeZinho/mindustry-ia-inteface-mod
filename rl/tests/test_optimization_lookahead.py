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
    assert scores.shape == (13,)

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

def test_existing_drills_increase_lookahead_score():
    """An agent with placed drills should see higher lookahead score for build actions
    (more resources available in future steps) than one without drills."""
    from rl.optimization.lookahead import compute_lookahead_scores
    from rl.env.spaces import ACTION_BUILD_DRILL

    base_state = {
        "player": {"x": 10, "y": 10, "alive": True},
        "resources": {"copper": 50},
        "buildings": [],
        "power": {"produced": 0.0, "consumed": 0.0},
        "oreGrid": [],
    }
    state_with_drills = {
        **base_state,
        "buildings": [
            {"block": "mechanical-drill", "team": "sharded", "x": 11, "y": 11},
            {"block": "mechanical-drill", "team": "sharded", "x": 12, "y": 11},
        ],
    }

    scores_no_drills = compute_lookahead_scores(base_state)
    scores_with_drills = compute_lookahead_scores(state_with_drills)

    assert scores_with_drills[ACTION_BUILD_DRILL] > scores_no_drills[ACTION_BUILD_DRILL]
