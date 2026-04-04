import numpy as np
import pytest
from rl.optimization.placement import compute_placement_scores

SIMPLE_STATE = {
    "player": {"x": 10, "y": 10, "alive": True},
    "core": {"x": 10, "y": 10, "size": 3},
    "enemies": [],
    "buildings": [],
    "oreGrid": [[10, 11, 1]],
    "blockedTiles": [],
}

def test_placement_scores_shape():
    scores = compute_placement_scores(SIMPLE_STATE)
    assert scores.shape == (9,)

def test_placement_scores_range():
    scores = compute_placement_scores(SIMPLE_STATE)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)

def test_ore_slot_scores_higher():
    scores = compute_placement_scores(SIMPLE_STATE)
    assert scores[1] > scores[3]

def test_blocked_slot_scores_zero():
    state = {**SIMPLE_STATE, "blockedTiles": [[9, 11]]}
    scores = compute_placement_scores(state)
    assert scores[0] == pytest.approx(-0.5)
