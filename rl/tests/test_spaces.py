"""Tests for observation/action space definitions."""
import numpy as np
import pytest
from gymnasium import spaces
from rl.env.spaces import (
    make_obs_space, make_action_space, parse_observation,
    BLOCK_TURRET, BLOCK_WALL, BLOCK_POWER, BLOCK_DRILL,
    NUM_ACTION_TYPES, NUM_SLOTS,
)

MINIMAL_STATE = {
    "tick": 1000, "time": 500, "wave": 3, "waveTime": 300,
    "resources": {"copper": 450, "lead": 120, "graphite": 75, "titanium": 50, "thorium": 0},
    "power": {"produced": 120.5, "consumed": 80.2, "stored": 500, "capacity": 1000},
    "core": {"hp": 0.95, "x": 15, "y": 15, "size": 3},
    "player": {"x": 16, "y": 17, "alive": True, "hp": 0.8},
    "enemies": [],
    "friendlyUnits": [],
    "buildings": [],
    "grid": [{"x": i % 31, "y": i // 31, "block": "air", "floor": "stone",
               "team": "neutral", "hp": 0.0, "rotation": 0} for i in range(961)],
}


def test_action_space_structure():
    act = make_action_space()
    assert isinstance(act, spaces.MultiDiscrete)
    assert list(act.nvec) == [NUM_ACTION_TYPES, NUM_SLOTS]


def test_obs_space_shape():
    obs = make_obs_space()
    assert isinstance(obs, spaces.Dict)
    assert obs["grid"].shape == (4, 31, 31)
    assert obs["features"].shape == (47,)


def test_parse_observation_returns_correct_shapes():
    obs = parse_observation(MINIMAL_STATE)
    assert obs["grid"].shape == (4, 31, 31)
    assert obs["features"].shape == (47,)


def test_parse_observation_grid_dtype():
    obs = parse_observation(MINIMAL_STATE)
    assert obs["grid"].dtype == np.float32


def test_parse_observation_features_dtype():
    obs = parse_observation(MINIMAL_STATE)
    assert obs["features"].dtype == np.float32


def test_parse_observation_grid_within_range():
    obs = parse_observation(MINIMAL_STATE)
    assert obs["grid"].min() >= 0.0
    assert obs["grid"].max() <= 1.0


def test_parse_observation_player_features():
    """feat[43..46] encode player position relative to core and alive/hp."""
    obs = parse_observation(MINIMAL_STATE)
    feat = obs["features"]
    # player at (16,17), core at (15,15) → dx=1, dy=2, both ÷15
    assert feat[43] == pytest.approx(1.0 / 15.0, abs=1e-5)
    assert feat[44] == pytest.approx(2.0 / 15.0, abs=1e-5)
    assert feat[45] == pytest.approx(1.0)   # alive
    assert feat[46] == pytest.approx(0.8)   # hp


def test_parse_observation_player_dead():
    """feat[45]=0 and feat[46]=0 when player not alive."""
    state = {**MINIMAL_STATE, "player": {"x": 0, "y": 0, "alive": False, "hp": 0.0}}
    obs = parse_observation(state)
    assert obs["features"][45] == pytest.approx(0.0)
    assert obs["features"][46] == pytest.approx(0.0)


def test_parse_observation_zero_pads_missing_enemies():
    obs = parse_observation(MINIMAL_STATE)
    enemy_block = obs["features"][11:31]
    assert np.all(enemy_block == 0.0)


def test_obs_space_contains_parsed_obs():
    obs_space = make_obs_space()
    obs = parse_observation(MINIMAL_STATE)
    assert obs_space["grid"].contains(obs["grid"])
    assert obs["features"].shape == obs_space["features"].shape
