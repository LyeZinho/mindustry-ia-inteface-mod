"""Tests for MindustryEnv — uses a mock MimiClient (no live game)."""
import numpy as np
import pytest
from unittest.mock import MagicMock
from rl.env.mindustry_env import MindustryEnv

MOCK_STATE = {
    "tick": 1000, "time": 500, "wave": 1, "waveTime": 300,
    "resources": {"copper": 100, "lead": 50, "graphite": 0, "titanium": 0, "thorium": 0},
    "power": {"produced": 10.0, "consumed": 5.0, "stored": 100, "capacity": 1000},
    "core": {"hp": 1.0, "x": 15, "y": 15, "size": 3},
    "player": {"x": 15, "y": 15},
    "enemies": [],
    "friendlyUnits": [],
    "buildings": [],
    "grid": [{"x": i % 31, "y": i // 31, "block": "air", "floor": "stone",
               "team": "neutral", "hp": 0.0, "rotation": 0} for i in range(961)],
}


def make_mock_client(states=None):
    """Mock MimiClient returning predefined states."""
    client = MagicMock()
    if states is None:
        states = [MOCK_STATE, MOCK_STATE]
    client.receive_state.side_effect = states
    return client


def test_reset_returns_valid_obs():
    """reset() returns obs dict with correct shapes."""
    env = MindustryEnv(client=make_mock_client())
    obs, info = env.reset()
    assert "grid" in obs and "features" in obs
    assert obs["grid"].shape == (4, 31, 31)
    assert obs["features"].shape == (43,)
    assert isinstance(info, dict)


def test_step_returns_five_tuple():
    """step() returns (obs, reward, terminated, truncated, info)."""
    env = MindustryEnv(client=make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE]))
    env.reset()
    action = {"action_type": 0, "x": np.array([15], dtype=np.int32), "y": np.array([15], dtype=np.int32)}
    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs["grid"].shape == (4, 31, 31)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_step_build_turret_sends_build_command():
    """action_type=1 sends BUILD;duo;x;y;0."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = {"action_type": 1, "x": np.array([10], dtype=np.int32), "y": np.array([12], dtype=np.int32)}
    env.step(action)
    client.build.assert_called_with("duo", 10, 12, rotation=0)


def test_step_wait_sends_msg():
    """action_type=0 (WAIT) sends message command."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = {"action_type": 0, "x": np.array([0], dtype=np.int32), "y": np.array([0], dtype=np.int32)}
    env.step(action)
    client.message.assert_any_call("WAIT")


def test_episode_terminates_on_core_destroyed():
    """terminated=True when core hp <= 0."""
    dead_state = {**MOCK_STATE, "core": {"hp": 0.0, "x": 15, "y": 15, "size": 3}}
    client = make_mock_client(states=[MOCK_STATE, dead_state])
    env = MindustryEnv(client=client)
    env.reset()
    action = {"action_type": 0, "x": np.array([0], dtype=np.int32), "y": np.array([0], dtype=np.int32)}
    _, _, terminated, _, _ = env.step(action)
    assert terminated is True


def test_episode_truncates_on_max_steps():
    """truncated=True when step count >= max_steps."""
    from itertools import repeat
    states = list(repeat(MOCK_STATE, 12))
    client = MagicMock()
    client.receive_state.side_effect = states
    env = MindustryEnv(client=client, max_steps=5)
    env.reset()
    action = {"action_type": 0, "x": np.array([0], dtype=np.int32), "y": np.array([0], dtype=np.int32)}
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(action)
    assert not truncated
    _, _, terminated, truncated, _ = env.step(action)
    assert truncated is True
