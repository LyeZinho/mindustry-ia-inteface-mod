"""Tests for MindustryEnv — uses a mock MimiClient (no live game)."""
import numpy as np
import pytest
from unittest.mock import MagicMock
from rl.env.mindustry_env import MindustryEnv

MOCK_STATE = {
    "tick": 1000, "time": 500, "wave": 1, "waveTime": 300,
    "resources": {"copper": 100, "lead": 50, "graphite": 20, "titanium": 0, "thorium": 0},
    "power": {"produced": 10.0, "consumed": 5.0, "stored": 100, "capacity": 1000},
    "core": {"hp": 1.0, "x": 15, "y": 15, "size": 3},
    "player": {"x": 15, "y": 15, "alive": True, "hp": 1.0},
    "enemies": [],
    "friendlyUnits": [],
    "buildings": [],
    "actionFailed": False,
    "grid": [],  # Empty grid (sparse format)
    "nearbyOres": [],
    "nearbyEnemies": [],
}


def make_mock_client(states=None):
    client = MagicMock()
    if states is None:
        states = [MOCK_STATE, MOCK_STATE]
    client.receive_state.side_effect = states
    return client


def test_reset_returns_valid_obs():
    env = MindustryEnv(client=make_mock_client())
    obs, info = env.reset()
    assert "grid" in obs and "features" in obs
    assert obs["grid"].shape == (8, 31, 31)
    assert obs["features"].shape == (121,)
    assert isinstance(info, dict)


def test_step_returns_five_tuple():
    env = MindustryEnv(client=make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE]))
    env.reset()
    action = np.array([0, 0], dtype=np.int64)  # WAIT, arg=0
    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs["grid"].shape == (8, 31, 31)
    assert obs["features"].shape == (121,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_step_wait_does_not_send_movement():
    """action_type=0 (WAIT) should not call send_command with PLAYER_MOVE."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([0, 0], dtype=np.int64)
    env.step(action)
    for call in client.send_command.call_args_list:
        assert not call[0][0].startswith("PLAYER_MOVE")


def test_step_move_sends_player_move_command():
    """action_type=1 sends PLAYER_MOVE;direction."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([1, 3], dtype=np.int64)  # MOVE direction=3 (SE)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_MOVE;3")


def test_step_build_turret_sends_player_build():
    """action_type=2 sends PLAYER_BUILD;duo;slot."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([2, 4], dtype=np.int64)  # BUILD_TURRET slot=4
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;duo;4")


def test_step_build_wall_sends_player_build():
    """action_type=3 sends PLAYER_BUILD;copper-wall;slot."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([3, 0], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;copper-wall;0")


def test_step_build_power_sends_player_build():
    """action_type=4 sends PLAYER_BUILD;solar-panel;slot."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([4, 1], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;solar-panel;1")


def test_step_build_drill_sends_player_build():
    """action_type=5 sends PLAYER_BUILD;mechanical-drill;slot."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([5, 2], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;mechanical-drill;2")


def test_step_repair_sends_player_build_repair():
    """action_type=6 sends REPAIR_SLOT;slot."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([6, 7], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("REPAIR_SLOT;7")


def test_episode_terminates_on_core_destroyed():
    dead_state = {**MOCK_STATE, "core": {"hp": 0.0, "x": 15, "y": 15, "size": 3},
                  "player": {"x": 15, "y": 15, "alive": True, "hp": 1.0}}
    client = make_mock_client(states=[MOCK_STATE, dead_state])
    env = MindustryEnv(client=client)
    env.reset()
    _, _, terminated, _, _ = env.step(np.array([0, 0], dtype=np.int64))
    assert terminated is True


def test_episode_terminates_on_player_dead():
    dead_player_state = {**MOCK_STATE, "player": {"x": 0, "y": 0, "alive": False, "hp": 0.0}}
    client = make_mock_client(states=[MOCK_STATE, dead_player_state])
    env = MindustryEnv(client=client)
    env.reset()
    _, _, terminated, _, _ = env.step(np.array([0, 0], dtype=np.int64))
    assert terminated is True


def test_episode_truncates_on_max_steps():
    from itertools import repeat
    states = list(repeat(MOCK_STATE, 12))
    client = MagicMock()
    client.receive_state.side_effect = states
    env = MindustryEnv(client=client, max_steps=5)
    env.reset()
    action = np.array([0, 0], dtype=np.int64)
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(action)
    assert not truncated
    _, _, terminated, truncated, _ = env.step(action)
    assert truncated is True


DEFAULT_MAPS = ["Ancient Caldera", "Windswept Islands"]


def test_reset_sends_reset_command_with_map():
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client, maps=DEFAULT_MAPS)
    env.reset()
    client.send_command.assert_called_with("RESET;Ancient Caldera")


def test_reset_cycles_maps_on_successive_resets():
    client = MagicMock()
    client.receive_state.return_value = MOCK_STATE
    env = MindustryEnv(client=client, maps=DEFAULT_MAPS)

    env.reset()
    client.send_command.assert_called_with("RESET;Ancient Caldera")

    env.reset()
    client.send_command.assert_called_with("RESET;Windswept Islands")

    env.reset()
    client.send_command.assert_called_with("RESET;Ancient Caldera")


def test_reset_uses_default_maps_when_none_provided():
    client = make_mock_client(states=[MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    client.send_command.assert_called_once()
    call_arg = client.send_command.call_args[0][0]
    assert call_arg.startswith("RESET;")


def test_env_accepts_tcp_port_parameter():
    """MindustryEnv should accept tcp_port and use it to connect."""
    env = MindustryEnv(tcp_port=9002, client=MagicMock())
    assert env._port == 9002


def test_action_masks_returns_correct_shape():
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    mask = env.action_masks()
    assert mask.shape == (21,)
    assert mask.dtype == np.bool_


def test_action_masks_before_reset_returns_all_true():
    env = MindustryEnv(client=MagicMock())
    mask = env.action_masks()
    assert mask.shape == (21,)
    assert np.all(mask)


def test_step_info_contains_game_state_keys():
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    _, _, _, _, info = env.step(np.array([0, 0], dtype=np.int64))
    assert "step_latency_ms" in info
    assert "resources" in info
    assert "power" in info
    assert "buildings" in info
    assert "units" in info
    assert "build_failed" in info
    assert isinstance(info["step_latency_ms"], float)
    assert info["step_latency_ms"] >= 0.0
    assert isinstance(info["resources"], dict)
    assert isinstance(info["buildings"], int)
    assert isinstance(info["units"], int)
    assert isinstance(info["build_failed"], bool)


def test_step_info_tracks_drills_built_this_step():
    """Info dict should contain drills_built_this_step count."""
    prev_state = {**MOCK_STATE, "buildings": []}
    curr_state = {
        **MOCK_STATE,
        "buildings": [
            {"block": "drill", "x": 10, "y": 20, "hp": 1.0},
            {"block": "duo", "x": 15, "y": 25, "hp": 1.0},
        ]
    }
    client = make_mock_client(states=[prev_state, curr_state])
    env = MindustryEnv(client=client)
    env.reset()
    _, _, _, _, info = env.step(np.array([5, 0], dtype=np.int64))
    assert "drills_built_this_step" in info
    assert info["drills_built_this_step"] == 1


def test_step_info_tracks_penalty_triggers():
    """Info dict should contain penalty_a_triggered and penalty_b_triggered."""
    prev_state = {**MOCK_STATE, "resources": {"copper": 100, "lead": 50, "graphite": 0, "titanium": 0, "thorium": 0}}
    curr_state = {
        **MOCK_STATE,
        "resources": {"copper": 100, "lead": 50, "graphite": 0, "titanium": 0, "thorium": 0}
    }
    client = make_mock_client(states=[prev_state, curr_state])
    env = MindustryEnv(client=client)
    env.reset()
    # Multiple WAIT actions (0) should not collect resources
    for _ in range(3):
        env._action_history.append(0)
    _, _, _, _, info = env.step(np.array([0, 0], dtype=np.int64))
    assert "penalty_a_triggered" in info
    assert "penalty_b_triggered" in info
    assert isinstance(info["penalty_a_triggered"], int)
    assert isinstance(info["penalty_b_triggered"], int)
    assert info["penalty_a_triggered"] in (0, 1)
    assert info["penalty_b_triggered"] in (0, 1)


def test_step_info_tracks_action_taken_and_step_count():
    """Info dict should contain action_taken_index and step_count."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([5, 2], dtype=np.int64)
    _, _, _, _, info = env.step(action)
    assert "action_taken_index" in info
    assert "step_count" in info
    assert info["action_taken_index"] == 5
    assert info["step_count"] == 1


def test_action_history_includes_current_action_in_reward():
    """Current action must be in history when reward is computed (Bug 3 fix).

    With 6 prior WAITs in history and a 7th WAIT as current action,
    penalty_a should fire on THIS step (not the next).
    min_history_len=7 (raised from 3 to reduce false positives from navigation).
    """
    mock_state = {
        "core": {"hp": 1.0, "x": 5, "y": 5, "size": 3},
        "wave": 1, "waveTime": 300,
        "resources": {"copper": 0, "lead": 0, "graphite": 0,
                      "titanium": 0, "thorium": 0},
        "power": {"produced": 0, "consumed": 0, "stored": 0, "capacity": 0},
        "buildings": [], "enemies": [], "friendlyUnits": [],
        "player": {"alive": True, "hp": 1.0, "x": 5, "y": 5},
        "grid": [], "inventory": {}, "actionFailed": False,
        "nearbyOres": [], "nearbyEnemies": [], "tick": 1000, "time": 500,
    }
    client = MagicMock()
    client.receive_state.return_value = mock_state

    env = MindustryEnv(client=client)
    env._prev_state = mock_state
    env._action_history = [0, 0, 0, 0, 0, 0]  # Six prior WAITs already in history

    # Seventh WAIT — penalty_a should fire on THIS step (current action in history)
    action = np.array([0, 0])  # WAIT
    _, _, _, _, info = env.step(action)

    assert info["penalty_a_triggered"] == 1, (
        "penalty_a should fire on the 7th WAIT when current action is in history"
    )


def test_step_build_conveyor_sends_player_build():
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([7, 3], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;conveyor;3")


def test_step_build_graphite_press_sends_player_build():
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([8, 0], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;graphite-press;0")


def test_step_build_silicon_smelter_sends_player_build():
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([9, 1], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;silicon-smelter;1")


def test_step_build_combustion_gen_sends_player_build():
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([10, 2], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;combustion-generator;2")


def test_step_build_pneumatic_drill_sends_player_build():
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([11, 5], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;pneumatic-drill;5")


def test_action_masks_respects_curriculum_phase0(monkeypatch):
    import rl.rewards.multi_objective as mo
    monkeypatch.setattr(mo, "CURRICULUM_ENABLED", True)

    from rl.env.spaces import NUM_ACTION_TYPES

    env = MindustryEnv(client=make_mock_client())
    env.reset()
    env._global_timestep = 10000

    mask = env.action_masks()
    action_mask = mask[:NUM_ACTION_TYPES]
    assert action_mask[0] is True or bool(action_mask[0]) is True
    assert action_mask[1] is True or bool(action_mask[1]) is True
    assert action_mask[5] is True or bool(action_mask[5]) is True
    assert bool(action_mask[2]) is False
    assert bool(action_mask[7]) is False


def test_env_has_global_timestep_attribute():
    env = MindustryEnv(client=make_mock_client())
    assert hasattr(env, "_global_timestep")
    assert env._global_timestep == 0
