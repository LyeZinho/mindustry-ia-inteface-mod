import json
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
import pytest

from rl.callbacks.training_callbacks import _write_metrics_json, LiveMetricsCallback
from rl.env.spaces import NUM_ACTION_TYPES


def test_write_metrics_json_creates_file(tmp_path):
    dest = tmp_path / "metrics.json"
    _write_metrics_json(dest, {"key": "value", "num": 42})
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["key"] == "value"
    assert data["num"] == 42


def test_write_metrics_json_overwrites(tmp_path):
    dest = tmp_path / "metrics.json"
    _write_metrics_json(dest, {"v": 1})
    _write_metrics_json(dest, {"v": 2})
    data = json.loads(dest.read_text())
    assert data["v"] == 2


def _make_callback(tmp_path) -> LiveMetricsCallback:
    cb = LiveMetricsCallback(
        metrics_path=str(tmp_path / "live_metrics.json"),
        verbose=0,
    )
    cb.model = MagicMock()
    return cb


def test_live_metrics_callback_step_accumulates_latency(tmp_path):
    cb = _make_callback(tmp_path)
    cb.locals = {"infos": [{"step_latency_ms": 12.5}, {"step_latency_ms": 7.0}]}
    cb._on_step()
    assert list(cb._step_latencies) == [12.5, 7.0]


def test_live_metrics_callback_step_accumulates_resources(tmp_path):
    cb = _make_callback(tmp_path)
    cb.locals = {"infos": [{"resources": {"copper": 150, "lead": 30}}]}
    cb._on_step()
    assert cb._last_resources == {"copper": 150.0, "lead": 30.0}


def test_live_metrics_callback_writes_on_rollout_end(tmp_path):
    cb = _make_callback(tmp_path)
    metrics_file = tmp_path / "live_metrics.json"

    buf = MagicMock()
    buf.actions = np.zeros((256, 1, 2), dtype=np.int32)
    buf.values = np.ones((256, 1))
    buf.action_masks = np.ones((256, 1, 16), dtype=bool)
    cb.model.rollout_buffer = buf
    cb.num_timesteps = 1024

    cb._on_rollout_end()

    assert metrics_file.exists()
    data = json.loads(metrics_file.read_text())
    assert "policy" in data
    assert "world" in data
    assert "pipeline" in data
    assert "training" in data
    assert data["training"]["total_timesteps"] == 1024


def test_live_metrics_callback_step_accumulates_build_fails(tmp_path):
    cb = _make_callback(tmp_path)
    cb.locals = {"infos": [{"build_failed": True}, {"build_failed": False}, {"build_failed": True}]}
    cb._on_step()
    assert cb._rollout_build_fails == 2


def test_compute_metrics_aggregates_drill_metrics(tmp_path):
    """Test that _compute_metrics aggregates drill metrics from episode_infos."""
    cb = _make_callback(tmp_path)
    buf = MagicMock()
    buf.actions = np.zeros((256, 1, 2), dtype=np.int32)
    buf.values = np.ones((256, 1))
    buf.action_masks = np.ones((256, 1, 16), dtype=bool)
    cb.model.rollout_buffer = buf
    cb.num_timesteps = 1024
    
    # Simulate episode infos with drill metrics
    episode_infos = [
        {"drills_built_this_step": 2},
        {"drills_built_this_step": 1},
        {"drills_built_this_step": 0},
    ]
    cb.locals = {"infos": episode_infos}
    cb._on_step()  # accumulate the step
    
    metrics = cb._compute_metrics()
    
    # Should have drill aggregation metrics
    assert "episode_metrics" in metrics
    assert metrics["episode_metrics"]["drills_built_total"] == 3
    assert metrics["episode_metrics"]["drill_build_frequency_pct"] == (3 / 3) * 100  # 100%


def test_compute_metrics_aggregates_penalty_metrics(tmp_path):
    """Test that _compute_metrics aggregates penalty metrics from episode_infos."""
    cb = _make_callback(tmp_path)
    buf = MagicMock()
    buf.actions = np.zeros((256, 1, 2), dtype=np.int32)
    buf.values = np.ones((256, 1))
    buf.action_masks = np.ones((256, 1, 16), dtype=bool)
    cb.model.rollout_buffer = buf
    cb.num_timesteps = 1024
    
    # Simulate episode infos with penalty metrics
    episode_infos = [
        {"penalty_a_triggered": True, "penalty_b_triggered": False},
        {"penalty_a_triggered": False, "penalty_b_triggered": True},
        {"penalty_a_triggered": True, "penalty_b_triggered": True},
    ]
    cb.locals = {"infos": episode_infos}
    cb._on_step()
    
    metrics = cb._compute_metrics()
    
    # Should have penalty aggregation metrics
    assert "episode_metrics" in metrics
    assert metrics["episode_metrics"]["penalty_a_count"] == 2
    assert metrics["episode_metrics"]["penalty_b_count"] == 2
    assert metrics["episode_metrics"]["penalty_frequency_pct"] == ((2 + 2) / 3) * 100  # 133.33%


def test_compute_metrics_aggregates_action_distribution(tmp_path):
    """Test that _compute_metrics aggregates action distribution from episode_infos."""
    cb = _make_callback(tmp_path)
    buf = MagicMock()
    buf.actions = np.zeros((256, 1, 2), dtype=np.int32)
    buf.values = np.ones((256, 1))
    buf.action_masks = np.ones((256, 1, 16), dtype=bool)
    cb.model.rollout_buffer = buf
    cb.num_timesteps = 1024
    
    episode_infos = [
        {"action_taken_index": 0},
        {"action_taken_index": 1},
        {"action_taken_index": 1},
        {"action_taken_index": 5},
    ]
    cb.locals = {"infos": episode_infos}
    cb._on_step()
    
    metrics = cb._compute_metrics()
    
    assert "episode_metrics" in metrics
    action_dist = metrics["episode_metrics"]["action_dist"]
    assert action_dist["WAIT"] == 0.25
    assert action_dist["MOVE"] == 0.50
    assert action_dist["BUILD_TURRET"] == 0.0
    assert action_dist["BUILD_WALL"] == 0.0
    assert action_dist["BUILD_POWER"] == 0.0
    assert action_dist["BUILD_DRILL"] == 0.25
    assert action_dist["REPAIR"] == 0.0
    assert abs(sum(action_dist.values()) - 1.0) < 1e-6


def test_compute_metrics_edge_case_zero_steps(tmp_path):
    """Test that _compute_metrics handles zero steps gracefully."""
    cb = _make_callback(tmp_path)
    buf = MagicMock()
    buf.actions = np.zeros((256, 1, 2), dtype=np.int32)
    buf.values = np.ones((256, 1))
    buf.action_masks = np.ones((256, 1, 16), dtype=bool)
    cb.model.rollout_buffer = buf
    cb.num_timesteps = 1024
    
    cb.locals = {"infos": []}
    cb._on_step()
    
    metrics = cb._compute_metrics()
    
    assert "episode_metrics" in metrics
    assert metrics["episode_metrics"]["drills_built_total"] == 0
    assert metrics["episode_metrics"]["drill_build_frequency_pct"] == 0.0
    assert metrics["episode_metrics"]["penalty_a_count"] == 0
    assert metrics["episode_metrics"]["penalty_b_count"] == 0
    assert metrics["episode_metrics"]["penalty_frequency_pct"] == 0.0
    action_dist = metrics["episode_metrics"]["action_dist"]
    assert all(abs(v - 1.0/NUM_ACTION_TYPES) < 1e-6 for v in action_dist.values())
    assert abs(sum(action_dist.values()) - 1.0) < 1e-6


def test_compute_metrics_edge_case_no_actions(tmp_path):
    """Test that _compute_metrics handles no action_taken_index gracefully."""
    cb = _make_callback(tmp_path)
    buf = MagicMock()
    buf.actions = np.zeros((256, 1, 2), dtype=np.int32)
    buf.values = np.ones((256, 1))
    buf.action_masks = np.ones((256, 1, 16), dtype=bool)
    cb.model.rollout_buffer = buf
    cb.num_timesteps = 1024
    
    episode_infos = [
        {"drills_built_this_step": 1},
        {"drills_built_this_step": 2},
    ]
    cb.locals = {"infos": episode_infos}
    cb._on_step()
    
    metrics = cb._compute_metrics()
    
    assert "episode_metrics" in metrics
    action_dist = metrics["episode_metrics"]["action_dist"]
    assert all(abs(v - 1.0/NUM_ACTION_TYPES) < 1e-6 for v in action_dist.values())
    assert abs(sum(action_dist.values()) - 1.0) < 1e-6


def test_callback_sets_global_timestep_on_rollout(tmp_path):
    """Test that _on_rollout_start propagates num_timesteps to training envs via set_attr."""
    cb = _make_callback(tmp_path)

    mock_env = MagicMock()
    cb.model.get_env.return_value = mock_env
    cb.num_timesteps = 75000

    cb._on_rollout_start()

    mock_env.set_attr.assert_called_with("_global_timestep", 75000)
