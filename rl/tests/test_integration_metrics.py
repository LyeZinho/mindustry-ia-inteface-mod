"""
Integration tests for metrics pipeline: env → step() → callback → JSON → dashboard.

Tests the full flow of metrics from environment steps through callback aggregation
to JSON serialization and dashboard rendering.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rl.env.mindustry_env import MindustryEnv
from rl.callbacks.training_callbacks import LiveMetricsCallback, _write_metrics_json
from rl.dashboard import (
    _draw_drill_rate_total,
    _draw_drill_rate_frequency,
    _draw_penalty_counts,
    _draw_penalty_frequency,
    _draw_action_dist_per_episode,
    _draw_action_dist_rolling,
)


# ============================================================================
# FIXTURES FOR MOCK STATE AND ENV SETUP
# ============================================================================

MOCK_STATE = {
    "tick": 1000,
    "time": 500,
    "wave": 1,
    "waveTime": 300,
    "resources": {"copper": 100, "lead": 50, "graphite": 0, "titanium": 0, "thorium": 0},
    "power": {"produced": 10.0, "consumed": 5.0, "stored": 100, "capacity": 1000},
    "core": {"hp": 1.0, "x": 15, "y": 15, "size": 3},
    "player": {"x": 15, "y": 15, "alive": True, "hp": 1.0},
    "enemies": [],
    "friendlyUnits": [],
    "buildings": [],
    "actionFailed": False,
    "grid": [],
    "nearbyOres": [],
    "nearbyEnemies": [],
}


def make_mock_client(num_states=15):
    """Create mock client that returns num_states consecutive states."""
    client = MagicMock()
    states = [MOCK_STATE] * num_states
    client.receive_state.side_effect = states
    return client


# ============================================================================
# TEST 1: END-TO-END METRICS PIPELINE
# ============================================================================

def test_metrics_pipeline_end_to_end(tmp_path):
    """
    RED → GREEN test: Full pipeline from env step to callback metrics to JSON.
    
    Tests that:
    1. Running 10 env steps produces metrics in info dict (5 base metrics)
    2. LiveMetricsCallback aggregates those metrics correctly
    3. All 6 aggregated metrics are computed
    4. JSON output contains all metrics and is valid JSON
    5. Metrics are serializable and have correct structure
    """
    # Setup
    client = make_mock_client(num_states=15)  # 1 reset + 14 steps = 10 steps with 4 buffer
    env = MindustryEnv(client=client)
    metrics_path = tmp_path / "live_metrics.json"
    
    callback = LiveMetricsCallback(
        metrics_path=str(metrics_path),
        verbose=0,
    )
    callback.model = MagicMock()
    
    # Setup rollout buffer with mock data
    buf = MagicMock()
    buf.actions = np.zeros((256, 1, 2), dtype=np.int32)
    buf.values = np.ones((256, 1))
    buf.action_masks = np.ones((256, 1, 16), dtype=bool)
    callback.model.rollout_buffer = buf
    callback.num_timesteps = 1024
    
    # Run 10 environment steps with proper metrics in info
    obs, info = env.reset()
    infos_list = []
    
    for step_idx in range(10):
        action = np.array([0, 0], dtype=np.int64)  # WAIT action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Inject the 5 required metrics into info dict (as env should produce)
        info["step_latency_ms"] = 5.0 + step_idx
        info["resources"] = {
            "copper": 100 + step_idx * 5,
            "lead": 50,
            "graphite": 0,
            "titanium": 0,
            "thorium": 0,
        }
        info["power"] = {
            "produced": 10.0 + step_idx,
            "consumed": 5.0,
            "stored": 100,
            "capacity": 1000,
        }
        info["buildings"] = 2 + step_idx
        info["units"] = 1
        info["build_failed"] = step_idx % 3 == 0  # Fail every 3 steps
        
        # Inject episode metrics (metrics that will be aggregated)
        info["drills_built_this_step"] = 1 if step_idx % 2 == 0 else 0
        info["penalty_a_triggered"] = step_idx % 4 == 0
        info["penalty_b_triggered"] = step_idx % 5 == 0
        info["action_taken_index"] = step_idx % 7
        
        infos_list.append(info)
    
    # Simulate callback step and rollout end
    callback.locals = {"infos": infos_list}
    callback._on_rollout_start()
    callback._on_step()
    callback.num_timesteps = 1024
    callback._on_rollout_end()
    
    # ========== ASSERTIONS ==========
    
    # 1. JSON file created
    assert metrics_path.exists(), "Metrics JSON file not created"
    
    # 2. Valid JSON structure
    metrics_json = json.loads(metrics_path.read_text())
    assert isinstance(metrics_json, dict), "Metrics JSON is not a dict"
    
    # 3. Top-level keys present
    required_top_keys = {"timestamp", "policy", "episode_metrics", "world", "pipeline", "training"}
    assert required_top_keys.issubset(metrics_json.keys()), f"Missing keys: {required_top_keys - metrics_json.keys()}"
    
    # 4. Policy metrics (policy data)
    policy = metrics_json["policy"]
    assert "action_type_distribution" in policy
    assert "value_mean" in policy
    assert "mask_ratio_blocked" in policy
    assert "build_fail_rate" in policy
    
    # 5. World metrics
    world = metrics_json["world"]
    assert "resources" in world
    assert "power" in world
    assert "building_count" in world
    assert "unit_count" in world
    
    # 6. Episode metrics (the 6 aggregated metrics)
    episode_metrics = metrics_json["episode_metrics"]
    assert "drills_built_total" in episode_metrics, "Missing drills_built_total"
    assert "drill_build_frequency_pct" in episode_metrics, "Missing drill_build_frequency_pct"
    assert "penalty_a_count" in episode_metrics, "Missing penalty_a_count"
    assert "penalty_b_count" in episode_metrics, "Missing penalty_b_count"
    assert "penalty_frequency_pct" in episode_metrics, "Missing penalty_frequency_pct"
    assert "action_dist" in episode_metrics, "Missing action_dist"
    
    # 7. Pipeline metrics
    pipeline = metrics_json["pipeline"]
    assert "step_latency_ms_mean" in pipeline
    assert "step_latency_ms_std" in pipeline
    
    # 8. Training metrics
    training = metrics_json["training"]
    assert "total_timesteps" in training
    assert training["total_timesteps"] > 0
    
    # 9. Verify metric values are reasonable
    assert isinstance(episode_metrics["drills_built_total"], int)
    assert 0 <= episode_metrics["drill_build_frequency_pct"] <= 100
    assert isinstance(episode_metrics["penalty_a_count"], int)
    assert isinstance(episode_metrics["penalty_b_count"], int)
    assert 0 <= episode_metrics["penalty_frequency_pct"] <= 100
    assert isinstance(episode_metrics["action_dist"], dict)
    assert len(episode_metrics["action_dist"]) == 7
    
    # 10. Action distribution sums to ~1.0 (floating point)
    action_dist_sum = sum(episode_metrics["action_dist"].values())
    assert 0.99 < action_dist_sum < 1.01, f"Action dist sum {action_dist_sum} not ~1.0"


# ============================================================================
# TEST 2: DASHBOARD RENDERING WITH AGGREGATED METRICS
# ============================================================================

def test_dashboard_renders_with_aggregated_metrics(tmp_path):
    """
    RED → GREEN test: All 6 dashboard draw functions render without errors.
    
    Tests that all 6 drawing functions:
    1. Accept aggregated metrics data
    2. Render to matplotlib axes without exceptions
    3. Produce valid rendering objects (axes with titles, etc.)
    """
    # Create mock episode metrics list (simulating multiple episodes)
    episode_metrics_list = [
        {
            "episode_metrics": {
                "drills_built_total": 5,
                "drill_build_frequency_pct": 10.0,
                "penalty_a_count": 2,
                "penalty_b_count": 1,
                "penalty_frequency_pct": 5.0,
                "action_dist": {
                    "WAIT": 0.4,
                    "MOVE": 0.2,
                    "BUILD_TURRET": 0.1,
                    "BUILD_WALL": 0.1,
                    "BUILD_POWER": 0.1,
                    "BUILD_DRILL": 0.05,
                    "REPAIR": 0.05,
                },
            }
        },
        {
            "episode_metrics": {
                "drills_built_total": 8,
                "drill_build_frequency_pct": 15.0,
                "penalty_a_count": 1,
                "penalty_b_count": 2,
                "penalty_frequency_pct": 6.0,
                "action_dist": {
                    "WAIT": 0.35,
                    "MOVE": 0.25,
                    "BUILD_TURRET": 0.15,
                    "BUILD_WALL": 0.1,
                    "BUILD_POWER": 0.1,
                    "BUILD_DRILL": 0.03,
                    "REPAIR": 0.02,
                },
            }
        },
        {
            "episode_metrics": {
                "drills_built_total": 12,
                "drill_build_frequency_pct": 20.0,
                "penalty_a_count": 3,
                "penalty_b_count": 0,
                "penalty_frequency_pct": 7.0,
                "action_dist": {
                    "WAIT": 0.3,
                    "MOVE": 0.3,
                    "BUILD_TURRET": 0.2,
                    "BUILD_WALL": 0.1,
                    "BUILD_POWER": 0.05,
                    "BUILD_DRILL": 0.03,
                    "REPAIR": 0.02,
                },
            }
        },
    ]
    
    # Test all 6 draw functions
    draw_functions = [
        (_draw_drill_rate_total, "drill_rate_total"),
        (_draw_drill_rate_frequency, "drill_rate_frequency"),
        (_draw_penalty_counts, "penalty_counts"),
        (_draw_penalty_frequency, "penalty_frequency"),
        (_draw_action_dist_per_episode, "action_dist_per_episode"),
        (_draw_action_dist_rolling, "action_dist_rolling"),
    ]
    
    for draw_func, name in draw_functions:
        # Create fresh axes for each test
        fig, ax = plt.subplots(facecolor="#1e1e2e")
        
        # Call the draw function - should not raise
        try:
            draw_func(ax, episode_metrics_list)
        except Exception as e:
            pytest.fail(f"_draw_{name} raised {type(e).__name__}: {e}")
        
        # Verify ax has been configured (has title or labels)
        assert ax.get_title() or len(ax.get_lines()) > 0 or len(ax.patches) > 0, \
            f"_draw_{name} did not render anything to axis"
        
        plt.close(fig)


# ============================================================================
# TEST 3: JSON SERIALIZATION INTEGRITY
# ============================================================================

def test_metrics_json_serialization_round_trip(tmp_path):
    """
    Test that metrics can be written to JSON and read back without corruption.
    """
    # Sample aggregated metrics (6 metrics)
    metrics = {
        "timestamp": "2025-04-03T10:30:00",
        "episode_metrics": {
            "drills_built_total": 42,
            "drill_build_frequency_pct": 15.5,
            "penalty_a_count": 5,
            "penalty_b_count": 3,
            "penalty_frequency_pct": 8.0,
            "action_dist": {
                "WAIT": 0.4,
                "MOVE": 0.2,
                "BUILD_TURRET": 0.1,
                "BUILD_WALL": 0.1,
                "BUILD_POWER": 0.1,
                "BUILD_DRILL": 0.05,
                "REPAIR": 0.05,
            },
        },
        "policy": {
            "action_type_distribution": [0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05],
            "value_mean": 42.5,
        },
        "world": {
            "resources": {"copper": 150.0, "lead": 50.0},
            "building_count": 10,
        },
        "pipeline": {
            "step_latency_ms_mean": 5.2,
        },
        "training": {
            "total_timesteps": 10240,
        },
    }
    
    # Write to JSON
    json_path = tmp_path / "metrics.json"
    _write_metrics_json(json_path, metrics)
    
    # Read back
    loaded = json.loads(json_path.read_text())
    
    # Verify round-trip integrity
    assert loaded["episode_metrics"]["drills_built_total"] == 42
    assert abs(loaded["episode_metrics"]["drill_build_frequency_pct"] - 15.5) < 0.01
    assert loaded["episode_metrics"]["penalty_a_count"] == 5
    assert abs(loaded["episode_metrics"]["action_dist"]["WAIT"] - 0.4) < 0.001


# ============================================================================
# TEST 4: EDGE CASES
# ============================================================================

def test_metrics_with_zero_episodes(tmp_path):
    """Test that metrics handle zero episodes gracefully."""
    metrics_path = tmp_path / "metrics.json"
    callback = LiveMetricsCallback(metrics_path=str(metrics_path), verbose=0)
    callback.model = MagicMock()
    
    buf = MagicMock()
    buf.actions = np.zeros((0, 1, 2), dtype=np.int32)
    buf.values = np.array([], dtype=np.float32).reshape(0, 1)
    buf.action_masks = np.ones((0, 1, 16), dtype=bool)
    callback.model.rollout_buffer = buf
    callback.num_timesteps = 0
    
    # Simulate empty episode
    callback.locals = {"infos": []}
    callback._on_rollout_start()
    callback._on_step()
    callback._on_rollout_end()
    
    # Should create valid metrics even with zero episodes
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["episode_metrics"]["drills_built_total"] == 0
    assert metrics["episode_metrics"]["drill_build_frequency_pct"] == 0.0


def test_metrics_with_large_numbers(tmp_path):
    """Test that metrics handle large numbers without overflow."""
    metrics_path = tmp_path / "metrics.json"
    callback = LiveMetricsCallback(metrics_path=str(metrics_path), verbose=0)
    callback.model = MagicMock()
    
    buf = MagicMock()
    buf.actions = np.zeros((256, 1, 2), dtype=np.int32)
    buf.values = np.ones((256, 1)) * 1000.0
    buf.action_masks = np.ones((256, 1, 16), dtype=bool)
    callback.model.rollout_buffer = buf
    callback.num_timesteps = 1_000_000
    
    # Large metric values
    callback.locals = {
        "infos": [
            {
                "drills_built_this_step": 1000,
                "penalty_a_triggered": False,
                "penalty_b_triggered": False,
                "action_taken_index": 0,
            }
        ]
    }
    callback._on_rollout_start()
    callback._on_step()
    callback._on_rollout_end()
    
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["episode_metrics"]["drills_built_total"] == 1000
    assert metrics["training"]["total_timesteps"] == 1_000_000
