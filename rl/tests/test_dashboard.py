import io
import pandas as pd
from rl.dashboard import load_monitor_csv

CSV_CONTENT = """\
#{"t_start": 1234, "env_id": "None"}
r,l,t
1.5,10,1.0
-0.5,5,2.0
2.0,20,3.0
"""

def test_load_monitor_csv_returns_dataframe():
    df = load_monitor_csv(io.StringIO(CSV_CONTENT))
    assert list(df.columns) == ["r", "l", "t"]
    assert len(df) == 3

def test_load_monitor_csv_correct_values():
    df = load_monitor_csv(io.StringIO(CSV_CONTENT))
    assert df["r"].iloc[0] == 1.5
    assert df["l"].iloc[1] == 5

def test_load_monitor_csv_empty_returns_empty_df():
    empty = "#header\nr,l,t\n"
    df = load_monitor_csv(io.StringIO(empty))
    assert len(df) == 0
    assert list(df.columns) == ["r", "l", "t"]

def test_load_monitor_csv_missing_file_returns_empty_df(tmp_path):
    from rl.dashboard import load_monitor_csv_path
    df = load_monitor_csv_path(tmp_path / "nonexistent.csv")
    assert len(df) == 0
    assert list(df.columns) == ["r", "l", "t"]

from rl.dashboard import compute_stats

def test_compute_stats_rolling_mean():
    df = pd.DataFrame({"r": [1.0, 2.0, 3.0, 4.0, 5.0], "l": [10, 20, 30, 40, 50], "t": range(5)})
    stats = compute_stats(df, window=3)
    assert stats["total_episodes"] == 5
    assert abs(stats["mean_reward"] - 4.0) < 0.01   # mean of last 3: 3,4,5
    assert stats["max_reward"] == 5.0
    assert abs(stats["mean_length"] - 40.0) < 0.01  # mean of last 3 lengths

def test_compute_stats_empty():
    df = pd.DataFrame(columns=["r", "l", "t"])
    stats = compute_stats(df, window=50)
    assert stats["total_episodes"] == 0
    assert stats["mean_reward"] == 0.0


from rl.dashboard import load_live_metrics

def test_load_live_metrics_returns_dict_from_json(tmp_path):
    p = tmp_path / "live_metrics.json"
    p.write_text('{"policy": {"value_mean": 1.5}, "training": {"total_timesteps": 500}}')
    data = load_live_metrics(p)
    assert data["policy"]["value_mean"] == 1.5
    assert data["training"]["total_timesteps"] == 500


def test_load_live_metrics_returns_empty_on_missing(tmp_path):
    data = load_live_metrics(tmp_path / "nonexistent.json")
    assert data == {}


def test_load_live_metrics_returns_empty_on_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not valid json {{{{")
    data = load_live_metrics(p)
    assert data == {}


# TDD: NEW TESTS FOR 6 DRAWING FUNCTIONS (RED PHASE)

from rl.dashboard import (
    _draw_drill_rate_total,
    _draw_drill_rate_frequency,
    _draw_penalty_counts,
    _draw_penalty_frequency,
    _draw_action_dist_per_episode,
    _draw_action_dist_rolling,
)
import matplotlib.pyplot as plt


def test_draw_drill_rate_total_renders_with_valid_metrics():
    """Test _draw_drill_rate_total renders without error with valid episode metrics."""
    metrics = [
        {"episode_metrics": {"drills_built_total": 5}},
        {"episode_metrics": {"drills_built_total": 8}},
        {"episode_metrics": {"drills_built_total": 12}},
    ]
    ax = plt.subplot(1, 1, 1)
    _draw_drill_rate_total(ax, metrics)
    assert ax.get_title() != ""
    plt.close("all")


def test_draw_drill_rate_frequency_renders_with_valid_metrics():
    """Test _draw_drill_rate_frequency renders without error with valid episode metrics."""
    metrics = [
        {"episode_metrics": {"drill_build_frequency_pct": 5.5}},
        {"episode_metrics": {"drill_build_frequency_pct": 8.2}},
        {"episode_metrics": {"drill_build_frequency_pct": 12.1}},
    ]
    ax = plt.subplot(1, 1, 1)
    _draw_drill_rate_frequency(ax, metrics)
    assert ax.get_title() != ""
    plt.close("all")


def test_draw_penalty_counts_renders_with_valid_metrics():
    """Test _draw_penalty_counts renders without error with valid episode metrics."""
    metrics = [
        {"episode_metrics": {"penalty_a_count": 3, "penalty_b_count": 2}},
        {"episode_metrics": {"penalty_a_count": 1, "penalty_b_count": 5}},
    ]
    ax = plt.subplot(1, 1, 1)
    _draw_penalty_counts(ax, metrics)
    assert ax.get_title() != ""
    plt.close("all")


def test_draw_penalty_frequency_renders_with_valid_metrics():
    """Test _draw_penalty_frequency renders without error with valid episode metrics."""
    metrics = [
        {"episode_metrics": {"penalty_frequency_pct": 2.5}},
        {"episode_metrics": {"penalty_frequency_pct": 3.8}},
        {"episode_metrics": {"penalty_frequency_pct": 1.2}},
    ]
    ax = plt.subplot(1, 1, 1)
    _draw_penalty_frequency(ax, metrics)
    assert ax.get_title() != ""
    plt.close("all")


def test_draw_action_dist_per_episode_renders_with_valid_metrics():
    """Test _draw_action_dist_per_episode renders without error with valid episode metrics."""
    metrics = [
        {"episode_metrics": {"action_dist": {
            "WAIT": 0.1, "MOVE": 0.2, "BUILD_TURRET": 0.15,
            "BUILD_WALL": 0.15, "BUILD_POWER": 0.2, "BUILD_DRILL": 0.15, "REPAIR": 0.05
        }}},
    ]
    ax = plt.subplot(1, 1, 1)
    _draw_action_dist_per_episode(ax, metrics)
    assert ax.get_title() != ""
    plt.close("all")


def test_draw_action_dist_rolling_renders_with_valid_metrics():
    """Test _draw_action_dist_rolling renders without error with valid episode metrics."""
    metrics = [
        {"episode_metrics": {"action_dist": {
            "WAIT": 0.1, "MOVE": 0.2, "BUILD_TURRET": 0.15,
            "BUILD_WALL": 0.15, "BUILD_POWER": 0.2, "BUILD_DRILL": 0.15, "REPAIR": 0.05
        }}} for _ in range(10)
    ]
    ax = plt.subplot(1, 1, 1)
    _draw_action_dist_rolling(ax, metrics)
    assert ax.get_title() != ""
    plt.close("all")


def test_draw_drill_rate_total_handles_empty_metrics():
    """Test _draw_drill_rate_total handles empty metrics gracefully."""
    ax = plt.subplot(1, 1, 1)
    _draw_drill_rate_total(ax, [])
    assert ax.get_title() != ""
    plt.close("all")


def test_draw_penalty_counts_handles_empty_metrics():
    """Test _draw_penalty_counts handles empty metrics gracefully."""
    ax = plt.subplot(1, 1, 1)
    _draw_penalty_counts(ax, [])
    assert ax.get_title() != ""
    plt.close("all")


def test_existing_draw_reward_still_works():
    """Regression test: existing _draw_reward function still works."""
    from rl.dashboard import _draw_reward
    df = pd.DataFrame({"r": [1.0, 2.0, 3.0, 4.0, 5.0], "l": [10, 20, 30, 40, 50], "t": range(5)})
    ax = plt.subplot(1, 1, 1)
    _draw_reward(ax, df, window=2)
    assert ax.get_title() != ""
    plt.close("all")


def test_existing_draw_length_still_works():
    """Regression test: existing _draw_length function still works."""
    from rl.dashboard import _draw_length
    df = pd.DataFrame({"r": [1.0, 2.0, 3.0, 4.0, 5.0], "l": [10, 20, 30, 40, 50], "t": range(5)})
    ax = plt.subplot(1, 1, 1)
    _draw_length(ax, df, window=2)
    assert ax.get_title() != ""
    plt.close("all")
