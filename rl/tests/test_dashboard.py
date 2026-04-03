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
