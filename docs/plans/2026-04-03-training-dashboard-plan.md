# Training Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Standalone matplotlib window (`rl/dashboard.py`) that live-monitors SB3 training by polling `monitor.monitor.csv` every N seconds.

**Architecture:** Single module, no imports from the training codebase. `FuncAnimation` drives 3s refresh cycle. Reads CSV with pandas, skips the SB3 JSON metadata line, computes rolling stats, redraws 4 axes.

**Tech Stack:** Python 3, matplotlib (FuncAnimation), pandas (read_csv), argparse

---

### Task 1: Skeleton + CLI + CSV reader

**Files:**
- Create: `rl/dashboard.py`
- Create: `rl/tests/test_dashboard.py`

**Step 1: Write the failing test**

```python
# rl/tests/test_dashboard.py
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

def test_load_monitor_csv_missing_file_returns_empty_df(tmp_path):
    from rl.dashboard import load_monitor_csv_path
    df = load_monitor_csv_path(tmp_path / "nonexistent.csv")
    assert len(df) == 0
```

**Step 2: Run to verify it fails**

```bash
source rl/venv/bin/activate
python -m pytest rl/tests/test_dashboard.py -v
```
Expected: `ImportError` — `rl.dashboard` doesn't exist yet.

**Step 3: Implement `rl/dashboard.py` skeleton**

```python
"""Standalone live dashboard for SB3 training — reads monitor CSV, no training deps."""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd


def load_monitor_csv(source) -> pd.DataFrame:
    """Parse SB3 monitor CSV from a file-like object, skipping the JSON header line."""
    if hasattr(source, "read"):
        text = source.read()
    else:
        text = source
    lines = text.splitlines()
    # Drop the first line if it starts with '#'
    data_lines = [l for l in lines if not l.startswith("#")]
    if len(data_lines) <= 1:  # only header or empty
        return pd.DataFrame(columns=["r", "l", "t"])
    return pd.read_csv(io.StringIO("\n".join(data_lines)))


def load_monitor_csv_path(path: Path) -> pd.DataFrame:
    """Load monitor CSV from a filesystem path; return empty DataFrame if missing."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["r", "l", "t"])
    return load_monitor_csv(path.read_text())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live training dashboard")
    p.add_argument("--csv", default="rl/logs/monitor.monitor.csv")
    p.add_argument("--interval", type=float, default=3.0, help="Refresh interval in seconds")
    p.add_argument("--window", type=int, default=50, help="Rolling average window")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
```

**Step 4: Run tests**

```bash
python -m pytest rl/tests/test_dashboard.py -v
```
Expected: 4 PASS.

**Step 5: Commit**

```bash
git add rl/dashboard.py rl/tests/test_dashboard.py
git commit -m "feat(dashboard): skeleton with CSV loader and CLI args"
```

---

### Task 2: Rolling stats helper

**Files:**
- Modify: `rl/dashboard.py`
- Modify: `rl/tests/test_dashboard.py`

**Step 1: Write the failing test**

```python
# append to rl/tests/test_dashboard.py
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
```

**Step 2: Run to verify it fails**

```bash
python -m pytest rl/tests/test_dashboard.py::test_compute_stats_rolling_mean -v
```
Expected: `ImportError` for `compute_stats`.

**Step 3: Implement `compute_stats`**

```python
# add to rl/dashboard.py
from typing import Dict, Any

def compute_stats(df: pd.DataFrame, window: int = 50) -> Dict[str, Any]:
    if len(df) == 0:
        return {"total_episodes": 0, "mean_reward": 0.0, "max_reward": 0.0, "mean_length": 0.0}
    tail = df.tail(window)
    return {
        "total_episodes": len(df),
        "mean_reward": float(tail["r"].mean()),
        "max_reward": float(df["r"].max()),
        "mean_length": float(tail["l"].mean()),
    }
```

**Step 4: Run tests**

```bash
python -m pytest rl/tests/test_dashboard.py -v
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add rl/dashboard.py rl/tests/test_dashboard.py
git commit -m "feat(dashboard): rolling stats helper with tests"
```

---

### Task 3: Matplotlib figure + FuncAnimation main loop

**Files:**
- Modify: `rl/dashboard.py`

No new tests for the matplotlib rendering (GUI is untestable in CI). The existing tests cover all logic.

**Step 1: Implement `build_figure` and `make_updater`**

```python
# add to rl/dashboard.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from datetime import datetime


def build_figure():
    """Create figure with 4 panels and return (fig, axes dict)."""
    fig = plt.figure(figsize=(12, 7), facecolor="#1e1e2e")
    fig.canvas.manager.set_window_title("Mindustry RL — Training Dashboard")

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_reward  = fig.add_subplot(gs[0, 0])
    ax_length  = fig.add_subplot(gs[0, 1])
    ax_hist    = fig.add_subplot(gs[1, :])
    ax_stats   = fig.add_subplot(gs[2, :])
    ax_stats.axis("off")

    for ax in (ax_reward, ax_length, ax_hist):
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

    ax_reward.set_title("Reward por episódio", color="#cdd6f4", fontsize=10)
    ax_reward.set_xlabel("Episódio", color="#a6adc8", fontsize=8)
    ax_length.set_title("Episode Length (rolling)", color="#cdd6f4", fontsize=10)
    ax_length.set_xlabel("Episódio", color="#a6adc8", fontsize=8)
    ax_hist.set_title("Distribuição de Rewards", color="#cdd6f4", fontsize=10)
    ax_hist.set_xlabel("Reward", color="#a6adc8", fontsize=8)

    return fig, {"reward": ax_reward, "length": ax_length, "hist": ax_hist, "stats": ax_stats}


def make_updater(axes, csv_path: str, window: int):
    """Return the FuncAnimation update callable."""
    def update(_frame):
        df = load_monitor_csv_path(Path(csv_path))
        stats = compute_stats(df, window)

        # --- Reward panel ---
        ax = axes["reward"]
        ax.cla()
        ax.set_facecolor("#181825")
        ax.set_title("Reward por episódio", color="#cdd6f4", fontsize=10)
        ax.set_xlabel("Episódio", color="#a6adc8", fontsize=8)
        ax.tick_params(colors="#cdd6f4")
        if len(df) > 0:
            ax.scatter(df.index, df["r"], s=4, alpha=0.4, color="#89b4fa")
            if len(df) >= window:
                rolling = df["r"].rolling(window).mean()
                ax.plot(df.index, rolling, color="#f38ba8", linewidth=1.5, label=f"Rolling {window}")
                ax.legend(fontsize=7, labelcolor="#cdd6f4", facecolor="#1e1e2e")

        # --- Length panel ---
        ax = axes["length"]
        ax.cla()
        ax.set_facecolor("#181825")
        ax.set_title("Episode Length (rolling)", color="#cdd6f4", fontsize=10)
        ax.set_xlabel("Episódio", color="#a6adc8", fontsize=8)
        ax.tick_params(colors="#cdd6f4")
        if len(df) >= window:
            rolling_l = df["l"].rolling(window).mean()
            ax.plot(df.index, rolling_l, color="#a6e3a1", linewidth=1.5)

        # --- Histogram panel ---
        ax = axes["hist"]
        ax.cla()
        ax.set_facecolor("#181825")
        ax.set_title("Distribuição de Rewards", color="#cdd6f4", fontsize=10)
        ax.set_xlabel("Reward", color="#a6adc8", fontsize=8)
        ax.tick_params(colors="#cdd6f4")
        if len(df) > 1:
            ax.hist(df["r"], bins=40, color="#cba6f7", alpha=0.7, edgecolor="#1e1e2e")

        # --- Stats text ---
        ax = axes["stats"]
        ax.cla()
        ax.axis("off")
        now = datetime.now().strftime("%H:%M:%S")
        txt = (
            f"Episódios: {stats['total_episodes']}    "
            f"Reward médio ({window}ep): {stats['mean_reward']:.3f}    "
            f"Reward máx: {stats['max_reward']:.3f}    "
            f"Length médio ({window}ep): {stats['mean_length']:.1f}    "
            f"Atualizado: {now}"
        )
        ax.text(0.5, 0.5, txt, transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#cdd6f4", fontfamily="monospace")

    return update


def main():
    args = parse_args()
    fig, axes = build_figure()
    updater = make_updater(axes, args.csv, args.window)
    ani = animation.FuncAnimation(
        fig, updater, interval=int(args.interval * 1000), cache_frame_data=False
    )
    updater(0)  # draw immediately without waiting first interval
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

**Step 2: Add `__main__.py` so `python -m rl.dashboard` works**

Check if `rl/` already has `__main__.py` — it won't, so just rely on the `if __name__ == "__main__"` block.  
The module is invoked as: `python -m rl.dashboard` (runs the file directly, `__main__` block fires).

**Step 3: Manual smoke test**

```bash
source rl/venv/bin/activate
python -m rl.dashboard --csv rl/logs/monitor.monitor.csv --interval 3
```
Expected: window opens, 4 panels render with data from existing CSV, updates every 3s.

**Step 4: Run unit tests one last time**

```bash
python -m pytest rl/tests/test_dashboard.py -v
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add rl/dashboard.py
git commit -m "feat(dashboard): live matplotlib window with reward, length, histogram and stats panels"
```
