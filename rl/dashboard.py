"""Standalone live dashboard for SB3 training — reads monitor CSV, no training deps."""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd


def load_monitor_csv(source: str | io.StringIO) -> pd.DataFrame:
    """Parse SB3 monitor CSV from a file-like object, skipping the JSON header line."""
    if hasattr(source, "read"):
        text = source.read()
    else:
        text = source
    lines = text.splitlines()
    data_lines = [l for l in lines if not l.startswith("#")]
    if len(data_lines) <= 1:
        return pd.DataFrame(columns=["r", "l", "t"])
    return pd.read_csv(io.StringIO("\n".join(data_lines)))


def load_monitor_csv_path(path: Path) -> pd.DataFrame:
    """Load monitor CSV from a filesystem path; return empty DataFrame if missing."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["r", "l", "t"])
    return load_monitor_csv(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live training dashboard")
    p.add_argument("--csv", default="rl/logs/monitor.monitor.csv")
    p.add_argument("--interval", type=float, default=3.0, help="Refresh interval in seconds")
    p.add_argument("--window", type=int, default=50, help="Rolling average window")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
