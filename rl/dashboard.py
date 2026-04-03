"""Standalone live dashboard for SB3 training — reads monitor CSV, no training deps."""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, Any

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



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live training dashboard")
    p.add_argument("--csv", default="rl/logs/monitor.monitor.csv")
    p.add_argument("--interval", type=float, default=3.0, help="Refresh interval in seconds")
    p.add_argument("--window", type=int, default=50, help="Rolling average window")
    return p.parse_args()


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
    _last_df: list[pd.DataFrame] = [pd.DataFrame(columns=["r", "l", "t"])]

    def update(_frame):
        try:
            df = load_monitor_csv_path(Path(csv_path))
            _last_df[0] = df
        except Exception:
            df = _last_df[0]
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
