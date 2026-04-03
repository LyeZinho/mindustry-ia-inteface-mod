from __future__ import annotations

import argparse
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PALETTE = {
    "bg_fig":  "#1e1e2e",
    "bg_ax":   "#181825",
    "text":    "#cdd6f4",
    "subtext": "#a6adc8",
    "blue":    "#89b4fa",
    "red":     "#f38ba8",
    "green":   "#a6e3a1",
    "purple":  "#cba6f7",
    "yellow":  "#f9e2af",
    "peach":   "#fab387",
    "teal":    "#89dceb",
    "grid":    "#45475a",
    "surface": "#313244",
}

_ACTION_LABELS = ["WAIT", "MOVE", "BUILD\nTRRT", "BUILD\nWALL", "BUILD\nPWR", "BUILD\nDRL", "REPAIR"]


def load_monitor_csv(source: str | io.StringIO) -> pd.DataFrame:
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
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["r", "l", "t"])
    return load_monitor_csv(path.read_text(encoding="utf-8"))


def load_live_metrics(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def _style_ax(ax, title: str, xlabel: str = "") -> None:
    ax.set_facecolor(_PALETTE["bg_ax"])
    ax.tick_params(colors=_PALETTE["text"], labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(_PALETTE["grid"])
    ax.set_title(title, color=_PALETTE["text"], fontsize=9, pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, color=_PALETTE["subtext"], fontsize=7)
    ax.yaxis.label.set_color(_PALETTE["subtext"])


def build_figure():
    fig = plt.figure(figsize=(17, 14), facecolor=_PALETTE["bg_fig"])
    fig.canvas.manager.set_window_title("Mindustry RL — Training Dashboard")

    gs = gridspec.GridSpec(
        5, 3,
        figure=fig,
        hspace=0.55,
        wspace=0.38,
        height_ratios=[2.5, 2.0, 2.0, 1.8, 0.7],
    )

    ax_reward   = fig.add_subplot(gs[0, 0])
    ax_length   = fig.add_subplot(gs[0, 1])
    ax_hist     = fig.add_subplot(gs[0, 2])

    ax_action   = fig.add_subplot(gs[1, 0])
    ax_value    = fig.add_subplot(gs[1, 1])
    ax_mask     = fig.add_subplot(gs[1, 2])

    ax_power    = fig.add_subplot(gs[2, 0])
    ax_latency  = fig.add_subplot(gs[2, 1])
    ax_counts   = fig.add_subplot(gs[2, 2])

    ax_resources = fig.add_subplot(gs[3, :])

    ax_stats    = fig.add_subplot(gs[4, :])
    ax_stats.axis("off")

    axes = {
        "reward":    ax_reward,
        "length":    ax_length,
        "hist":      ax_hist,
        "action":    ax_action,
        "value":     ax_value,
        "mask":      ax_mask,
        "power":     ax_power,
        "latency":   ax_latency,
        "counts":    ax_counts,
        "resources": ax_resources,
        "stats":     ax_stats,
    }
    return fig, axes


def make_updater(axes, csv_path: str, metrics_path: str, window: int):
    _state: Dict[str, Any] = {"df": pd.DataFrame(columns=["r", "l", "t"])}

    def _waiting(ax, title: str) -> None:
        ax.cla()
        _style_ax(ax, title)
        ax.text(0.5, 0.5, "Aguardando dados…", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])

    def update(_frame):
        try:
            _state["df"] = load_monitor_csv_path(Path(csv_path))
        except Exception:
            pass
        df = _state["df"]
        metrics = load_live_metrics(Path(metrics_path))
        stats = compute_stats(df, window)

        _draw_reward(axes["reward"], df, window)
        _draw_length(axes["length"], df, window)
        _draw_hist(axes["hist"], df)

        policy = metrics.get("policy", {})
        if policy:
            _draw_action_dist(axes["action"], policy)
            _draw_value(axes["value"], policy)
            _draw_mask(axes["mask"], policy)
        else:
            _waiting(axes["action"], "Distribuição de Ações")
            _waiting(axes["value"], "Value Estimate")
            _waiting(axes["mask"], "Ações Bloqueadas")

        world = metrics.get("world", {})
        pipeline = metrics.get("pipeline", {})
        if world or pipeline:
            _draw_power(axes["power"], world)
            _draw_latency(axes["latency"], pipeline)
            _draw_counts(axes["counts"], world)
            _draw_resources(axes["resources"], world)
        else:
            _waiting(axes["power"], "Power Grid")
            _waiting(axes["latency"], "Step Latency (ms)")
            _waiting(axes["counts"], "Buildings / Units")
            _waiting(axes["resources"], "Resource Throughput")

        _draw_stats(axes["stats"], stats, metrics, window)

    return update


def _draw_reward(ax, df: pd.DataFrame, window: int) -> None:
    ax.cla()
    _style_ax(ax, "Reward por episódio", "Episódio")
    if len(df) > 0:
        ax.scatter(df.index, df["r"], s=4, alpha=0.4, color=_PALETTE["blue"])
        if len(df) >= window:
            rolling = df["r"].rolling(window).mean()
            ax.plot(df.index, rolling, color=_PALETTE["red"], linewidth=1.5, label=f"Rolling {window}")
            ax.legend(fontsize=7, labelcolor=_PALETTE["text"], facecolor=_PALETTE["bg_fig"])


def _draw_length(ax, df: pd.DataFrame, window: int) -> None:
    ax.cla()
    _style_ax(ax, "Episode Length (rolling)", "Episódio")
    if len(df) >= window:
        rolling_l = df["l"].rolling(window).mean()
        ax.plot(df.index, rolling_l, color=_PALETTE["green"], linewidth=1.5)


def _draw_hist(ax, df: pd.DataFrame) -> None:
    ax.cla()
    _style_ax(ax, "Distribuição de Rewards", "Reward")
    if len(df) > 1:
        ax.hist(df["r"], bins=40, color=_PALETTE["purple"], alpha=0.7, edgecolor=_PALETTE["bg_fig"])


def _draw_action_dist(ax, policy: Dict[str, Any]) -> None:
    ax.cla()
    _style_ax(ax, "Distribuição de Ações")
    dist = policy.get("action_type_distribution", [])
    if not dist:
        return
    colors = [_PALETTE["blue"], _PALETTE["teal"], _PALETTE["red"],
              _PALETTE["yellow"], _PALETTE["peach"], _PALETTE["green"], _PALETTE["purple"]]
    xs = list(range(len(dist)))
    ax.bar(xs, dist, color=colors[:len(dist)], edgecolor=_PALETTE["bg_fig"], width=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(_ACTION_LABELS[:len(dist)], fontsize=6)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))


def _draw_value(ax, policy: Dict[str, Any]) -> None:
    ax.cla()
    _style_ax(ax, "Value Estimate", "Rollout")
    history = policy.get("value_history", [])
    if history:
        ax.plot(history, color=_PALETTE["peach"], linewidth=1.5)
        ax.axhline(policy.get("value_mean", 0.0), color=_PALETTE["red"],
                   linewidth=1, linestyle="--", alpha=0.7)


def _draw_mask(ax, policy: Dict[str, Any]) -> None:
    ax.cla()
    _style_ax(ax, "Ações Bloqueadas (%)")
    ratio = policy.get("mask_ratio_blocked", 0.0)
    pct = ratio * 100.0
    bar_color = _PALETTE["red"] if pct > 60 else _PALETTE["yellow"] if pct > 30 else _PALETTE["green"]
    ax.barh([0], [pct], color=bar_color, height=0.5)
    ax.barh([0], [100 - pct], left=[pct], color=_PALETTE["surface"], height=0.5)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.text(50, 0, f"{pct:.1f}%", ha="center", va="center",
            fontsize=11, color=_PALETTE["text"], fontweight="bold")


def _draw_power(ax, world: Dict[str, Any]) -> None:
    ax.cla()
    _style_ax(ax, "Power Grid", "Rollout")
    history = world.get("power_history", [])
    if history:
        produced = [e.get("produced", 0.0) for e in history]
        consumed = [e.get("consumed", 0.0) for e in history]
        xs = list(range(len(produced)))
        ax.plot(xs, produced, color=_PALETTE["yellow"], linewidth=1.5, label="Produção")
        ax.plot(xs, consumed, color=_PALETTE["red"], linewidth=1.5, label="Consumo")
        ax.fill_between(xs, produced, consumed,
                        where=[p >= c for p, c in zip(produced, consumed)],
                        alpha=0.15, color=_PALETTE["green"])
        ax.fill_between(xs, produced, consumed,
                        where=[p < c for p, c in zip(produced, consumed)],
                        alpha=0.15, color=_PALETTE["red"])
        ax.legend(fontsize=7, labelcolor=_PALETTE["text"], facecolor=_PALETTE["bg_fig"])


def _draw_latency(ax, pipeline: Dict[str, Any]) -> None:
    ax.cla()
    _style_ax(ax, "Step Latency (ms)", "Step (últimos 50)")
    history = pipeline.get("step_latency_history", [])
    if history:
        xs = list(range(len(history)))
        ax.fill_between(xs, history, alpha=0.3, color=_PALETTE["teal"])
        ax.plot(xs, history, color=_PALETTE["teal"], linewidth=1.0)
        mean_val = pipeline.get("step_latency_ms_mean", 0.0)
        ax.axhline(mean_val, color=_PALETTE["yellow"], linewidth=1, linestyle="--", alpha=0.8)
        ax.text(0.98, 0.95, f"μ={mean_val:.1f}ms",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=_PALETTE["yellow"])


def _draw_counts(ax, world: Dict[str, Any]) -> None:
    ax.cla()
    _style_ax(ax, "Buildings / Units", "Rollout")
    b_hist = world.get("building_history", [])
    if b_hist:
        ax.plot(b_hist, color=_PALETTE["blue"], linewidth=1.5, label="Buildings")
    unit_count = world.get("unit_count", 0)
    if b_hist:
        ax.axhline(unit_count, color=_PALETTE["peach"], linewidth=1, linestyle=":", alpha=0.8, label=f"Units={unit_count}")
        ax.legend(fontsize=7, labelcolor=_PALETTE["text"], facecolor=_PALETTE["bg_fig"])


def _draw_resources(ax, world: Dict[str, Any]) -> None:
    ax.cla()
    _style_ax(ax, "Resource Throughput (delta por rollout)")
    deltas = world.get("resource_deltas", {})
    if not deltas:
        ax.text(0.5, 0.5, "Aguardando dados…", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])
        return
    res_colors = {
        "copper": _PALETTE["peach"], "lead": _PALETTE["subtext"],
        "graphite": _PALETTE["purple"], "titanium": _PALETTE["teal"],
        "thorium": _PALETTE["red"], "silicon": _PALETTE["blue"],
        "metaglass": _PALETTE["yellow"],
    }
    keys = list(deltas.keys())
    vals = [deltas[k] for k in keys]
    colors = [res_colors.get(k, _PALETTE["text"]) for k in keys]
    bar_colors = [_PALETTE["green"] if v >= 0 else _PALETTE["red"] for v in vals]
    ax.bar(keys, vals, color=bar_colors, edgecolor=_PALETTE["bg_fig"], width=0.6)
    ax.axhline(0, color=_PALETTE["grid"], linewidth=0.8)
    ax.tick_params(axis="x", labelsize=8)


def _draw_stats(ax, stats: Dict[str, Any], metrics: Dict[str, Any], window: int) -> None:
    ax.cla()
    ax.axis("off")
    now = datetime.now().strftime("%H:%M:%S")
    ts = metrics.get("training", {})
    total_ts = ts.get("total_timesteps", 0)
    txt = (
        f"Episódios: {stats['total_episodes']}    "
        f"Reward médio ({window}ep): {stats['mean_reward']:.3f}    "
        f"Reward máx: {stats['max_reward']:.3f}    "
        f"Length médio: {stats['mean_length']:.1f}    "
        f"Timesteps: {total_ts:,}    "
        f"Atualizado: {now}"
    )
    ax.text(0.5, 0.5, txt, transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color=_PALETTE["text"], fontfamily="monospace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live training dashboard")
    p.add_argument("--csv", default="rl/logs/monitor.monitor.csv")
    p.add_argument("--metrics", default="rl/logs/live_metrics.json")
    p.add_argument("--interval", type=float, default=3.0, help="Refresh interval in seconds")
    p.add_argument("--window", type=int, default=50, help="Rolling average window")
    return p.parse_args()


def main():
    args = parse_args()
    fig, axes = build_figure()
    updater = make_updater(axes, args.csv, args.metrics, args.window)
    ani = animation.FuncAnimation(
        fig, updater, interval=int(args.interval * 1000), cache_frame_data=False
    )
    updater(0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
