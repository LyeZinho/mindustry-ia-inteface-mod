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

from rl.env.spaces import ACTION_NAMES

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

_ACTION_LABELS = ACTION_NAMES
_HEAD_NAMES = ["survival", "economy", "defense", "build"]
_HEAD_COLORS = ["#f38ba8", "#a6e3a1", "#89b4fa", "#cba6f7"]


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
    fig = plt.figure(figsize=(18, 26), facecolor=_PALETTE["bg_fig"])
    try:
        fig.canvas.manager.set_window_title("Mindustry RL — Training Dashboard")
    except Exception:
        pass

    gs = gridspec.GridSpec(
        7, 3,
        figure=fig,
        hspace=0.60,
        wspace=0.40,
        height_ratios=[2.5, 2.5, 2.5, 2.0, 2.0, 2.0, 0.5],
    )

    ax_reward   = fig.add_subplot(gs[0, 0])
    ax_length   = fig.add_subplot(gs[0, 1])
    ax_hist     = fig.add_subplot(gs[0, 2])

    ax_3d       = fig.add_subplot(gs[1:3, 0:2], projection="3d")
    ax_critic   = fig.add_subplot(gs[1, 2])
    ax_mask     = fig.add_subplot(gs[2, 2])

    ax_power    = fig.add_subplot(gs[3, 0])
    ax_latency  = fig.add_subplot(gs[3, 1])
    ax_counts   = fig.add_subplot(gs[3, 2])

    ax_resources      = fig.add_subplot(gs[4, 0:2])
    ax_stability      = fig.add_subplot(gs[4, 2])

    ax_lookahead      = fig.add_subplot(gs[5, 0])
    ax_placement      = fig.add_subplot(gs[5, 1])
    ax_defense_gap    = fig.add_subplot(gs[5, 2])

    ax_stats    = fig.add_subplot(gs[6, :])
    ax_stats.axis("off")

    axes = {
        "reward":       ax_reward,
        "length":       ax_length,
        "hist":         ax_hist,
        "3d_surface":   ax_3d,
        "critic":       ax_critic,
        "mask":         ax_mask,
        "power":        ax_power,
        "latency":      ax_latency,
        "counts":       ax_counts,
        "resources":    ax_resources,
        "stability":    ax_stability,
        "lookahead":    ax_lookahead,
        "placement":    ax_placement,
        "defense_gap":  ax_defense_gap,
        "stats":        ax_stats,
    }
    return fig, axes


def make_updater(axes, csv_path: str, metrics_path: str, window: int):
    _state: Dict[str, Any] = {
        "df": pd.DataFrame(columns=["r", "l", "t"]),
        "metrics_history": [],
    }

    def _waiting(ax, title: str) -> None:
        ax.cla()
        _style_ax(ax, title)
        # Axes3D.text() requires (x,y,z,s) — use text2D which works for both 2D and 3D
        text_fn = getattr(ax, "text2D", ax.text)
        text_fn(0.5, 0.5, "Aguardando dados…", transform=ax.transAxes,
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
        _draw_stability(axes["stability"], df)

        if metrics:
            _state["metrics_history"].append(metrics)
            if len(_state["metrics_history"]) > 200:
                _state["metrics_history"] = _state["metrics_history"][-200:]

        mh = _state["metrics_history"]

        policy = metrics.get("policy", {})
        if policy:
            _draw_mask(axes["mask"], policy)
        else:
            _waiting(axes["mask"], "Ações Bloqueadas")

        critic_heads_all = []
        for m in mh:
            for entry in m.get("critic_heads", []):
                if len(entry) == 4:
                    critic_heads_all.append(entry)

        if critic_heads_all:
            _draw_3d_surface(axes["3d_surface"], critic_heads_all)
            _draw_critic_heads(axes["critic"], critic_heads_all)
        else:
            _waiting(axes["3d_surface"], "3D Reward Surface")
            _waiting(axes["critic"], "Critic Head Values")

        world = metrics.get("world", {})
        pipeline = metrics.get("pipeline", {})
        if world or pipeline:
            _draw_power(axes["power"], world)
            _draw_latency(axes["latency"], pipeline)
            _draw_counts(axes["counts"], world)
            _draw_resources(axes["resources"], world)
        else:
            _waiting(axes["power"], "Power Grid")
            _waiting(axes["latency"], "Step Latency + Jitter (ms)")
            _waiting(axes["counts"], "Buildings / Units")
            _waiting(axes["resources"], "Resource Throughput")

        if mh:
            _draw_lookahead_heatmap(axes["lookahead"], mh)
        else:
            _waiting(axes["lookahead"], "Lookahead Heatmap")

        latest = metrics.get("world", {})
        if latest:
            _draw_placement_scores(axes["placement"], mh)
            _draw_defense_gap(axes["defense_gap"], mh)
        else:
            _waiting(axes["placement"], "Placement Scores")
            _waiting(axes["defense_gap"], "Defense Gap / Power Deficit")

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


def _draw_stability(ax, df: pd.DataFrame, window: int = 20) -> None:
    ax.cla()
    _style_ax(ax, f"Stability Index (σ reward, {window}ep)", "Episódio")
    if len(df) < window:
        ax.text(0.5, 0.5, f"Aguardando {window} ep…", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])
        return
    rolling_std = df["r"].rolling(window).std().dropna()
    xs = rolling_std.index
    vals = rolling_std.values
    bar_color = _PALETTE["red"] if vals[-1] > 1.0 else _PALETTE["yellow"] if vals[-1] > 0.5 else _PALETTE["green"]
    ax.plot(xs, vals, color=bar_color, linewidth=1.5)
    ax.fill_between(xs, vals, alpha=0.15, color=bar_color)
    ax.axhline(vals[-1], color=bar_color, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(0.98, 0.95, f"σ={vals[-1]:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=bar_color, fontweight="bold")


def _draw_3d_surface(ax, head_history: list) -> None:
    ax.cla()
    ax.set_facecolor(_PALETTE["bg_ax"])
    ax.set_title("3D Reward Surface", color=_PALETTE["text"], fontsize=9, pad=4)

    if len(head_history) < 10:
        return

    arr = np.array(head_history[-200:], dtype=np.float32)
    economy  = arr[:, 1]
    defense  = arr[:, 2]
    survival = arr[:, 0]
    build    = arr[:, 3]
    combined = 0.8 * survival + 0.8 * economy + 0.8 * defense + 0.6 * build

    try:
        from scipy.interpolate import griddata
        xi = np.linspace(economy.min(), economy.max(), 20)
        yi = np.linspace(defense.min(), defense.max(), 20)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata((economy, defense), combined, (XI, YI), method="linear")
        mask = ~np.isnan(ZI)
        if mask.any():
            ax.plot_surface(
                XI, YI, np.where(mask, ZI, 0),
                cmap="RdYlGn", alpha=0.8, linewidth=0,
            )
    except Exception:
        ax.scatter(economy, defense, combined, c=combined, cmap="RdYlGn", s=4, alpha=0.7)

    ax.set_xlabel("Economy", color=_PALETTE["subtext"], fontsize=6)
    ax.set_ylabel("Defense", color=_PALETTE["subtext"], fontsize=6)
    ax.set_zlabel("Combined", color=_PALETTE["subtext"], fontsize=6)
    ax.tick_params(colors=_PALETTE["text"], labelsize=5)


def _draw_critic_heads(ax, head_history: list) -> None:
    ax.cla()
    _style_ax(ax, "Critic Head Values", "Step")
    arr = np.array(head_history[-100:], dtype=np.float32)
    xs = list(range(len(arr)))
    for i, (name, color) in enumerate(zip(_HEAD_NAMES, _HEAD_COLORS)):
        ax.plot(xs, arr[:, i], color=color, linewidth=1.2, label=name)
    ax.legend(fontsize=6, labelcolor=_PALETTE["text"], facecolor=_PALETTE["bg_fig"],
              loc="upper left", ncol=2)


def _draw_mask(ax, policy: Dict[str, Any]) -> None:
    ax.cla()
    _style_ax(ax, "Ações Bloqueadas / Build Fails")
    mask_pct = policy.get("mask_ratio_blocked", 0.0) * 100
    fail_pct = policy.get("build_fail_rate", 0.0) * 100
    labels = ["Mascaradas", "Build Fail"]
    values = [mask_pct, fail_pct]
    bar_colors = [
        _PALETTE["red"] if mask_pct > 60 else _PALETTE["yellow"] if mask_pct > 30 else _PALETTE["green"],
        _PALETTE["red"] if fail_pct > 20 else _PALETTE["yellow"] if fail_pct > 5 else _PALETTE["green"],
    ]
    for i, (label, val, color) in enumerate(zip(labels, values, bar_colors)):
        ax.barh([i], [val], color=color, height=0.4)
        ax.barh([i], [100 - val], left=[val], color=_PALETTE["surface"], height=0.4)
        ax.text(50, i, f"{label}: {val:.1f}%", ha="center", va="center",
                fontsize=9, color=_PALETTE["text"], fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_yticks([])


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
    _style_ax(ax, "Step Latency + Jitter (ms)", "Step (últimos 50)")
    history = pipeline.get("step_latency_history", [])
    if history:
        xs = list(range(len(history)))
        arr = np.array(history)
        ax.fill_between(xs, arr, alpha=0.3, color=_PALETTE["teal"])
        ax.plot(xs, arr, color=_PALETTE["teal"], linewidth=1.0)
        mean_val = pipeline.get("step_latency_ms_mean", 0.0)
        std_val = pipeline.get("step_latency_ms_std", 0.0)
        ax.axhline(mean_val, color=_PALETTE["yellow"], linewidth=1, linestyle="--", alpha=0.8)
        if std_val > 0:
            ax.axhline(mean_val + std_val, color=_PALETTE["red"], linewidth=0.8,
                       linestyle=":", alpha=0.6)
            ax.axhline(mean_val - std_val, color=_PALETTE["red"], linewidth=0.8,
                       linestyle=":", alpha=0.6)
        ax.text(0.98, 0.95, f"μ={mean_val:.1f}ms  σ={std_val:.1f}ms",
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
        ax.axhline(unit_count, color=_PALETTE["peach"], linewidth=1, linestyle=":", alpha=0.8,
                   label=f"Units={unit_count}")
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
    bar_colors = [_PALETTE["green"] if v >= 0 else _PALETTE["red"] for v in vals]
    ax.bar(keys, vals, color=bar_colors, edgecolor=_PALETTE["bg_fig"], width=0.6)
    ax.axhline(0, color=_PALETTE["grid"], linewidth=0.8)
    ax.tick_params(axis="x", labelsize=8)


def _draw_lookahead_heatmap(ax, metrics_history: list) -> None:
    ax.cla()
    _style_ax(ax, "Lookahead Scores Heatmap")

    rows = []
    for m in metrics_history[-50:]:
        for entry in m.get("critic_heads", []):
            if len(entry) == 4:
                rows.append(entry)

    if len(rows) < 2:
        ax.text(0.5, 0.5, "Aguardando dados…", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])
        return

    arr = np.array(rows[-50:], dtype=np.float32).T
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_yticks(list(range(len(_HEAD_NAMES))))
    ax.set_yticklabels(_HEAD_NAMES, fontsize=6, color=_PALETTE["text"])
    ax.set_xlabel("Step", color=_PALETTE["subtext"], fontsize=6)


def _draw_placement_scores(ax, metrics_history: list) -> None:
    ax.cla()
    _style_ax(ax, "Placement Scores (Slot 0-8)")

    slots = list(range(9))
    scores = [0.0] * 9

    for m in reversed(metrics_history[-20:]):
        world = m.get("world", {})
        resources = world.get("resources", {})
        if resources:
            copper = float(resources.get("copper", 0.0))
            for i in range(9):
                scores[i] = min(1.0, copper / (500.0 * (i + 1)))
            break

    colors = [_PALETTE["green"] if s > 0.5 else _PALETTE["yellow"] if s > 0.2 else _PALETTE["red"]
              for s in scores]
    ax.bar(slots, scores, color=colors, edgecolor=_PALETTE["bg_fig"], width=0.7)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(slots)
    ax.set_xticklabels([f"S{i}" for i in slots], fontsize=6)


def _draw_defense_gap(ax, metrics_history: list) -> None:
    ax.cla()
    _style_ax(ax, "Defense Gap / Power Deficit", "Rollout")

    defense_vals = []
    power_vals = []
    for m in metrics_history:
        world = m.get("world", {})
        power = world.get("power", {})
        produced = float(power.get("produced", 0.0))
        consumed = float(power.get("consumed", 0.0))
        deficit = max(0.0, consumed - produced)
        power_vals.append(min(1.0, deficit / max(produced, 1.0)))

        enemies_present = 1.0 if len(m.get("world", {}).get("resources", {})) < 2 else 0.0
        defense_vals.append(enemies_present)

    if power_vals:
        xs = list(range(len(power_vals)))
        ax.plot(xs, power_vals, color=_PALETTE["yellow"], linewidth=1.2, label="Power deficit")
        ax.legend(fontsize=6, labelcolor=_PALETTE["text"], facecolor=_PALETTE["bg_fig"])
        ax.set_ylim(0, 1.1)


def _draw_drill_rate_total(ax, metrics_history: list) -> None:
    ax.cla()
    _style_ax(ax, "Drills Construídos (total)", "Episódio")
    vals = [m.get("episode_metrics", {}).get("drills_built_total", 0) for m in metrics_history]
    if vals:
        ax.plot(list(range(len(vals))), vals, color=_PALETTE["green"], linewidth=1.5)
    else:
        ax.text(0.5, 0.5, "Sem dados", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])


def _draw_drill_rate_frequency(ax, metrics_history: list) -> None:
    ax.cla()
    _style_ax(ax, "Drill Build Frequency (%)", "Episódio")
    vals = [m.get("episode_metrics", {}).get("drill_build_frequency_pct", 0.0) for m in metrics_history]
    if vals:
        ax.plot(list(range(len(vals))), vals, color=_PALETTE["teal"], linewidth=1.5)
        ax.set_ylim(0, 100)
    else:
        ax.text(0.5, 0.5, "Sem dados", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])


def _draw_penalty_counts(ax, metrics_history: list) -> None:
    ax.cla()
    _style_ax(ax, "Penalizações (contagem)", "Episódio")
    a_vals = [m.get("episode_metrics", {}).get("penalty_a_count", 0) for m in metrics_history]
    b_vals = [m.get("episode_metrics", {}).get("penalty_b_count", 0) for m in metrics_history]
    if a_vals or b_vals:
        xs = list(range(max(len(a_vals), len(b_vals))))
        ax.plot(xs[:len(a_vals)], a_vals, color=_PALETTE["red"], linewidth=1.5, label="Penalty A")
        ax.plot(xs[:len(b_vals)], b_vals, color=_PALETTE["yellow"], linewidth=1.5, label="Penalty B")
        ax.legend(fontsize=6, labelcolor=_PALETTE["text"], facecolor=_PALETTE["bg_fig"])
    else:
        ax.text(0.5, 0.5, "Sem dados", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])


def _draw_penalty_frequency(ax, metrics_history: list) -> None:
    ax.cla()
    _style_ax(ax, "Penalty Frequency (%)", "Episódio")
    vals = [m.get("episode_metrics", {}).get("penalty_frequency_pct", 0.0) for m in metrics_history]
    if vals:
        ax.plot(list(range(len(vals))), vals, color=_PALETTE["peach"], linewidth=1.5)
        ax.set_ylim(0, 100)
    else:
        ax.text(0.5, 0.5, "Sem dados", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])


def _draw_action_dist_per_episode(ax, metrics_history: list) -> None:
    ax.cla()
    _style_ax(ax, "Action Dist (last episode)")
    if not metrics_history:
        ax.text(0.5, 0.5, "Sem dados", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])
        return
    dist = metrics_history[-1].get("episode_metrics", {}).get("action_dist", {})
    if not dist:
        ax.text(0.5, 0.5, "Sem dados", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])
        return
    keys = list(dist.keys())
    vals = [dist[k] for k in keys]
    colors = [_PALETTE["blue"], _PALETTE["green"], _PALETTE["red"], _PALETTE["purple"],
              _PALETTE["yellow"], _PALETTE["teal"], _PALETTE["peach"]][:len(keys)]
    ax.bar(keys, vals, color=colors[:len(keys)], edgecolor=_PALETTE["bg_fig"], width=0.6)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_ylim(0, 1.0)


def _draw_action_dist_rolling(ax, metrics_history: list) -> None:
    ax.cla()
    _style_ax(ax, "Action Dist Rolling (pie, últimos 10 ep)")
    window = metrics_history[-10:] if len(metrics_history) >= 10 else metrics_history
    if not window:
        ax.text(0.5, 0.5, "Sem dados", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])
        return
    combined: Dict[str, float] = {}
    for m in window:
        dist = m.get("episode_metrics", {}).get("action_dist", {})
        for k, v in dist.items():
            combined[k] = combined.get(k, 0.0) + float(v)
    if not combined:
        ax.text(0.5, 0.5, "Sem dados", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])
        return
    keys = list(combined.keys())
    vals = [combined[k] for k in keys]
    total = sum(vals)
    if total > 0:
        vals = [v / total for v in vals]
    pie_colors = [_PALETTE["blue"], _PALETTE["green"], _PALETTE["red"], _PALETTE["purple"],
                  _PALETTE["yellow"], _PALETTE["teal"], _PALETTE["peach"]][:len(keys)]
    ax.pie(vals, labels=keys, colors=pie_colors[:len(keys)],
           textprops={"color": _PALETTE["text"], "fontsize": 6},
           startangle=90)


def _draw_extended_resources(ax, metrics_history: list) -> None:
    ax.cla()
    _style_ax(ax, "Extended Resources (silicon/oil/water)")
    extended_keys = ["silicon", "oil", "water", "metaglass"]
    if not metrics_history:
        ax.text(0.5, 0.5, "Sem dados", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])
        return
    for key, color in zip(extended_keys, [_PALETTE["blue"], _PALETTE["peach"], _PALETTE["teal"], _PALETTE["green"]]):
        vals = [m.get("world", {}).get("resources", {}).get(key, 0.0) for m in metrics_history]
        if any(v > 0 for v in vals):
            ax.plot(list(range(len(vals))), vals, color=color, linewidth=1.2, label=key)
    ax.legend(fontsize=6, labelcolor=_PALETTE["text"], facecolor=_PALETTE["bg_fig"])


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
