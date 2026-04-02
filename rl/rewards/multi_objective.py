"""
Multi-objective reward function for the Mindustry RL agent.

reward = 0.50 * core_hp_delta
       + 0.20 * wave_survived_bonus
       + 0.15 * resources_delta / 500
       + 0.15 * friendly_units_ratio
       - 0.001  (time penalty)

Terminal: -1.0 if core_destroyed.
"""
from __future__ import annotations

from typing import Any, Dict


def compute_reward(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
    done: bool,
) -> float:
    prev_hp = float(prev_state.get("core", {}).get("hp", 0.0))
    curr_hp = float(curr_state.get("core", {}).get("hp", 0.0))
    core_hp_delta = curr_hp - prev_hp

    prev_wave = int(prev_state.get("wave", 0))
    curr_wave = int(curr_state.get("wave", 0))
    wave_survived_bonus = 1.0 if curr_wave > prev_wave else 0.0

    def _total_resources(state: Dict[str, Any]) -> float:
        res = state.get("resources", {})
        return sum(float(v) for v in res.values())

    resources_delta = _total_resources(curr_state) - _total_resources(prev_state)

    friendly = curr_state.get("friendlyUnits", [])
    enemies = curr_state.get("enemies", [])
    total_units = len(friendly) + len(enemies)
    friendly_ratio = len(friendly) / total_units if total_units > 0 else 0.0

    reward = (
        0.50 * core_hp_delta
        + 0.20 * wave_survived_bonus
        + 0.15 * (resources_delta / 500.0)
        + 0.15 * friendly_ratio
        - 0.001
    )

    if done and curr_hp <= 0.0:
        reward -= 1.0

    return float(reward)
