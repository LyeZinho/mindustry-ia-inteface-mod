"""
Multi-objective reward function for the Mindustry RL player agent.

reward = 0.30 * core_hp_delta
       + 0.20 * wave_survived_bonus
       + 0.10 * resources_delta / 500
       + 0.08 * drill_bonus          (continuous: copper_delta / 10.0, clamped [0,1])
       + 0.07 * power_balance_bonus
       + 0.05 * build_efficiency_bonus
       + 0.20 * player_alive_bonus
       - 0.002                       (time penalty)

Terminal penalties:
  core destroyed        → -0.4
  player dead, core ok  → -0.5
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
        return sum(float(v) for v in state.get("resources", {}).values())

    resources_delta = _total_resources(curr_state) - _total_resources(prev_state)

    prev_copper = float(prev_state.get("resources", {}).get("copper", 0.0))
    curr_copper = float(curr_state.get("resources", {}).get("copper", 0.0))
    copper_delta = curr_copper - prev_copper
    drill_bonus = min(1.0, max(0.0, copper_delta / 10.0))

    power = curr_state.get("power", {})
    produced = float(power.get("produced", 0.0))
    consumed = float(power.get("consumed", 0.0))
    if produced > 0:
        power_balance_bonus = max(0.0, min(1.0, (produced - consumed) / produced))
    else:
        power_balance_bonus = 0.0

    prev_buildings = len(prev_state.get("buildings", []))
    curr_buildings = len(curr_state.get("buildings", []))
    new_buildings = max(0, curr_buildings - prev_buildings)
    build_efficiency_bonus = min(1.0, new_buildings * 0.1)

    player_alive = bool(curr_state.get("player", {}).get("alive", False))
    core_destroyed = curr_hp <= 0.0
    player_alive_bonus = 1.0 if (player_alive and not core_destroyed) else 0.0

    reward = (
        0.30 * core_hp_delta
        + 0.20 * wave_survived_bonus
        + 0.10 * (resources_delta / 500.0)
        + 0.08 * drill_bonus
        + 0.07 * power_balance_bonus
        + 0.05 * build_efficiency_bonus
        + 0.20 * player_alive_bonus
        - 0.002
    )

    if done:
        if curr_hp <= 0.0:
            reward -= 0.4
        elif not player_alive:
            reward -= 0.5

    return float(reward)
