"""
Multi-objective reward function for the Mindustry RL player agent.

reward = 0.30 * core_hp_delta
       + 0.20 * wave_survived_bonus
       + 0.10 * resources_delta / 500
       + 0.08 * drill_bonus          (continuous: copper_delta / 10.0, clamped [0,1])
       + 0.07 * power_balance_bonus
       + 0.05 * build_efficiency_bonus
       + 0.20 * player_alive_bonus
       + 0.05 * manual_mining_reward (mining delta > 0)
       + 1.00 * delivery_bonus       (items delivered to core)
       - 0.002                       (time penalty)

Terminal penalties:
  core destroyed        → -0.4
  player dead, core ok  → -0.5
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# ------------------------------------------------------------------ #
# STATE STRUCTURE & ASSUMPTIONS FOR UPCOMING REWARD FEATURES          #
# (drill bonus, action repetition penalty, resource bleeding penalty) #
# ------------------------------------------------------------------ #
#
# == State dict keys currently accessed by compute_reward() ==
#
#   prev_state / curr_state share the same structure (raw Mimi Gateway JSON):
#
#   "core"           : dict  {"hp": float(0.0-1.0), "x": int, "y": int, "size": int}
#   "wave"           : int   (current wave number)
#   "resources"      : dict  {str → float}  e.g. {"copper": 450.0, "lead": 120.0, ...}
#   "power"          : dict  {"produced": float, "consumed": float, "stored": float, "capacity": float}
#   "buildings"      : list[dict]  each entry: {"block": str, "team": str, "x": int, "y": int,
#                                                "hp": float, "rotation": int}
#   "player"         : dict  {"x": int, "y": int, "alive": bool, "hp": float}
#   "inventory"      : dict  {str → float}  (items carried by player, NOT core resources)
#   "actionFailed"   : bool  (True if last command was rejected by the game)
#
#   Additional keys present in state but NOT used here (used in spaces.py):
#     "grid"           : list[dict]  — sparse tile grid (31×31, often empty)
#     "enemies"        : list[dict]  — nearby enemy units
#     "friendlyUnits"  : list[dict]  — nearby friendly units
#     "waveTime"       : int         — ticks until next wave
#     "nearbyOres"     : list[dict]  — sparse ore features
#     "nearbyEnemies"  : list[dict]  — sparse enemy features
#     "tick"           : int         — Unix timestamp
#     "time"           : int         — game ticks since start
#
# == FEATURE 1: Drill placement bonus (+0.15) ==
#
#   Detection method: Compare prev_state["buildings"] vs curr_state["buildings"].
#   A "new drill" is a building entry where entry["block"] contains "drill"
#   (matches "mechanical-drill", "pneumatic-drill", etc.) that exists in
#   curr_state["buildings"] but not in prev_state["buildings"].
#   Comparison key: (entry["x"], entry["y"]) since coordinates are unique.
#
#   Data types:  buildings is list[dict]; block is str; x,y are int.
#
# == FEATURE 2: Resource bleeding penalty (-0.10) ==
#
#   Detection method: Sum all values in state["resources"] dict.
#   "Bleeding" = total resources decreased AND a build action was taken
#   (i.e., the agent built something expensive that cost more than income).
#   Formula: if resources_delta < 0 AND new_buildings > 0 → apply penalty.
#   Note: resources dict values are float (copper, lead, graphite, titanium, thorium).
#
# == FEATURE 3: Action repetition penalty (-0.05) ==
#
#   Action history is NOT currently tracked in state.
#   The state dict from Mimi Gateway has no "lastAction" or "actions_history" field.
#   MindustryEnv.step() receives action as np.ndarray([action_type, arg]) but does
#   not store it in state or pass it to compute_reward().
#
#   Action types (from spaces.py, no named constants exist yet):
#     0 = WAIT, 1 = MOVE, 2 = BUILD_TURRET, 3 = BUILD_WALL,
#     4 = BUILD_POWER, 5 = BUILD_DRILL, 6 = REPAIR
#
#   → Action history will need to be added to MindustryEnv (Task 2):
#     Option A: Pass action_taken as extra arg to compute_reward()
#     Option B: Inject "lastAction" / "actionHistory" into state dict before calling compute_reward()
#     Option C: Track in a reward wrapper class with internal ring buffer
#
# ------------------------------------------------------------------ #


def _detect_new_drills(prev_state: Dict[str, Any], curr_state: Dict[str, Any]) -> int:
    """Count newly constructed drills by comparing building lists."""
    prev_drills = {
        (b.get("x"), b.get("y"))
        for b in prev_state.get("buildings", [])
        if b.get("block") == "drill"
    }

    curr_drills = {
        (b.get("x"), b.get("y"))
        for b in curr_state.get("buildings", [])
        if b.get("block") == "drill"
    }

    return len(curr_drills - prev_drills)


def _detect_action_repetition_penalty(
    action_history: Optional[list[int]],
    resources_delta: float,
    min_history_len: int = 3,
    no_progress_threshold: float = 0.0,
) -> float:
    """
    Penalize excessive repetition of passive actions (WAIT, MOVE) without progress.

    Rationale: Agent should explore different actions. If it repeats WAIT/MOVE
    for multiple steps AND collects no resources, it's being idle/unproductive.

    Args:
        action_history: List of last N actions (0=WAIT, 1=MOVE, 2-6=BUILD/REPAIR)
        resources_delta: Total resources gained this step
        min_history_len: Minimum consecutive actions to trigger penalty (default: 3)
        no_progress_threshold: If resources_delta <= this, consider no progress (default: 0)

    Returns:
        Penalty (negative) or 0.0 if no violation
    """
    if action_history is None or len(action_history) < min_history_len:
        return 0.0

    recent_actions = action_history[-min_history_len:]
    all_passive = all(a in (0, 1) for a in recent_actions)

    if all_passive and resources_delta <= no_progress_threshold:
        return -0.05

    return 0.0


def compute_reward(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
    done: bool,
    action_taken: Optional[int] = None,
    action_history: Optional[list[int]] = None,
) -> float:
    """
    Compute multi-objective reward.

    Args:
        prev_state: Game state at t-1
        curr_state: Game state at t
        done: Episode terminal flag
        action_taken: Current action type (0-6, see spaces.py ACTION enum)
        action_history: List of last N actions (default: None, skip inactivity checks)

    Returns:
        Scalar reward for this transition
    """
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

    new_drills = _detect_new_drills(prev_state, curr_state)
    drill_construction_bonus = min(1.0, new_drills * 0.15)

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

    def _total_inventory(state: Dict[str, Any]) -> float:
        return sum(float(v) for v in state.get("inventory", {}).values())

    inventory_delta = _total_inventory(curr_state) - _total_inventory(prev_state)
    manual_mining_reward = max(0.0, inventory_delta * 0.1)

    delivery_bonus = 1.0 if resources_delta > 0 and inventory_delta == 0 else 0.0

    inactivity_penalty_a = _detect_action_repetition_penalty(
        action_history=action_history,
        resources_delta=resources_delta,
    )

    reward = (
        0.30 * core_hp_delta
        + 0.20 * wave_survived_bonus
        + 0.10 * (resources_delta / 500.0)
        + 0.05 * drill_bonus
        + 0.15 * drill_construction_bonus
        + 0.07 * power_balance_bonus
        + 0.05 * build_efficiency_bonus
        + 0.20 * player_alive_bonus
        + 0.05 * manual_mining_reward
        + 1.00 * delivery_bonus
        - 0.002
        + inactivity_penalty_a
    )

    if done:
        if curr_hp <= 0.0:
            reward -= 0.4
        elif not player_alive:
            reward -= 0.5

    if bool(curr_state.get("actionFailed", False)):
        reward -= 0.15

    return float(reward)
