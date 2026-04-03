"""
Multi-objective reward function for the Mindustry RL player agent.

================================================================================
REWARD COMPOSITION (Total Weight: ~2.88)
================================================================================

Base Rewards:
  0.30 * core_hp_delta              (defend core from damage)
  0.20 * wave_survived_bonus        (survive wave transitions)
  0.10 * resources_delta / 500      (collect resources passively)
  0.05 * drill_bonus                (passive mining gain: copper + lead + coal)
  0.15 * drill_construction_bonus   (ACTIVE: build new drills - instantaneous!)
  0.07 * power_balance_bonus        (maintain power generation)
  0.05 * build_efficiency_bonus     (construct efficiently)
  0.20 * player_alive_bonus         (keep player alive)
  0.05 * manual_mining_reward       (manual inventory collection)
  0.20 * delivery_bonus             (gradual: min(1.0, resources_delta/50); was 1.00*binary)
  0.10 * drill_on_ore_bonus         (drill placed on ore tile; uses drillsOnOre from mod v1.0.5)
  0.10 * conveyor_connectivity_bonus (new conveyor adjacent to drill)
  0.05 * time_penalty               (discourage stalling)

Inactivity Penalties (NEW):
  Scenario A: -0.05
    Condition: Last 3+ actions are WAIT(0) or MOVE(1) AND no resource progress
    Rationale: Penalize excessive passive action repetition without collecting resources

  Scenario B: -0.10
    Condition: New buildings constructed BUT resources dropped >10 units
    Rationale: Penalize wasteful builds (expensive structures without income)

Terminal Penalties:
  core destroyed        → -0.4
  player dead, core ok  → -0.5
  action failed         → -0.15

================================================================================
CURRICULUM LEARNING (ENABLED)
================================================================================

4-phase curriculum for 12 actions:
  Phase 0 (0-50k steps):     WAIT, MOVE, BUILD_DRILL (mining focus)
  Phase 1 (50k-150k steps):  Add BUILD_CONVEYOR (connect drill output)
  Phase 2 (150k-300k steps): Add defense + power + processing
  Phase 3 (300k+ steps):     All 12 actions

================================================================================
KEY DESIGN DECISIONS
================================================================================

1. Instantaneous Drill Bonus (+0.15):
   - Agent receives reward immediately when building a drill
   - Complements passive mining reward (copper collection)
   - Incentivizes action, not just outcome

2. Smart Inactivity Detection:
   - Scenario A penalizes idle patterns but allows resource-generating builds
   - Scenario B prevents wasteful construction
   - Allows strategic movement (e.g., setting up long conveyor chains)

3. Action History Tracking:
   - Ring buffer of last 10 actions enables inactivity detection
   - Resets on episode boundaries
   - Optional parameter for backward compatibility

4. Backward Compatibility:
   - All new parameters are Optional
   - Tests verify old code paths still work
   - Disabled curriculum learning has zero impact on training
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from rl.env.spaces import ACTION_REGISTRY, _action_idx

# ============================================================================
# Curriculum Learning Framework (DISABLED BY DEFAULT)
# ============================================================================
# Set CURRICULUM_ENABLED = True to restrict agent to specific actions
# during early training phases. Currently unused; prepared for future.

CURRICULUM_ENABLED = True

ACTION_WAIT = _action_idx("WAIT")
ACTION_MOVE = _action_idx("MOVE")
ACTION_BUILD_TURRET = _action_idx("BUILD_TURRET")
ACTION_BUILD_WALL = _action_idx("BUILD_WALL")
ACTION_BUILD_POWER = _action_idx("BUILD_POWER")
ACTION_BUILD_DRILL = _action_idx("BUILD_DRILL")
ACTION_REPAIR = _action_idx("REPAIR")
ACTION_BUILD_CONVEYOR = _action_idx("BUILD_CONVEYOR")
ACTION_BUILD_GRAPHITE_PRESS = _action_idx("BUILD_GRAPHITE_PRESS")
ACTION_BUILD_SILICON_SMELTER = _action_idx("BUILD_SILICON_SMELTER")
ACTION_BUILD_COMBUSTION_GEN = _action_idx("BUILD_COMBUSTION_GEN")
ACTION_BUILD_PNEUMATIC_DRILL = _action_idx("BUILD_PNEUMATIC_DRILL")

CURRICULUM_PHASES = [
    ("mining_only", (0, 50_000), [
        ACTION_WAIT, ACTION_MOVE, ACTION_BUILD_DRILL,
    ]),
    ("drill_connect", (50_000, 150_000), [
        ACTION_WAIT, ACTION_MOVE, ACTION_BUILD_DRILL, ACTION_BUILD_CONVEYOR,
    ]),
    ("defense_power", (150_000, 300_000), [
        ACTION_WAIT, ACTION_MOVE, ACTION_BUILD_DRILL, ACTION_BUILD_CONVEYOR,
        ACTION_BUILD_POWER, ACTION_BUILD_WALL, ACTION_BUILD_TURRET,
        ACTION_REPAIR, ACTION_BUILD_GRAPHITE_PRESS, ACTION_BUILD_COMBUSTION_GEN,
    ]),
    ("full", (300_000, float('inf')), list(range(12))),
]

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
        if "drill" in b.get("block", "")
    }

    curr_drills = {
        (b.get("x"), b.get("y"))
        for b in curr_state.get("buildings", [])
        if "drill" in b.get("block", "")
    }

    return len(curr_drills - prev_drills)


def _detect_action_repetition_penalty(
    action_history: Optional[list[int]],
    new_buildings: int = 0,
    min_history_len: int = 3,
) -> float:
    """
    Penalize excessive repetition of passive actions (WAIT, MOVE) without agent progress.

    Rationale: Agent should take productive actions. If it repeats WAIT/MOVE
    for multiple steps AND placed no buildings this step, it is idle.

    Note: Intentionally decoupled from resources_delta. Passive drill income
    should NOT excuse idle behavior — otherwise agent learns "build 1 drill,
    idle forever" as a stable local optimum.

    Args:
        action_history: List of last N actions (0=WAIT, 1=MOVE, 2-6=BUILD/REPAIR).
                        Should include the CURRENT action (updated before calling).
        new_buildings: Number of new buildings placed THIS step (default: 0).
                       If > 0, the agent made progress and penalty is waived.
        min_history_len: Minimum consecutive passive actions to trigger (default: 3).

    Returns:
        -0.05 if idle streak detected with no building progress, else 0.0
    """
    if action_history is None or len(action_history) < min_history_len:
        return 0.0

    recent_actions = action_history[-min_history_len:]
    all_passive = all(a in (0, 1) for a in recent_actions)

    if all_passive and new_buildings == 0:
        return -0.05

    return 0.0


def _detect_resource_bleeding_penalty(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
    bleeding_threshold: float = -10.0,
) -> float:
    """
    Penalize building structures that cost more resources than collected.

    If buildings increased BUT total resources decreased significantly,
    the agent is making expensive, unproductive builds. Penalize this behavior.

    Args:
        prev_state: Game state at t-1
        curr_state: Game state at t
        bleeding_threshold: If resources_delta < this, apply penalty (default -10)

    Returns:
        Penalty (negative) or 0.0
    """
    prev_buildings = len(prev_state.get("buildings", []))
    curr_buildings = len(curr_state.get("buildings", []))
    new_buildings = curr_buildings - prev_buildings

    def _total_resources(state: Dict[str, Any]) -> float:
        return sum(float(v) for v in state.get("resources", {}).values())

    resources_delta = _total_resources(curr_state) - _total_resources(prev_state)

    if new_buildings > 0 and resources_delta < bleeding_threshold:
        return -0.10

    return 0.0


def _compute_drill_on_ore_bonus(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
) -> float:
    prev_on_ore = int(prev_state.get("drillsOnOre", 0))
    curr_on_ore = int(curr_state.get("drillsOnOre", 0))
    new_on_ore = max(0, curr_on_ore - prev_on_ore)
    return min(1.0, float(new_on_ore))


def _compute_conveyor_connectivity_bonus(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
) -> float:
    prev_positions = {
        (b.get("x"), b.get("y"))
        for b in prev_state.get("buildings", [])
    }
    curr_buildings = curr_state.get("buildings", [])

    new_conveyors = [
        b for b in curr_buildings
        if "conveyor" in b.get("block", "")
        and (b.get("x"), b.get("y")) not in prev_positions
    ]
    if not new_conveyors:
        return 0.0

    drill_positions = {
        (b.get("x"), b.get("y"))
        for b in curr_buildings
        if "drill" in b.get("block", "")
    }
    if not drill_positions:
        return 0.0

    for conv in new_conveyors:
        cx, cy = conv.get("x", 0), conv.get("y", 0)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if (cx + dx, cy + dy) in drill_positions:
                return 1.0

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

    _DRILL_MINED = ("copper", "lead", "coal")
    mined_delta = sum(
        float(curr_state.get("resources", {}).get(r, 0.0))
        - float(prev_state.get("resources", {}).get(r, 0.0))
        for r in _DRILL_MINED
    )
    drill_bonus = min(1.0, max(0.0, mined_delta / 10.0))

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

    delivery_bonus = min(1.0, max(0.0, resources_delta / 50.0))

    drill_on_ore_bonus = _compute_drill_on_ore_bonus(prev_state, curr_state)
    conveyor_connectivity_bonus = _compute_conveyor_connectivity_bonus(prev_state, curr_state)

    inactivity_penalty_a = _detect_action_repetition_penalty(
        action_history=action_history,
        new_buildings=new_buildings,
    )

    inactivity_penalty_b = _detect_resource_bleeding_penalty(prev_state, curr_state)

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
        + 0.20 * delivery_bonus
        + 0.10 * drill_on_ore_bonus
        + 0.10 * conveyor_connectivity_bonus
        - 0.002
        + inactivity_penalty_a
        + inactivity_penalty_b
    )

    if done:
        if curr_hp <= 0.0:
            reward -= 0.4
        elif not player_alive:
            reward -= 0.5

    if bool(curr_state.get("actionFailed", False)):
        reward -= 0.15

    return float(reward)


def apply_curriculum_action_mask(
    timestep: int,
) -> list[bool]:
    from rl.env.spaces import NUM_ACTION_TYPES

    if not CURRICULUM_ENABLED:
        return [True] * NUM_ACTION_TYPES

    current_phase_actions = list(range(NUM_ACTION_TYPES))

    for _phase_name, (start, end), actions in CURRICULUM_PHASES:
        if start <= timestep < end:
            current_phase_actions = actions
            break

    mask = [False] * NUM_ACTION_TYPES
    for action_idx in current_phase_actions:
        mask[action_idx] = True

    return mask


# ============================================================================
# CURRICULUM LEARNING - HOW TO ENABLE
# ============================================================================
# To use phased action learning:
#
# 1. In this file (multi_objective.py):
#    Set CURRICULUM_ENABLED = True
#
# 2. In rl/env/mindustry_env.py step() method:
#    Call: action_mask = apply_curriculum_action_mask(self.env.num_timesteps)
#    Apply mask before environment step
#
# 3. Test convergence on early phases before enabling full action space
#
# Current phases:
#   - Phase 0 (0-50k steps): WAIT, MOVE, BUILD_DRILL only (mining focus)
#   - Phase 1 (50k-150k steps): Add BUILD_POWER (energy generation)
#   - Phase 2 (150k+ steps): All actions including BUILD_TURRET, BUILD_WALL, REPAIR
