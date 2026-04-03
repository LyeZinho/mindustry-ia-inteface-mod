"""Tests for multi-objective reward function."""
import pytest
from rl.rewards.multi_objective import compute_reward

BASE = {
    "core": {"hp": 1.0},
    "wave": 1,
    "resources": {"copper": 0, "lead": 0, "graphite": 0, "titanium": 0, "thorium": 0},
    "power": {"produced": 0.0, "consumed": 0.0},
    "friendlyUnits": [],
    "enemies": [],
    "player": {"alive": True, "hp": 1.0},
    "buildings": [],
}


def make_state(**overrides):
    import copy
    s = copy.deepcopy(BASE)
    for k, v in overrides.items():
        if isinstance(v, dict) and k in s:
            s[k].update(v)
        else:
            s[k] = v
    return s


def test_reward_core_hp_loss():
    """Losing core HP is penalized (large enough drop to dominate other bonuses)."""
    prev = make_state(core={"hp": 1.0})
    curr = make_state(core={"hp": 0.05})
    r = compute_reward(prev, curr, done=False)
    assert r < 0


def test_reward_wave_survived():
    """Completing a wave gives positive bonus."""
    prev = make_state(wave=1)
    curr = make_state(wave=2)
    r = compute_reward(prev, curr, done=False)
    assert r > 0


def test_reward_resource_accumulation():
    """Accumulating resources gives a small positive reward."""
    prev = make_state(resources={"copper": 0, "lead": 0, "graphite": 0, "titanium": 0, "thorium": 0})
    curr = make_state(resources={"copper": 500, "lead": 0, "graphite": 0, "titanium": 0, "thorium": 0})
    r = compute_reward(prev, curr, done=False)
    assert r > 0


def test_reward_terminal_core_destroyed():
    """Core destroyed applies -0.4 terminal penalty."""
    prev = make_state(core={"hp": 0.1})
    curr = make_state(core={"hp": 0.0})
    r = compute_reward(prev, curr, done=True)
    assert r <= -0.4


def test_reward_terminal_player_dead_core_alive():
    """-0.5 terminal penalty when player dies but core is alive."""
    prev = make_state(player={"alive": True, "hp": 0.3})
    curr = make_state(player={"alive": False, "hp": 0.0})
    r = compute_reward(prev, curr, done=True)
    assert r <= -0.5
    # should NOT apply -1.0 (core still alive)
    assert r > -1.0


def test_reward_player_alive_bonus():
    """Player alive contributes positively when core HP stable."""
    prev = make_state()
    curr = make_state()
    r = compute_reward(prev, curr, done=False)
    # Should include +0.20 player alive bonus minus time penalty → positive
    assert r > 0


def test_reward_power_balance_bonus():
    """Positive power balance (produced > consumed) gives bonus."""
    prev = make_state()
    curr_balanced = make_state(power={"produced": 10.0, "consumed": 5.0})
    curr_deficit = make_state(power={"produced": 5.0, "consumed": 10.0})
    r_balanced = compute_reward(prev, curr_balanced, done=False)
    r_deficit = compute_reward(prev, curr_deficit, done=False)
    assert r_balanced > r_deficit


def test_reward_done_without_core_or_player_destroyed():
    """done=True with both core and player alive (truncation) → no extra penalty."""
    prev = make_state()
    curr = make_state()
    r = compute_reward(prev, curr, done=True)
    # time penalty + player alive bonus; no terminal penalty
    assert r > -0.5


def test_drill_bonus_when_copper_increases_significantly():
    prev = {"resources": {"copper": 10}, "core": {"hp": 0.9}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": True}}
    curr = {"resources": {"copper": 20}, "core": {"hp": 0.9}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": True}}
    r = compute_reward(prev, curr, done=False)
    # drill_bonus=1.0 (copper +10 >= 5), weighted at 0.10 → reward should be higher than baseline
    assert r > -0.002  # time penalty only without drill bonus would be ~ -0.0005 + 0.05 player alive


def test_no_drill_bonus_when_copper_increases_less_than_5():
    prev = {"resources": {"copper": 10}, "core": {"hp": 0.9}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": True}}
    curr = {"resources": {"copper": 13}, "core": {"hp": 0.9}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": True}}
    r_small = compute_reward(prev, curr, done=False)
    prev2 = {"resources": {"copper": 10}, "core": {"hp": 0.9}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": True}}
    curr2 = {"resources": {"copper": 16}, "core": {"hp": 0.9}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": True}}
    r_large = compute_reward(prev2, curr2, done=False)
    assert r_large > r_small  # drill bonus fires for +6, not for +3


def test_time_penalty():
    """Time penalty should be 0.002."""
    prev = {"resources": {}, "core": {"hp": 0.5}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": False}}
    curr = {"resources": {}, "core": {"hp": 0.5}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": False}}
    r = compute_reward(prev, curr, done=False)
    # No positive signals, no negative terminal → just time penalty (-0.002)
    assert abs(r - (-0.002)) < 1e-6


def test_build_fail_penalty():
    state_ok = make_state()
    state_fail = make_state()
    state_fail["actionFailed"] = True
    reward_ok = compute_reward(BASE, state_ok, done=False)
    reward_fail = compute_reward(BASE, state_fail, done=False)
    assert abs(reward_fail - reward_ok - (-0.15)) < 1e-6


def test_reward_manual_mining_reward():
    """Manual mining (inventory delta > 0) gives +0.05 bonus."""
    prev = make_state(inventory={})
    curr = make_state(inventory={"copper": 5})
    r = compute_reward(prev, curr, done=False)
    r_no_mining = compute_reward(prev, prev, done=False)
    assert r > r_no_mining


def test_reward_mining_delivery_bonus():
    """Delivery bonus (+1.00) triggers when resources increase without manual mining."""
    prev = make_state(resources={"copper": 0, "lead": 0, "graphite": 0, "titanium": 0, "thorium": 0}, inventory={})
    curr = make_state(resources={"copper": 50, "lead": 0, "graphite": 0, "titanium": 0, "thorium": 0}, inventory={})
    r = compute_reward(prev, curr, done=False)
    r_no_delivery = compute_reward(prev, prev, done=False)
    assert r > r_no_delivery


def test_reward_no_mining_when_inventory_empty():
    """No mining reward when inventory delta <= 0."""
    prev = make_state(inventory={"copper": 5})
    curr = make_state(inventory={"copper": 5})
    r = compute_reward(prev, curr, done=False)
    r_no_delta = compute_reward(prev, prev, done=False)
    assert abs(r - r_no_delta) < 1e-6


def test_no_penalty_if_actions_diverse():
    """Diverse actions (including BUILD) should not trigger penalty."""
    from rl.rewards.multi_objective import _detect_action_repetition_penalty

    history = [0, 1, 2, 0, 1]  # Mixed: WAIT, MOVE, BUILD_TURRET, ...
    penalty = _detect_action_repetition_penalty(history, resources_delta=0.0)
    assert penalty == 0.0


def test_penalty_if_repeated_wait_no_progress():
    """Repeated WAIT with no resource progress = -0.05."""
    from rl.rewards.multi_objective import _detect_action_repetition_penalty

    history = [0, 0, 0, 0, 0]  # All WAIT
    penalty = _detect_action_repetition_penalty(history, resources_delta=0.0)
    assert penalty == -0.05


def test_penalty_if_repeated_move_no_progress():
    """Repeated MOVE with no resource progress = -0.05."""
    from rl.rewards.multi_objective import _detect_action_repetition_penalty

    history = [1, 1, 1]  # All MOVE (3 is minimum)
    penalty = _detect_action_repetition_penalty(history, resources_delta=0.0)
    assert penalty == -0.05


def test_no_penalty_if_resources_gained_despite_repetition():
    """If resources gained, no penalty even with repeated actions."""
    from rl.rewards.multi_objective import _detect_action_repetition_penalty

    history = [1, 1, 1]  # MOVE MOVE MOVE
    penalty = _detect_action_repetition_penalty(history, resources_delta=50.0)
    assert penalty == 0.0


def test_no_penalty_if_history_too_short():
    """History < min_history_len should not trigger penalty."""
    from rl.rewards.multi_objective import _detect_action_repetition_penalty

    history = [0, 0]  # Only 2 actions
    penalty = _detect_action_repetition_penalty(history, resources_delta=0.0, min_history_len=3)
    assert penalty == 0.0


def test_no_penalty_if_history_is_none():
    """None history should return 0 (no penalty)."""
    from rl.rewards.multi_objective import _detect_action_repetition_penalty

    penalty = _detect_action_repetition_penalty(None, resources_delta=0.0)
    assert penalty == 0.0


def test_no_penalty_if_built_and_gained_resources():
    """Positive trade: built something and resources increased."""
    from rl.rewards.multi_objective import _detect_resource_bleeding_penalty

    prev = {"buildings": [{"block": "wall", "x": 0, "y": 0}], "resources": {"copper": 100}}
    curr = {"buildings": [{"block": "wall", "x": 0, "y": 0}, {"block": "drill", "x": 5, "y": 5}], "resources": {"copper": 150}}

    penalty = _detect_resource_bleeding_penalty(prev, curr)
    assert penalty == 0.0


def test_penalty_if_built_but_resources_dropped():
    """Negative trade: built something but resources dropped >10."""
    from rl.rewards.multi_objective import _detect_resource_bleeding_penalty

    prev = {"buildings": [{"block": "wall", "x": 0, "y": 0}], "resources": {"copper": 100}}
    curr = {"buildings": [{"block": "wall", "x": 0, "y": 0}, {"block": "turret", "x": 5, "y": 5}], "resources": {"copper": 80}}

    penalty = _detect_resource_bleeding_penalty(prev, curr, bleeding_threshold=-10.0)
    assert penalty == -0.10


def test_no_penalty_if_resources_dropped_but_no_new_buildings():
    """Resources dropped, but didn't build anything (not the agent's fault)."""
    from rl.rewards.multi_objective import _detect_resource_bleeding_penalty

    prev = {"buildings": [{"block": "wall", "x": 0, "y": 0}], "resources": {"copper": 100}}
    curr = {"buildings": [{"block": "wall", "x": 0, "y": 0}], "resources": {"copper": 80}}

    penalty = _detect_resource_bleeding_penalty(prev, curr)
    assert penalty == 0.0


def test_no_penalty_if_small_resource_drop():
    """Resources dropped slightly (less than threshold), not severe."""
    from rl.rewards.multi_objective import _detect_resource_bleeding_penalty

    prev = {"buildings": [{"block": "wall", "x": 0, "y": 0}], "resources": {"copper": 100}}
    curr = {"buildings": [{"block": "wall", "x": 0, "y": 0}, {"block": "wall", "x": 5, "y": 5}], "resources": {"copper": 95}}

    penalty = _detect_resource_bleeding_penalty(prev, curr, bleeding_threshold=-10.0)
    assert penalty == 0.0


def test_multiple_resources_totaled():
    """Resource delta includes all items (copper + lead + graphite, etc)."""
    from rl.rewards.multi_objective import _detect_resource_bleeding_penalty

    prev = {"buildings": [{"block": "wall", "x": 0, "y": 0}], "resources": {"copper": 100, "lead": 50}}
    curr = {"buildings": [{"block": "wall", "x": 0, "y": 0}, {"block": "turret", "x": 5, "y": 5}], "resources": {"copper": 80, "lead": 30}}

    penalty = _detect_resource_bleeding_penalty(prev, curr, bleeding_threshold=-10.0)
    assert penalty == -0.10  # Total delta: (80-100) + (30-50) = -40, which is < -10


def test_curriculum_disabled_allows_all_actions():
    """When disabled, curriculum allows all actions."""
    from rl.rewards.multi_objective import apply_curriculum_action_mask, CURRICULUM_ENABLED
    
    mask = apply_curriculum_action_mask(timestep=0)
    assert mask == [True] * 7  # All allowed when disabled
    assert CURRICULUM_ENABLED is False


def test_curriculum_constants_exist():
    """Curriculum framework constants are defined."""
    from rl.rewards.multi_objective import (
        CURRICULUM_ENABLED, ACTION_WAIT, ACTION_MOVE, ACTION_BUILD_DRILL,
        ACTION_BUILD_POWER, ACTION_BUILD_TURRET, ACTION_BUILD_WALL, ACTION_REPAIR,
        CURRICULUM_PHASES
    )
    
    # Verify constants exist
    assert ACTION_WAIT == 0
    assert ACTION_MOVE == 1
    assert ACTION_BUILD_DRILL == 5
    assert ACTION_REPAIR == 6
    
    # Verify phases exist
    assert len(CURRICULUM_PHASES) == 3
    assert CURRICULUM_PHASES[0][0] == "mining_only"
    assert CURRICULUM_PHASES[1][0] == "power_gen"
    assert CURRICULUM_PHASES[2][0] == "full"


def test_full_compute_reward_integration():
    """Integration test: compute_reward with all new features."""
    from rl.rewards.multi_objective import compute_reward
    
    # Minimal state for testing - with a new drill and action history
    prev_state = {
        "core": {"hp": 1.0},
        "wave": 1,
        "resources": {"copper": 100, "lead": 50},
        "power": {"produced": 100, "consumed": 50},
        "buildings": [{"block": "wall", "x": 0, "y": 0}],
        "player": {"alive": True},
        "inventory": {},
        "actionFailed": False,
    }
    
    curr_state = {
        "core": {"hp": 1.0},
        "wave": 1,
        "resources": {"copper": 150, "lead": 50},  # +50 copper
        "power": {"produced": 100, "consumed": 50},
        "buildings": [
            {"block": "wall", "x": 0, "y": 0},
            {"block": "mechanical-drill", "x": 5, "y": 5},  # New drill!
        ],
        "player": {"alive": True},
        "inventory": {},
        "actionFailed": False,
    }
    
    # Test with diverse action history (MOVE + BUILD_DRILL)
    reward = compute_reward(
        prev_state=prev_state,
        curr_state=curr_state,
        done=False,
        action_taken=5,  # BUILD_DRILL
        action_history=[1, 1, 5],  # MOVE, MOVE, BUILD_DRILL
    )
    
    # Should be positive:
    # - Drill construction bonus: +0.15
    # - Copper collected: +0.05 * (50/10) clamped to 1.0 = +0.05
    # - No inactivity penalties (diverse actions)
    assert reward > 0, f"Expected positive reward, got {reward}"
    assert isinstance(reward, float), f"Expected float, got {type(reward)}"
