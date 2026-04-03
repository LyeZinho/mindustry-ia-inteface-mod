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
    curr = make_state(core={"hp": 0.4})
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
    """Core destroyed applies -1.0 terminal penalty."""
    prev = make_state(core={"hp": 0.1})
    curr = make_state(core={"hp": 0.0})
    r = compute_reward(prev, curr, done=True)
    assert r <= -1.0


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
    # Should include +0.10 player alive bonus minus time penalty → positive
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
