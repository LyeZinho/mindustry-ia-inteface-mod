"""Tests for multi-objective reward function."""
import pytest
from rl.rewards.multi_objective import compute_reward

BASE = {
    "core": {"hp": 1.0},
    "wave": 1,
    "resources": {"copper": 0, "lead": 0, "graphite": 0, "titanium": 0, "thorium": 0},
    "friendlyUnits": [],
    "enemies": [],
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
    """Losing core HP is penalized."""
    prev = make_state(core={"hp": 1.0})
    curr = make_state(core={"hp": 0.8})
    r = compute_reward(prev, curr, done=False)
    assert r < 0
    assert r == pytest.approx(-0.101, abs=1e-5)


def test_reward_core_hp_gain():
    """Maintaining full HP gives a small positive contribution."""
    prev = make_state(core={"hp": 0.9})
    curr = make_state(core={"hp": 0.9})
    r = compute_reward(prev, curr, done=False)
    # time penalty applies, so result is slightly negative
    assert r == pytest.approx(-0.001, abs=1e-4)


def test_reward_wave_survived():
    """Completing a wave gives +0.20 bonus."""
    prev = make_state(wave=1)
    curr = make_state(wave=2)
    r = compute_reward(prev, curr, done=False)
    assert r > 0
    assert r == pytest.approx(0.199, abs=1e-5)


def test_reward_resource_accumulation():
    """Accumulating resources gives a small positive reward."""
    prev = make_state(resources={"copper": 0, "lead": 0, "graphite": 0, "titanium": 0, "thorium": 0})
    curr = make_state(resources={"copper": 500, "lead": 0, "graphite": 0, "titanium": 0, "thorium": 0})
    r = compute_reward(prev, curr, done=False)
    assert r > 0
    assert r == pytest.approx(0.149, abs=1e-5)


def test_reward_terminal_penalty():
    """Core destroyed applies -1.0 terminal penalty."""
    prev = make_state(core={"hp": 0.1})
    curr = make_state(core={"hp": 0.0})
    r = compute_reward(prev, curr, done=True)
    # -1.0 terminal + core_hp_delta penalty
    assert r <= -1.0
    assert r == pytest.approx(-1.051, abs=1e-5)


def test_reward_done_without_core_destroyed():
    """done=True with core alive (truncation) does not apply -1.0 terminal penalty."""
    prev = make_state(core={"hp": 1.0})
    curr = make_state(core={"hp": 1.0})
    r = compute_reward(prev, curr, done=True)
    assert r == pytest.approx(-0.001, abs=1e-4)


def test_reward_friendly_units_ratio():
    """More friendly units alive → higher reward."""
    prev_state = make_state()
    curr_no_units = make_state(friendlyUnits=[], enemies=[{"hp": 1.0}])
    curr_with_units = make_state(friendlyUnits=[{"hp": 1.0}, {"hp": 1.0}], enemies=[{"hp": 1.0}])
    r_no = compute_reward(prev_state, curr_no_units, done=False)
    r_with = compute_reward(prev_state, curr_with_units, done=False)
    assert r_with > r_no
