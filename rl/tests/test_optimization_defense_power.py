import pytest
from rl.optimization.defense import compute_defense_gap
from rl.optimization.power import compute_power_deficit

BASE_STATE = {
    "core": {"x": 15, "y": 15, "size": 3, "hp": 1.0},
    "buildings": [],
    "power": {"produced": 120.0, "consumed": 80.0, "stored": 500, "capacity": 1000},
}

def test_defense_gap_no_turrets():
    gap = compute_defense_gap(BASE_STATE)
    assert gap == pytest.approx(1.0, abs=0.01)

def test_defense_gap_range():
    gap = compute_defense_gap(BASE_STATE)
    assert 0.0 <= gap <= 1.0

def test_power_deficit_surplus():
    deficit = compute_power_deficit(BASE_STATE)
    assert deficit == pytest.approx(0.0)

def test_power_deficit_shortage():
    state = {**BASE_STATE, "power": {"produced": 50.0, "consumed": 100.0, "stored": 0, "capacity": 1000}}
    deficit = compute_power_deficit(state)
    assert deficit > 0.0
    assert deficit <= 1.0
