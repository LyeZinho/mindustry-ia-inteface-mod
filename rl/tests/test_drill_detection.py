"""Tests for drill detection logic in multi_objective rewards."""
import pytest
from rl.rewards.multi_objective import _detect_new_drills


def test_detect_mechanical_drill():
    """Detect newly built mechanical-drill."""
    prev = {"buildings": [{"block": "wall", "x": 5, "y": 5}]}
    curr = {
        "buildings": [
            {"block": "wall", "x": 5, "y": 5},
            {"block": "mechanical-drill", "x": 10, "y": 10},
        ]
    }
    assert _detect_new_drills(prev, curr) == 1


def test_detect_pneumatic_drill():
    """Detect newly built pneumatic-drill."""
    prev = {"buildings": []}
    curr = {"buildings": [{"block": "pneumatic-drill", "x": 3, "y": 3}]}
    assert _detect_new_drills(prev, curr) == 1


def test_detect_electric_drill():
    """Detect newly built electric-drill."""
    prev = {"buildings": []}
    curr = {"buildings": [{"block": "electric-drill", "x": 7, "y": 7}]}
    assert _detect_new_drills(prev, curr) == 1


def test_detect_multiple_drill_types():
    """Detect multiple drills of different types."""
    prev = {"buildings": []}
    curr = {
        "buildings": [
            {"block": "mechanical-drill", "x": 10, "y": 10},
            {"block": "pneumatic-drill", "x": 11, "y": 11},
            {"block": "wall", "x": 5, "y": 5},
        ]
    }
    assert _detect_new_drills(prev, curr) == 2


def test_existing_drill_not_counted():
    """Existing drills are not counted as new."""
    prev = {"buildings": [{"block": "mechanical-drill", "x": 10, "y": 10}]}
    curr = {"buildings": [{"block": "mechanical-drill", "x": 10, "y": 10}]}
    assert _detect_new_drills(prev, curr) == 0


def test_non_drill_buildings_not_counted():
    """Non-drill buildings (walls, turrets) are not counted."""
    prev = {"buildings": []}
    curr = {"buildings": [{"block": "wall", "x": 5, "y": 5}]}
    assert _detect_new_drills(prev, curr) == 0


def test_empty_buildings_returns_zero():
    """Empty building lists return 0."""
    prev = {"buildings": []}
    curr = {"buildings": []}
    assert _detect_new_drills(prev, curr) == 0
