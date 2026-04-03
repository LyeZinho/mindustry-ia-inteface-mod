"""Tests for drill detection logic in multi_objective rewards."""

import pytest
from rl.rewards.multi_objective import _detect_new_drills


def test_detect_single_new_drill():
    """Test detecting one newly built drill."""
    prev = {"buildings": [{"block": "wall", "x": 5, "y": 5}]}
    curr = {
        "buildings": [
            {"block": "wall", "x": 5, "y": 5},
            {"block": "drill", "x": 10, "y": 10},
        ]
    }
    assert _detect_new_drills(prev, curr) == 1


def test_detect_multiple_new_drills():
    """Test detecting multiple newly built drills."""
    prev = {"buildings": []}
    curr = {
        "buildings": [
            {"block": "drill", "x": 10, "y": 10},
            {"block": "drill", "x": 11, "y": 11},
            {"block": "wall", "x": 5, "y": 5},
        ]
    }
    assert _detect_new_drills(prev, curr) == 2


def test_drill_not_counted_if_already_present():
    """Test that existing drills are not counted as new."""
    prev = {"buildings": [{"block": "drill", "x": 10, "y": 10}]}
    curr = {"buildings": [{"block": "drill", "x": 10, "y": 10}]}
    assert _detect_new_drills(prev, curr) == 0


def test_no_drills_returns_zero():
    """Test empty building list returns 0 new drills."""
    prev = {"buildings": []}
    curr = {"buildings": [{"block": "wall", "x": 5, "y": 5}]}
    assert _detect_new_drills(prev, curr) == 0
