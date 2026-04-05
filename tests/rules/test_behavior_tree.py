import pytest
from mindustry_ai.rules.behavior_tree import BehaviorTree, Action


def test_behavior_tree_init():
    bt = BehaviorTree()
    assert bt is not None


def test_get_feasible_actions_returns_list():
    bt = BehaviorTree()
    state = {
        "resources": {"copper": 100, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 500, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 0, "wave_number": 1, "time_to_wave": 600},
        "infrastructure": {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    feasible = bt.get_feasible_actions(state)
    
    assert isinstance(feasible, list)
    assert len(feasible) > 0


def test_threat_detection():
    bt = BehaviorTree()
    state_safe = {
        "resources": {"copper": 100, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 500, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 0, "wave_number": 1, "time_to_wave": 600},
        "infrastructure": {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    state_threat = {
        **state_safe,
        "threat": {"enemies_nearby": 5, "wave_number": 2, "time_to_wave": 10},
    }
    
    feasible_safe = bt.get_feasible_actions(state_safe)
    feasible_threat = bt.get_feasible_actions(state_threat)
    
    assert len(feasible_threat) > 0
    assert Action.PLACE_TURRET in feasible_threat
