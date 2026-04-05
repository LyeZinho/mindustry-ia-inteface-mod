import pytest
from mindustry_ai.rules.hybrid_decider import HybridDecider


def test_hybrid_decider_decides_action():
    decider = HybridDecider()
    state = {
        "resources": {"copper": 100, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 500, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 0, "wave_number": 1, "time_to_wave": 600},
        "infrastructure": {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    
    action = decider.decide(state)
    assert isinstance(action, int)
    assert 0 <= action <= 6


def test_hybrid_decider_prioritizes_defense():
    decider = HybridDecider()
    state = {
        "resources": {"copper": 100, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 500, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 5, "wave_number": 2, "time_to_wave": 10},
        "infrastructure": {"drills_count": 2, "turrets_count": 0, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    
    action = decider.decide(state)
    assert action == 3
