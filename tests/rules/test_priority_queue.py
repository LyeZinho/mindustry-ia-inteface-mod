import pytest
from mindustry_ai.rules.priority_queue import PriorityQueue


def test_priority_queue_init():
    pq = PriorityQueue()
    assert pq is not None


def test_compute_priorities_survival_spike():
    pq = PriorityQueue()
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
    
    priorities_safe = pq.compute_priorities(state_safe)
    priorities_threat = pq.compute_priorities(state_threat)
    
    assert priorities_threat["survive"] > priorities_safe["survive"]


def test_get_highest_priority_action():
    pq = PriorityQueue()
    state = {
        "resources": {"copper": 10, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 900, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 0, "wave_number": 1, "time_to_wave": 600},
        "infrastructure": {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    
    priorities = pq.compute_priorities(state)
    action = pq.get_highest_priority_category(priorities)
    
    assert action == "mining"
