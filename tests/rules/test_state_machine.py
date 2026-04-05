import pytest
from mindustry_ai.rules.state_machine import StateMachine, GameState


def test_state_machine_init():
    sm = StateMachine()
    assert sm.current_state == GameState.MINING


def test_transition_mining_to_crafting():
    sm = StateMachine()
    state = {
        "resources": {"copper": 300, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 500, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 0, "wave_number": 1, "time_to_wave": 600},
        "infrastructure": {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    sm.update(state)
    assert sm.current_state == GameState.CRAFTING


def test_transition_to_defense():
    sm = StateMachine()
    state = {
        "resources": {"copper": 100, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 500, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 5, "wave_number": 2, "time_to_wave": 10},
        "infrastructure": {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    sm.update(state)
    assert sm.current_state == GameState.DEFENSE
