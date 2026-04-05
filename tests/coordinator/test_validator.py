import pytest
from dataclasses import dataclass
from mindustry_ai.coordinator.action_queue import Action
from mindustry_ai.coordinator.validator import ActionValidator


@dataclass
class GameState:
    structures: dict
    resources: dict
    map_width: int
    map_height: int


class TestActionValidator:
    def test_validate_action_on_empty_cell(self):
        validator = ActionValidator()
        state = GameState(structures={}, resources={"copper": 100}, map_width=20, map_height=20)
        action = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        
        valid, reason = validator.validate(action, state)
        assert valid is True
    
    def test_validate_action_on_occupied_cell_fails(self):
        validator = ActionValidator()
        state = GameState(structures={(5, 5): "DRILL"}, resources={"copper": 100}, map_width=20, map_height=20)
        action = Action(type="PLACE_CONVEYOR", position=(5, 5), source="ai", timestamp=0.0)
        
        valid, reason = validator.validate(action, state)
        assert valid is False
        assert "occupied" in reason.lower()
    
    def test_validate_action_insufficient_resources(self):
        validator = ActionValidator()
        state = GameState(structures={}, resources={"copper": 5}, map_width=20, map_height=20)
        action = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        
        valid, reason = validator.validate(action, state)
        assert valid is False
        assert "resource" in reason.lower()
    
    def test_detect_conflict_same_position(self):
        validator = ActionValidator()
        a1 = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        a2 = Action(type="PLACE_CONVEYOR", position=(5, 5), source="ai", timestamp=0.5)
        
        conflicts = validator.detect_conflict(a1, a2)
        assert conflicts is True
    
    def test_detect_no_conflict_different_positions(self):
        validator = ActionValidator()
        a1 = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        a2 = Action(type="PLACE_CONVEYOR", position=(6, 5), source="ai", timestamp=0.5)
        
        conflicts = validator.detect_conflict(a1, a2)
        assert conflicts is False
