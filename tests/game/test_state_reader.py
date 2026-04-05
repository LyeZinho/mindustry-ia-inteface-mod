import pytest
from mindustry_ai.game.state_reader import GameStateReader


def test_read_game_state_returns_dict():
    """GameStateReader should return a state dict with expected keys."""
    reader = GameStateReader()
    state = reader.read_state()
    
    assert isinstance(state, dict)
    assert "resources" in state
    assert "power" in state
    assert "threat" in state
    assert "infrastructure" in state
    assert "status" in state


def test_flat_vector_generation():
    """Should convert state dict to flat observation vector."""
    reader = GameStateReader()
    state = reader.read_state()
    flat_vec = reader.to_flat_vector(state)
    
    assert len(flat_vec) == 15  # As per design doc


def test_spatial_map_generation():
    """Should generate 2D spatial representation of game map."""
    reader = GameStateReader()
    state = reader.read_state()
    spatial_map = reader.to_spatial_map(state)
    
    assert "blocks" in spatial_map
    assert "resources" in spatial_map
    assert "enemies" in spatial_map
