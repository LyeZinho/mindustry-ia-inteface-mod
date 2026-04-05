import pytest
from unittest.mock import Mock, patch
from mindustry_ai.hybrid.cooperative_play import CooperativePlayManager


class TestCooperativePlayManager:
    def test_manager_init(self):
        manager = CooperativePlayManager(
            host="localhost",
            port=8080,
            model_checkpoint="model.pt"
        )
        
        assert manager.host == "localhost"
        assert manager.port == 8080
    
    def test_manager_start_stop(self):
        manager = CooperativePlayManager(
            host="localhost",
            port=8080,
            model_checkpoint="model.pt"
        )
        
        with patch.object(manager, "game_loop") as mock_loop:
            manager.start()
            mock_loop.start.assert_called_once()
            
            manager.stop()
            mock_loop.stop.assert_called_once()
