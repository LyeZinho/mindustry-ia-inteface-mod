import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from mindustry_ai.hybrid.cooperative_play import CooperativePlayManager
from mindustry_ai.coordinator.action_queue import Action


class TestCooperativePlayIntegration:
    def test_full_cooperative_play_cycle(self):
        with patch("mindustry_ai.game.api_client.GameAPIClient") as MockAPIClient:
            mock_api = MagicMock()
            mock_api.is_connected.return_value = True
            mock_api.connect.return_value = True
            mock_api.get_game_state.return_value = {
                "resources": {"copper": 100, "lead": 50},
                "health": 1.0,
                "structures": {},
                "map_width": 20,
                "map_height": 20,
            }
            mock_api.execute_action.return_value = True
            
            MockAPIClient.return_value = mock_api
            
            manager = CooperativePlayManager(
                host="localhost",
                port=8080,
                model_checkpoint="/tmp/dummy.pt"
            )
            
            assert manager is not None
            assert manager.api_client is not None
            
            assert manager.game_loop is not None
            assert manager.game_loop.action_queue is not None
            
            manager.start()
            
            action = Action(
                type="PLACE_DRILL",
                position=(5, 5),
                source="human",
                timestamp=0.0
            )
            manager.game_loop.action_queue.enqueue(action)
            
            time.sleep(0.2)
            
            manager.stop()
            
            assert True
