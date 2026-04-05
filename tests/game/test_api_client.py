import pytest
from unittest.mock import Mock, patch
from mindustry_ai.game.api_client import GameAPIClient


class TestGameAPIClient:
    def test_api_client_init(self):
        client = GameAPIClient(host="localhost", port=8080)
        assert client.host == "localhost"
        assert client.port == 8080
        assert client.is_connected() is False
    
    def test_connect_success(self):
        client = GameAPIClient(host="localhost", port=8080)
        with patch.object(client, "_establish_connection", return_value=True):
            client.connect()
            assert client.is_connected() is True
    
    def test_disconnect(self):
        client = GameAPIClient(host="localhost", port=8080)
        client._connected = True
        client.disconnect()
        assert client.is_connected() is False
    
    def test_get_game_state_returns_dict(self):
        client = GameAPIClient(host="localhost", port=8080)
        client._connected = True
        
        mock_state = {
            "resources": {"copper": 100},
            "health": 0.8,
            "structures": []
        }
        
        with patch.object(client, "_fetch_state", return_value=mock_state):
            state = client.get_game_state()
            assert "resources" in state
            assert "health" in state
    
    def test_execute_action_calls_api(self):
        client = GameAPIClient(host="localhost", port=8080)
        client._connected = True
        
        from mindustry_ai.coordinator.action_queue import Action
        action = Action(type="PLACE_DRILL", position=(5, 5), source="ai", timestamp=0.0)
        
        with patch.object(client, "_send_action", return_value=True):
            result = client.execute_action(action)
            assert result is True
