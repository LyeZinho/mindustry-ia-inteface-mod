import pytest
import time
import threading
from unittest.mock import Mock, patch
from mindustry_ai.coordinator.game_loop import HybridGameLoop
from mindustry_ai.coordinator.action_queue import ActionQueue, Action


class TestHybridGameLoop:
    def test_game_loop_init(self):
        api_client = Mock()
        policy_net = Mock()
        action_queue = ActionQueue()
        
        loop = HybridGameLoop(api_client=api_client, policy_net=policy_net, action_queue=action_queue)
        
        assert loop.api_client is not None
        assert loop.action_queue is not None
        assert loop.running is False
    
    def test_game_loop_start_stop(self):
        api_client = Mock()
        api_client.is_connected.return_value = True
        api_client.get_game_state.return_value = {
            "resources": {"copper": 100},
            "health": 1.0,
            "structures": {}
        }
        
        policy_net = Mock()
        action_queue = ActionQueue()
        
        loop = HybridGameLoop(api_client=api_client, policy_net=policy_net, action_queue=action_queue)
        
        loop.start()
        assert loop.running is True
        
        time.sleep(0.1)
        loop.stop()
        
        assert loop.running is False
    
    def test_game_loop_executes_queued_actions(self):
        api_client = Mock()
        api_client.is_connected.return_value = True
        api_client.get_game_state.return_value = {
            "resources": {"copper": 100},
            "health": 1.0,
            "structures": {},
            "map_width": 20,
            "map_height": 20,
        }
        api_client.execute_action.return_value = True
        
        policy_net = Mock()
        action_queue = ActionQueue()
        
        inference_engine = Mock()
        inference_engine.infer.return_value = Action(
            type="WAIT", position=(10, 10), source="ai", timestamp=0.0
        )
        
        loop = HybridGameLoop(
            api_client=api_client,
            policy_net=policy_net,
            action_queue=action_queue,
            inference_engine=inference_engine,
            cycle_interval=0.01
        )
        
        action = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        action_queue.enqueue(action)
        
        loop.start()
        time.sleep(0.05)
        loop.stop()
        
        api_client.execute_action.assert_called()
