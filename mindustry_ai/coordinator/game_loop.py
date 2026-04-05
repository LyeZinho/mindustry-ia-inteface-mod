import threading
import time
import logging
from typing import Optional
import torch

from mindustry_ai.game.api_client import GameAPIClient
from mindustry_ai.rl.policy_net import PolicyNetwork
from mindustry_ai.rl.inference import InferenceEngine
from mindustry_ai.coordinator.action_queue import ActionQueue
from mindustry_ai.coordinator.validator import ActionValidator, GameState


logger = logging.getLogger(__name__)


class HybridGameLoop:
    def __init__(
        self,
        api_client: GameAPIClient,
        policy_net: PolicyNetwork,
        action_queue: ActionQueue,
        inference_engine: Optional[InferenceEngine] = None,
        cycle_interval: float = 1.0,
    ):
        self.api_client = api_client
        self.policy_net = policy_net
        self.action_queue = action_queue
        self.inference_engine = inference_engine or InferenceEngine(policy_net)
        self.validator = ActionValidator()
        
        self.cycle_interval = cycle_interval
        self.running = False
        self._main_thread = None
        self._inference_thread = None
    
    def start(self) -> None:
        if self.running:
            logger.warning("Game loop already running")
            return
        
        self.running = True
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        
        self._main_thread.start()
        self._inference_thread.start()
        
        logger.info("Game loop started")
    
    def stop(self) -> None:
        self.running = False
        
        if self._main_thread:
            self._main_thread.join(timeout=2.0)
        if self._inference_thread:
            self._inference_thread.join(timeout=2.0)
        
        logger.info("Game loop stopped")
    
    def _main_loop(self) -> None:
        while self.running:
            try:
                if not self.api_client.is_connected():
                    logger.warning("API not connected, waiting...")
                    time.sleep(0.5)
                    continue
                
                game_state_dict = self.api_client.get_game_state()
                game_state = self._dict_to_game_state(game_state_dict)
                
                while not self.action_queue.size() == 0:
                    action = self.action_queue.peek()
                    
                    valid, reason = self.validator.validate(action, game_state)
                    if valid:
                        self.action_queue.dequeue()
                        self.api_client.execute_action(action)
                        logger.debug(f"Executed {action.source} action: {action.type}")
                    else:
                        self.action_queue.dequeue()
                        logger.warning(f"Invalid action discarded: {reason}")
                
                time.sleep(self.cycle_interval)
            
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(0.5)
    
    def _inference_loop(self) -> None:
        while self.running:
            try:
                if not self.api_client.is_connected():
                    time.sleep(0.5)
                    continue
                
                game_state = self.api_client.get_game_state()
                
                flat_state = self._dict_to_tensor(game_state)
                spatial_state = torch.randn(1, 3, 16, 16)
                
                action = self.inference_engine.infer(flat_state, spatial_state)
                self.action_queue.enqueue(action)
                
                time.sleep(self.cycle_interval)
            
            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                time.sleep(self.cycle_interval)
    
    def _dict_to_tensor(self, game_state: dict) -> torch.Tensor:
        resources = list(game_state.get("resources", {}).values())[:15]
        resources += [0] * (15 - len(resources))
        return torch.tensor([resources], dtype=torch.float32)
    
    def _dict_to_game_state(self, game_state_dict: dict) -> GameState:
        return GameState(
            structures=game_state_dict.get("structures", {}),
            resources=game_state_dict.get("resources", {}),
            map_width=game_state_dict.get("map_width", 20),
            map_height=game_state_dict.get("map_height", 20)
        )
