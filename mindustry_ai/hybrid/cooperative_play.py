import logging
from mindustry_ai.game.api_client import GameAPIClient
from mindustry_ai.rl.policy_net import PolicyNetwork
from mindustry_ai.rl.inference import InferenceEngine
from mindustry_ai.coordinator.action_queue import ActionQueue
from mindustry_ai.coordinator.game_loop import HybridGameLoop


logger = logging.getLogger(__name__)


class CooperativePlayManager:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        model_checkpoint: str = "model.pt",
        flat_dim: int = 15,
        spatial_h: int = 16,
        spatial_w: int = 16,
    ):
        self.host = host
        self.port = port
        self.model_checkpoint = model_checkpoint
        
        self.api_client = GameAPIClient(host=host, port=port)
        self.policy_net = PolicyNetwork(flat_dim=flat_dim, spatial_h=spatial_h, spatial_w=spatial_w)
        self.action_queue = ActionQueue()
        self.inference_engine = InferenceEngine(policy_net=self.policy_net)
        self.game_loop = HybridGameLoop(
            api_client=self.api_client,
            policy_net=self.policy_net,
            action_queue=self.action_queue,
            inference_engine=self.inference_engine,
        )
        
        self._load_model()
    
    def start(self) -> None:
        logger.info(f"Connecting to Mindustry at {self.host}:{self.port}")
        
        if not self.api_client.connect():
            raise RuntimeError(f"Failed to connect to {self.host}:{self.port}")
        
        self.game_loop.start()
        logger.info("Cooperative play started")
    
    def stop(self) -> None:
        self.game_loop.stop()
        self.api_client.disconnect()
        logger.info("Cooperative play stopped")
    
    def _load_model(self) -> None:
        try:
            self.inference_engine.load_checkpoint(self.model_checkpoint)
            logger.info(f"Model loaded from {self.model_checkpoint}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Using untrained network.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run cooperative play with Mindustry")
    parser.add_argument("--host", default="localhost", help="Mindustry server host")
    parser.add_argument("--port", type=int, default=8080, help="Mindustry server port")
    parser.add_argument("--model", default="model.pt", help="Path to trained model checkpoint")
    
    args = parser.parse_args()
    
    manager = CooperativePlayManager(host=args.host, port=args.port, model_checkpoint=args.model)
    
    try:
        manager.start()
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        manager.stop()
