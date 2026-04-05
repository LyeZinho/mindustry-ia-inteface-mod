import torch
import logging
from typing import Optional, Tuple
from mindustry_ai.rl.policy_net import PolicyNetwork
from mindustry_ai.rules.hybrid_decider import HybridDecider
from mindustry_ai.coordinator.action_queue import Action


logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, policy_net: PolicyNetwork, fallback_decider: Optional[HybridDecider] = None):
        self.policy_net = policy_net
        self.fallback_decider = fallback_decider
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
    
    def infer(self, flat_state: torch.Tensor, spatial_state: torch.Tensor) -> Action:
        try:
            # Move inputs to device
            flat_state = flat_state.to(self.device)
            spatial_state = spatial_state.to(self.device)
            
            with torch.no_grad():
                action_logits, placement_mu, placement_sigma, value = self.policy_net(flat_state, spatial_state)
            
            action_idx = torch.argmax(action_logits, dim=1).item()
            position = self._decode_position(placement_mu[0].cpu().numpy())
            
            action_type = self._idx_to_action_type(action_idx)
            
            return Action(
                type=action_type,
                position=position,
                source="ai",
                timestamp=0.0
            )
        except Exception as e:
            logger.error(f"NN inference failed: {e}")
            
            if self.fallback_decider is not None:
                return self.fallback_decider.decide({})
            
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _decode_position(self, placement_mu: list) -> Tuple[int, int]:
        x = max(0, min(19, int(placement_mu[0] * 20)))
        y = max(0, min(19, int(placement_mu[1] * 20)))
        return (x, y)
    
    def _idx_to_action_type(self, idx: int) -> str:
        action_map = {
            0: "PLACE_DRILL",
            1: "PLACE_CONVEYOR",
            2: "PLACE_CONTAINER",
            3: "WAIT",
            4: "REMOVE",
            5: "BUILD_TURRET",
            6: "PLACE_POWER",
        }
        return action_map.get(idx, "WAIT")
