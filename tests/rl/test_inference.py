import pytest
import torch
from unittest.mock import Mock, patch
from mindustry_ai.rl.inference import InferenceEngine
from mindustry_ai.rl.policy_net import PolicyNetwork
from mindustry_ai.coordinator.action_queue import Action
from mindustry_ai.rules.hybrid_decider import HybridDecider


class TestInferenceEngine:
    def test_inference_engine_init(self):
        policy_net = PolicyNetwork(flat_dim=15, spatial_h=16, spatial_w=16)
        engine = InferenceEngine(policy_net=policy_net)
        
        assert engine.policy_net is not None
        assert engine.fallback_decider is None
    
    def test_infer_returns_action(self):
        policy_net = PolicyNetwork(flat_dim=15, spatial_h=16, spatial_w=16)
        engine = InferenceEngine(policy_net=policy_net)
        
        flat_state = torch.randn(1, 15)
        spatial_state = torch.randn(1, 3, 16, 16)
        
        action = engine.infer(flat_state, spatial_state)
        
        assert isinstance(action, Action)
        assert action.source == "ai"
        assert hasattr(action, "type")
        assert hasattr(action, "position")
    
    def test_infer_with_fallback_on_error(self):
        policy_net = PolicyNetwork(flat_dim=15, spatial_h=16, spatial_w=16)
        fallback_decider = Mock(spec=HybridDecider)
        fallback_decider.decide.return_value = Action(
            type="PLACE_DRILL", position=(5, 5), source="ai", timestamp=0.0
        )
        
        engine = InferenceEngine(policy_net=policy_net, fallback_decider=fallback_decider)
        
        flat_state = torch.randn(1, 15)
        spatial_state = torch.randn(1, 3, 16, 16)
        
        with patch.object(engine.policy_net, "forward", side_effect=RuntimeError("NN error")):
            action = engine.infer(flat_state, spatial_state)
            
            assert action.type == "PLACE_DRILL"
            fallback_decider.decide.assert_called_once_with({})
    
    def test_load_checkpoint(self):
        policy_net = PolicyNetwork(flat_dim=15, spatial_h=16, spatial_w=16)
        engine = InferenceEngine(policy_net=policy_net)
        
        checkpoint_path = "/tmp/test_checkpoint.pt"
        torch.save(policy_net.state_dict(), checkpoint_path)
        
        engine.load_checkpoint(checkpoint_path)
        
        assert engine.policy_net is not None
