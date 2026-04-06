import pytest
import torch
from mindustry_ai.rl.policy_net import PolicyNetwork


def test_policy_network_init():
    net = PolicyNetwork(flat_dim=15, spatial_h=32, spatial_w=32)
    assert net is not None


def test_forward_pass():
    net = PolicyNetwork(flat_dim=15, spatial_h=32, spatial_w=32)
    flat_obs = torch.randn(1, 15)
    spatial_obs = torch.randn(1, 3, 32, 32)
    
    action_logits, placement_mu, placement_sigma, value = net(flat_obs, spatial_obs)
    
    assert action_logits.shape == (1, 7)
    assert placement_mu.shape == (1, 2)
    assert placement_sigma.shape == (1, 2)
    assert value.shape == (1, 1)


def test_network_output_types():
    net = PolicyNetwork(flat_dim=15, spatial_h=32, spatial_w=32)
    flat_obs = torch.randn(1, 15)
    spatial_obs = torch.randn(1, 3, 32, 32)
    
    action_logits, placement_mu, placement_sigma, value = net(flat_obs, spatial_obs)
    
    assert isinstance(action_logits, torch.Tensor)
    assert isinstance(placement_mu, torch.Tensor)
    assert isinstance(placement_sigma, torch.Tensor)
    assert isinstance(value, torch.Tensor)
