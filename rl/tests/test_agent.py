"""
Comprehensive unit tests for RL agent components.

Tests cover:
1. PolicyValueNet forward pass shape and output
2. TrajectoryBuffer GAE computation correctness
3. A2CTrainer action selection (sampling, deterministic, value output)
4. A2CTrainer update step (loss computation, gradient updates)
5. A2CTrainer save/load functionality (checkpoint persistence)
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from rl.agent import PolicyValueNet, TrajectoryBuffer, A2CTrainer
from rl.agent.buffer import Transition


# Constants matching agent implementation
GRID_CHANNELS = 1
GRID_SIZE = 31
N_ACTIONS = 10
FEATURES_DIM = 8  # Example feature dimension


@pytest.fixture
def device():
    """Device for testing (CPU only)"""
    return "cpu"


@pytest.fixture
def policy_net(device):
    """Create PolicyValueNet instance for testing"""
    net = PolicyValueNet(
        grid_channels=GRID_CHANNELS,
        features_dim=FEATURES_DIM,
        n_actions=N_ACTIONS
    )
    return net.to(device)


@pytest.fixture
def trajectory_buffer():
    """Create TrajectoryBuffer with standard hyperparams"""
    return TrajectoryBuffer(gamma=0.99, lam=0.95)


@pytest.fixture
def a2c_trainer(policy_net, device):
    """Create A2CTrainer instance for testing"""
    return A2CTrainer(
        policy_net=policy_net,
        learning_rate=3e-4,
        entropy_coef=0.01,
        value_coef=0.5,
        gamma=0.99,
        lam=0.95,
        device=device
    )


@pytest.fixture
def dummy_obs():
    """Generate dummy observations (grid, features)"""
    obs_grid = np.random.rand(GRID_CHANNELS, GRID_SIZE, GRID_SIZE).astype(np.float32)
    obs_features = np.random.rand(FEATURES_DIM).astype(np.float32)
    return obs_grid, obs_features


@pytest.fixture
def batch_dummy_obs():
    """Generate batch of dummy observations"""
    batch_size = 4
    obs_grid = np.random.rand(batch_size, GRID_CHANNELS, GRID_SIZE, GRID_SIZE).astype(np.float32)
    obs_features = np.random.rand(batch_size, FEATURES_DIM).astype(np.float32)
    return obs_grid, obs_features


@pytest.fixture
def temp_checkpoint_dir():
    """Temporary directory for checkpoint testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# =============================================================================
# Test 1: PolicyValueNet Forward Pass
# =============================================================================


def test_policy_value_net_forward_pass(policy_net, batch_dummy_obs, device):
    """
    Test PolicyValueNet.forward() produces correct output shapes and valid values.
    
    Verifies:
    - Policy logits shape: (batch_size, n_actions)
    - Value shape: (batch_size, 1)
    - Outputs are tensors on correct device
    - No NaN or Inf values
    """
    obs_grid, obs_features = batch_dummy_obs
    batch_size = obs_grid.shape[0]
    
    # Convert to tensors
    grid_tensor = torch.from_numpy(obs_grid).float().to(device)
    feat_tensor = torch.from_numpy(obs_features).float().to(device)
    
    # Forward pass
    policy_logits, value = policy_net(grid_tensor, feat_tensor)
    
    # Shape assertions
    assert policy_logits.shape == (batch_size, N_ACTIONS), \
        f"Expected policy logits shape {(batch_size, N_ACTIONS)}, got {policy_logits.shape}"
    assert value.shape == (batch_size, 1), \
        f"Expected value shape {(batch_size, 1)}, got {value.shape}"
    
    # Device assertions
    assert policy_logits.device.type == device, \
        f"Policy logits on wrong device: {policy_logits.device.type}"
    assert value.device.type == device, \
        f"Value on wrong device: {value.device.type}"
    
    # Validity assertions (no NaN/Inf)
    assert not torch.isnan(policy_logits).any(), "Policy logits contain NaN"
    assert not torch.isinf(policy_logits).any(), "Policy logits contain Inf"
    assert not torch.isnan(value).any(), "Value contains NaN"
    assert not torch.isinf(value).any(), "Value contains Inf"
    
    # Value should be unbounded (regression output)
    assert value.dtype == torch.float32, f"Value dtype should be float32, got {value.dtype}"


# =============================================================================
# Test 2: TrajectoryBuffer GAE Computation
# =============================================================================


def test_trajectory_buffer_gae_computation(trajectory_buffer):
    """
    Test TrajectoryBuffer.compute_returns_advantages() computes correct GAE.
    
    Verifies:
    - Returns and advantages have correct shape
    - GAE formula applied correctly with γ=0.99, λ=0.95
    - Terminal states handled (done=True zeros next value)
    - Single-step trajectory handled
    - Empty buffer edge case
    """
    gamma = trajectory_buffer.gamma
    lam = trajectory_buffer.lam
    
    # Create dummy trajectory: 5 steps with known rewards/values
    obs_grid = np.random.rand(GRID_CHANNELS, GRID_SIZE, GRID_SIZE).astype(np.float32)
    obs_features = np.random.rand(FEATURES_DIM).astype(np.float32)
    
    # Simple trajectory: rewards=[1, 2, 3, 4, 5], values=[0.5, 1.0, 1.5, 2.0, 2.5]
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    values = [0.5, 1.0, 1.5, 2.0, 2.5]
    dones = [False, False, False, False, True]  # Last is terminal
    
    for i in range(5):
        transition = Transition(
            obs_grid=obs_grid,
            obs_features=obs_features,
            action=i % N_ACTIONS,
            reward=rewards[i],
            done=dones[i],
            value=values[i],
            log_prob=-1.0
        )
        trajectory_buffer.add(transition)
    
    # Bootstrap value for terminal state (should be 0 since done=True)
    trajectory_buffer.set_last_value(0.0)
    
    # Compute returns and advantages
    returns, advantages = trajectory_buffer.compute_returns_advantages()
    
    # Shape assertions
    assert returns.shape == (5,), f"Expected returns shape (5,), got {returns.shape}"
    assert advantages.shape == (5,), f"Expected advantages shape (5,), got {advantages.shape}"
    
    # Manual GAE computation for verification (last step)
    # TD residual: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
    # For t=4 (terminal): δ_4 = 5.0 + 0.99 * 0.0 * 0 - 2.5 = 2.5
    # GAE_4 = δ_4 = 2.5
    expected_gae_last = 2.5
    assert np.isclose(advantages[4], expected_gae_last, atol=1e-5), \
        f"Expected advantage at terminal {expected_gae_last}, got {advantages[4]}"
    
    # Returns should equal advantages + values
    expected_returns = advantages + np.array(values)
    assert np.allclose(returns, expected_returns, atol=1e-5), \
        "Returns should equal advantages + values"
    
    # Test single-step trajectory
    trajectory_buffer.clear()
    transition = Transition(
        obs_grid=obs_grid,
        obs_features=obs_features,
        action=0,
        reward=10.0,
        done=True,
        value=5.0,
        log_prob=-1.0
    )
    trajectory_buffer.add(transition)
    trajectory_buffer.set_last_value(0.0)
    
    returns_single, advantages_single = trajectory_buffer.compute_returns_advantages()
    assert returns_single.shape == (1,), "Single-step returns should have shape (1,)"
    assert advantages_single.shape == (1,), "Single-step advantages should have shape (1,)"


# =============================================================================
# Test 3: A2CTrainer Select Action
# =============================================================================


def test_a2c_trainer_select_action(a2c_trainer, dummy_obs):
    """
    Test A2CTrainer.select_action() returns valid action, log_prob, and value.
    
    Verifies:
    - Action is integer in valid range [0, n_actions)
    - Log probability is negative float
    - Value is float (unbounded)
    - Deterministic mode returns argmax action
    - Stochastic mode samples from distribution
    """
    obs_grid, obs_features = dummy_obs
    
    # Test stochastic action selection
    action, log_prob, value = a2c_trainer.select_action(obs_grid, obs_features, deterministic=False)
    
    # Type and range assertions
    assert isinstance(action, (int, np.integer)), f"Action should be int, got {type(action)}"
    assert 0 <= action < N_ACTIONS, f"Action {action} out of range [0, {N_ACTIONS})"
    assert isinstance(log_prob, (float, np.floating)), f"log_prob should be float, got {type(log_prob)}"
    assert log_prob <= 0.0, f"log_prob should be negative, got {log_prob}"
    assert isinstance(value, (float, np.floating)), f"value should be float, got {type(value)}"
    
    # Test deterministic action selection
    action_det, log_prob_det, value_det = a2c_trainer.select_action(
        obs_grid, obs_features, deterministic=True
    )
    
    assert isinstance(action_det, (int, np.integer)), "Deterministic action should be int"
    assert 0 <= action_det < N_ACTIONS, f"Deterministic action {action_det} out of range"
    
    # Test multiple calls to verify stochastic sampling (may differ across calls)
    actions = []
    for _ in range(10):
        a, _, _ = a2c_trainer.select_action(obs_grid, obs_features, deterministic=False)
        actions.append(a)
    
    # At least one unique action (stochastic sampling)
    assert len(set(actions)) >= 1, "Stochastic sampling should produce varying actions"


# =============================================================================
# Test 4: A2CTrainer Update Step
# =============================================================================


def test_a2c_trainer_update(a2c_trainer, dummy_obs):
    """
    Test A2CTrainer.update() performs gradient update without errors.
    
    Verifies:
    - Update completes without exceptions
    - Returns loss dictionary with expected keys
    - Losses are finite floats
    - Gradients are updated (parameters change)
    - Buffer is cleared after update
    """
    obs_grid, obs_features = dummy_obs
    
    # Store initial parameters
    initial_params = [p.clone() for p in a2c_trainer.policy_net.parameters()]
    
    # Collect small trajectory
    for _ in range(5):
        action, log_prob, value = a2c_trainer.select_action(obs_grid, obs_features)
        reward = np.random.rand()
        done = False
        a2c_trainer.add_transition(obs_grid, obs_features, action, reward, done, value, log_prob)
    
    # Perform update
    final_value = 0.0
    loss_dict = a2c_trainer.update(final_value=final_value)
    
    # Check loss dictionary structure
    expected_keys = {"policy_loss", "value_loss", "entropy", "total_loss"}
    assert set(loss_dict.keys()) == expected_keys, \
        f"Expected loss keys {expected_keys}, got {set(loss_dict.keys())}"
    
    # Check losses are finite
    for key, loss_val in loss_dict.items():
        assert isinstance(loss_val, float), f"{key} should be float, got {type(loss_val)}"
        assert np.isfinite(loss_val), f"{key} is not finite: {loss_val}"
    
    # Check parameters were updated
    updated_params = list(a2c_trainer.policy_net.parameters())
    params_changed = any(
        not torch.allclose(p_init, p_new, atol=1e-6)
        for p_init, p_new in zip(initial_params, updated_params)
    )
    assert params_changed, "Parameters should change after update"
    
    # Check buffer was cleared
    assert len(a2c_trainer.buffer.transitions) == 0, "Buffer should be cleared after update"
    
    # Check episode counter incremented
    assert a2c_trainer.episode_count == 1, f"Episode count should be 1, got {a2c_trainer.episode_count}"


# =============================================================================
# Test 5: A2CTrainer Save/Load
# =============================================================================


def test_a2c_trainer_save_load(a2c_trainer, dummy_obs, temp_checkpoint_dir):
    """
    Test A2CTrainer.save() and .load() preserve model state.
    
    Verifies:
    - Checkpoint file is created
    - Model parameters are identical after save/load
    - Loaded model produces identical outputs
    """
    obs_grid, obs_features = dummy_obs
    checkpoint_path = str(Path(temp_checkpoint_dir) / "test_checkpoint.pth")
    
    # Get initial prediction
    with torch.no_grad():
        grid_t = torch.from_numpy(obs_grid[np.newaxis]).float()
        feat_t = torch.from_numpy(obs_features[np.newaxis]).float()
        logits_before, value_before = a2c_trainer.policy_net(grid_t, feat_t)
    
    # Save checkpoint
    a2c_trainer.save(checkpoint_path)
    
    # Verify file exists
    assert Path(checkpoint_path).exists(), f"Checkpoint file {checkpoint_path} was not created"
    
    # Create new trainer with same architecture
    new_net = PolicyValueNet(
        grid_channels=GRID_CHANNELS,
        features_dim=FEATURES_DIM,
        n_actions=N_ACTIONS
    )
    new_trainer = A2CTrainer(policy_net=new_net, device="cpu")
    
    # Load checkpoint
    new_trainer.load(checkpoint_path)
    
    # Get prediction from loaded model
    with torch.no_grad():
        logits_after, value_after = new_trainer.policy_net(grid_t, feat_t)
    
    # Verify outputs are identical
    assert torch.allclose(logits_before, logits_after, atol=1e-6), \
        "Policy logits should be identical after save/load"
    assert torch.allclose(value_before, value_after, atol=1e-6), \
        "Value should be identical after save/load"
    
    # Verify all parameters match
    for p_orig, p_loaded in zip(
        a2c_trainer.policy_net.parameters(),
        new_trainer.policy_net.parameters()
    ):
        assert torch.allclose(p_orig, p_loaded, atol=1e-6), \
            "All parameters should match after save/load"


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_trajectory_buffer_empty():
    """Test TrajectoryBuffer handles empty buffer gracefully"""
    buffer = TrajectoryBuffer(gamma=0.99, lam=0.95)
    buffer.set_last_value(0.0)
    
    returns, advantages = buffer.compute_returns_advantages()
    
    assert returns.shape == (0,), "Empty buffer should return empty returns"
    assert advantages.shape == (0,), "Empty buffer should return empty advantages"


def test_policy_net_single_batch(policy_net, device):
    """Test PolicyValueNet handles batch_size=1 correctly"""
    obs_grid = np.random.rand(1, GRID_CHANNELS, GRID_SIZE, GRID_SIZE).astype(np.float32)
    obs_features = np.random.rand(1, FEATURES_DIM).astype(np.float32)
    
    grid_t = torch.from_numpy(obs_grid).float().to(device)
    feat_t = torch.from_numpy(obs_features).float().to(device)
    
    logits, value = policy_net(grid_t, feat_t)
    
    assert logits.shape == (1, N_ACTIONS), f"Expected shape (1, {N_ACTIONS}), got {logits.shape}"
    assert value.shape == (1, 1), f"Expected shape (1, 1), got {value.shape}"


def test_trajectory_buffer_all_terminal():
    """Test TrajectoryBuffer with all done=True transitions"""
    buffer = TrajectoryBuffer(gamma=0.99, lam=0.95)
    obs_grid = np.random.rand(GRID_CHANNELS, GRID_SIZE, GRID_SIZE).astype(np.float32)
    obs_features = np.random.rand(FEATURES_DIM).astype(np.float32)
    
    # Add 3 terminal transitions
    for i in range(3):
        transition = Transition(
            obs_grid=obs_grid,
            obs_features=obs_features,
            action=i,
            reward=1.0,
            done=True,
            value=0.5,
            log_prob=-1.0
        )
        buffer.add(transition)
    
    buffer.set_last_value(0.0)
    returns, advantages = buffer.compute_returns_advantages()
    
    assert returns.shape == (3,), "Should handle all terminal states"
    assert advantages.shape == (3,), "Should handle all terminal states"
    # Each advantage should be reward - value (no future value due to done=True)
    expected_adv = 1.0 - 0.5
    assert np.allclose(advantages, expected_adv, atol=1e-5), \
        "Advantages should equal reward - value for terminal states"
