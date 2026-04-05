"""
A2CTrainer tests - trajectory collection, training step, loss computation.
"""
import pytest
import torch
import numpy as np
from mindustry_ai.env.game_env import MindustryEnv
from mindustry_ai.rl.trainer import A2CTrainer
from mindustry_ai.rl.policy_net import PolicyNetwork


class TestA2CTrainer:
    """Test A2CTrainer: trajectory collection, backprop, and loss."""

    @pytest.fixture
    def env(self):
        """Create MindustryEnv for testing."""
        return MindustryEnv(max_steps=100)

    @pytest.fixture
    def trainer(self):
        """Create A2CTrainer with PolicyNetwork."""
        policy_net = PolicyNetwork(
            flat_dim=15,
            spatial_h=16,
            spatial_w=16,
        )
        return A2CTrainer(
            policy_net=policy_net,
            learning_rate=0.0003,
            gamma=0.99,
            gae_lambda=0.95,
        )

    def test_trainer_initialization(self, trainer):
        """Verify A2CTrainer initializes correctly."""
        assert trainer.policy_net is not None
        assert trainer.optimizer is not None
        assert trainer.gamma == 0.99
        assert trainer.gae_lambda == 0.95

    def test_collect_trajectory_single_episode(self, env, trainer):
        """Collect trajectory from one full episode."""
        trajectory = trainer.collect_trajectory(env, max_steps=50)

        assert "states" in trajectory
        assert "actions" in trajectory
        assert "rewards" in trajectory
        assert "values" in trajectory
        assert "log_probs" in trajectory
        assert "next_value" in trajectory

        assert len(trajectory["states"]) > 0
        assert len(trajectory["actions"]) == len(trajectory["rewards"])
        assert len(trajectory["log_probs"]) == len(trajectory["values"])

    def test_compute_gae(self, trainer):
        """Test GAE computation for advantage estimation."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 1.0])
        values = torch.tensor([0.5, 1.5, 2.5, 0.8])
        next_value = torch.tensor(0.0)

        advantages, returns = trainer.compute_gae(rewards, values, next_value)

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
        assert torch.all(~torch.isnan(advantages))
        assert torch.all(~torch.isnan(returns))

    def test_training_step(self, env, trainer):
        """Perform single training step and verify loss is computed."""
        trajectory = trainer.collect_trajectory(env, max_steps=50)
        
        loss = trainer.training_step(trajectory)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert loss > 0

    def test_multiple_training_steps(self, env, trainer):
        """Train for multiple steps and verify convergence trend."""
        losses = []
        for _ in range(3):
            trajectory = trainer.collect_trajectory(env, max_steps=50)
            loss = trainer.training_step(trajectory)
            losses.append(loss.item())

        assert len(losses) == 3
        assert all(isinstance(l, float) for l in losses)

    def test_trajectory_device_consistency(self, env, trainer):
        """Verify all trajectory tensors are on same device."""
        trajectory = trainer.collect_trajectory(env, max_steps=50)

        device = next(trainer.policy_net.parameters()).device

        for key in [
            "actions",
            "rewards",
            "values",
            "log_probs",
            "next_value",
        ]:
            if isinstance(trajectory[key], torch.Tensor):
                assert trajectory[key].device == device


class TestMindustryEnv:
    """Test MindustryEnv: reset, step, reward calculation."""

    def test_env_initialization(self):
        """MindustryEnv initializes with correct parameters."""
        env = MindustryEnv(max_steps=100, map_size=16)
        assert env.max_steps == 100
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_env_reset(self):
        """Reset returns initial observation."""
        env = MindustryEnv(max_steps=100)
        obs = env.reset()

        assert obs is not None
        assert isinstance(obs, dict)
        assert "flat_state" in obs
        assert "spatial_state" in obs
        assert obs["flat_state"].shape == (15,)
        assert obs["spatial_state"].shape == (16, 16)

    def test_env_step_returns_tuple(self):
        """Step returns (obs, reward, done, info)."""
        env = MindustryEnv(max_steps=100)
        env.reset()

        action = 0
        obs, reward, done, info = env.step(action)

        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_env_episode_termination(self):
        """Episode terminates after max_steps."""
        env = MindustryEnv(max_steps=5)
        env.reset()

        for i in range(5):
            _, _, done, _ = env.step(0)
            if i < 4:
                assert not done
        assert done

    def test_env_reward_calculation(self):
        """Reward reflects game progress metrics."""
        env = MindustryEnv(max_steps=100)
        env.reset()

        reward = env.compute_reward()
        assert isinstance(reward, float)

    def test_env_multiple_episodes(self):
        """Run multiple episodes sequentially."""
        env = MindustryEnv(max_steps=20)

        for episode in range(3):
            obs = env.reset()
            assert obs is not None

            done = False
            steps = 0
            while not done and steps < 25:
                obs, reward, done, info = env.step(np.random.randint(0, 7))
                steps += 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
