import pytest
import yaml
import torch
from pathlib import Path
from mindustry_ai.rl.trainer import A2CTrainer
from mindustry_ai.rl.policy_net import PolicyNetwork
from mindustry_ai.env.game_env import MindustryEnv


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class TestTrainingConfig:
    def test_config_loads(self):
        config = load_config()
        assert config is not None
        assert "phases" in config
        assert "training" in config
        assert "environment" in config

    def test_phases_structure(self):
        config = load_config()
        phases = config["phases"]
        assert "phase_1_survival" in phases
        assert "phase_2_production" in phases
        assert "phase_3_defense" in phases

        for phase_key, phase_config in phases.items():
            assert "name" in phase_config
            assert "waves" in phase_config
            assert "episodes_per_wave" in phase_config
            assert "total_episodes" in phase_config

    def test_training_hyperparameters(self):
        config = load_config()
        training = config["training"]
        assert training["learning_rate"] > 0
        assert training["gamma"] > 0
        assert training["gae_lambda"] > 0

    def test_environment_config(self):
        config = load_config()
        env = config["environment"]
        assert env["map_size"] > 0
        assert env["max_steps_per_episode"] > 0


class TestTrainingLoop:
    def test_single_phase_training_setup(self):
        config = load_config()
        phase_config = config["phases"]["phase_1_survival"]

        policy_net = PolicyNetwork(
            flat_dim=config["model"]["flat_dim"],
            spatial_h=config["model"]["spatial_h"],
            spatial_w=config["model"]["spatial_w"],
        )
        trainer = A2CTrainer(
            policy_net=policy_net,
            learning_rate=config["training"]["learning_rate"],
            gamma=config["training"]["gamma"],
            gae_lambda=config["training"]["gae_lambda"],
        )

        assert trainer is not None
        assert trainer.policy_net is not None

    def test_env_creation_with_config(self):
        config = load_config()
        env = MindustryEnv(
            max_steps=config["environment"]["max_steps_per_episode"],
            map_size=config["environment"]["map_size"],
        )

        assert env is not None
        obs = env.reset()
        assert obs is not None

    def test_episode_simulation(self):
        config = load_config()
        env = MindustryEnv(
            max_steps=config["environment"]["max_steps_per_episode"],
            map_size=config["environment"]["map_size"],
        )
        policy_net = PolicyNetwork(
            flat_dim=config["model"]["flat_dim"],
            spatial_h=config["model"]["spatial_h"],
            spatial_w=config["model"]["spatial_w"],
        )
        trainer = A2CTrainer(
            policy_net=policy_net,
            learning_rate=config["training"]["learning_rate"],
            gamma=config["training"]["gamma"],
            gae_lambda=config["training"]["gae_lambda"],
        )

        trajectory = trainer.collect_trajectory(env, max_steps=50)
        assert len(trajectory["states"]) > 0
        assert len(trajectory["rewards"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
