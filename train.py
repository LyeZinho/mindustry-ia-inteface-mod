import argparse
import yaml
import torch
from pathlib import Path
from mindustry_ai.rl.trainer import A2CTrainer
from mindustry_ai.rl.policy_net import PolicyNetwork
from mindustry_ai.env.game_env import MindustryEnv


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_policy_network(config):
    return PolicyNetwork(
        flat_dim=config["model"]["flat_dim"],
        spatial_h=config["model"]["spatial_h"],
        spatial_w=config["model"]["spatial_w"],
    )


def create_trainer(policy_net, config):
    return A2CTrainer(
        policy_net=policy_net,
        learning_rate=config["training"]["learning_rate"],
        gamma=config["training"]["gamma"],
        gae_lambda=config["training"]["gae_lambda"],
    )


def create_environment(config):
    return MindustryEnv(
        max_steps=config["environment"]["max_steps_per_episode"],
        map_size=config["environment"]["map_size"],
    )


def train_phase(trainer, env, phase_config, phase_name):
    print(f"\n{'='*60}")
    print(f"Starting {phase_name}")
    print(f"{'='*60}")
    print(f"Goal: {phase_config['goal']}")
    print(f"Episodes: {phase_config['total_episodes']}")

    episode_rewards = []

    for episode in range(phase_config["total_episodes"]):
        trajectory = trainer.collect_trajectory(env, max_steps=100)
        loss = trainer.training_step(trajectory)

        episode_reward = sum(trajectory["rewards"].cpu().numpy())
        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(
                f"Episode {episode + 1}/{phase_config['total_episodes']}: "
                f"Loss={loss.item():.4f}, Avg Reward={avg_reward:.4f}"
            )

    return episode_rewards


def main():
    parser = argparse.ArgumentParser(
        description="Train hybrid RL agent for Mindustry"
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["phase_1_survival", "phase_2_production", "phase_3_defense", "all"],
        default="phase_1_survival",
        help="Which training phase to run",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to load checkpoint from",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    policy_net = create_policy_network(config)
    trainer = create_trainer(policy_net, config)
    env = create_environment(config)

    if args.checkpoint:
        policy_net.load_state_dict(torch.load(args.checkpoint))
        print(f"Loaded checkpoint: {args.checkpoint}")

    phases_to_run = []
    if args.phase == "all":
        phases_to_run = [
            ("phase_1_survival", config["phases"]["phase_1_survival"]),
            ("phase_2_production", config["phases"]["phase_2_production"]),
            ("phase_3_defense", config["phases"]["phase_3_defense"]),
        ]
    else:
        phases_to_run = [
            (args.phase, config["phases"][args.phase]),
        ]

    all_rewards = []

    for phase_key, phase_config in phases_to_run:
        rewards = train_phase(trainer, env, phase_config, phase_config["name"])
        all_rewards.extend(rewards)

    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}")
    print(f"Total episodes trained: {len(all_rewards)}")
    print(f"Average episode reward: {sum(all_rewards) / len(all_rewards):.4f}")


if __name__ == "__main__":
    main()
