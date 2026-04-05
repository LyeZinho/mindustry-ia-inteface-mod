import argparse
import yaml
import torch
import time
from pathlib import Path
from collections import deque
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
    print(f"\n{'='*70}")
    print(f"🚀 {phase_name.upper()}")
    print(f"{'='*70}")
    print(f"Goal: {phase_config['goal']}")
    print(f"Episodes: {phase_config['total_episodes']}")
    print(f"Learning Rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")
    print(f"{'='*70}\n")

    episode_rewards = []
    episode_losses = deque(maxlen=10)
    phase_start = time.time()

    for episode in range(phase_config["total_episodes"]):
        ep_start = time.time()
        
        trajectory = trainer.collect_trajectory(env, max_steps=100)
        loss = trainer.training_step(trajectory)
        
        episode_reward = sum(trajectory["rewards"].cpu().numpy())
        episode_rewards.append(episode_reward)
        episode_losses.append(loss.item())
        
        ep_time = time.time() - ep_start
        
        progress = (episode + 1) / phase_config["total_episodes"]
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        if (episode + 1) % 1 == 0:
            avg_loss = sum(episode_losses) / len(episode_losses)
            avg_reward_10 = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            
            elapsed = time.time() - phase_start
            rate = elapsed / (episode + 1)
            remaining = rate * (phase_config["total_episodes"] - episode - 1)
            
            print(
                f"[{bar}] {episode + 1:3d}/{phase_config['total_episodes']} | "
                f"Loss: {avg_loss:.4f} | "
                f"Reward(10): {avg_reward_10:7.4f} | "
                f"Time: {ep_time:.2f}s | "
                f"ETA: {int(remaining//60):2d}m {int(remaining%60):02d}s"
            )

    phase_time = time.time() - phase_start
    final_avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
    
    print(f"\n{'─'*70}")
    print(f"✅ Phase Complete: {int(phase_time//60):2d}m {int(phase_time%60):02d}s")
    print(f"   Final Avg Reward (last 10): {final_avg_reward:.4f}")
    print(f"   Total Avg Reward: {sum(episode_rewards) / len(episode_rewards):.4f}")
    print(f"{'─'*70}\n")

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
    training_start = time.time()

    for phase_key, phase_config in phases_to_run:
        rewards = train_phase(trainer, env, phase_config, phase_config["name"])
        all_rewards.extend(rewards)

    total_time = time.time() - training_start
    
    print(f"\n{'='*70}")
    print("🎉 TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total Time: {int(total_time//60):2d}m {int(total_time%60):02d}s")
    print(f"Total Episodes: {len(all_rewards)}")
    print(f"Avg Episode Reward: {sum(all_rewards) / len(all_rewards):.4f}")
    print(f"Max Reward: {max(all_rewards):.4f}")
    print(f"Min Reward: {min(all_rewards):.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
