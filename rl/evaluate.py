"""
Evaluate a saved Mindustry A2C model.

Usage:
    python -m rl.evaluate --model rl/models/final_model
    python -m rl.evaluate --model rl/models/final_model --episodes 5
"""
from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import A2C

from rl.env.mindustry_env import MindustryEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved Mindustry A2C model")
    p.add_argument("--model", required=True, help="Path to model .zip (without extension)")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=5000, dest="max_steps")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = A2C.load(args.model)
    env = MindustryEnv(host=args.host, port=args.port, max_steps=args.max_steps)

    episode_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)
        print(f"Episode {ep + 1}: reward={total_reward:.3f}")

    print(f"\nMean reward over {args.episodes} episodes: {np.mean(episode_rewards):.3f}")
    env.close()


if __name__ == "__main__":
    main()
