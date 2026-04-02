"""
Training entry point for the Mindustry A2C agent.

Usage:
    python -m rl.train
    python -m rl.train --timesteps 500000 --host localhost --port 9000

Requires:
    - Mindustry running with Mimi Gateway mod loaded
    - pip install -r rl/requirements.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from rl.env.mindustry_env import MindustryEnv
from rl.callbacks.training_callbacks import make_callbacks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Mindustry A2C agent")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--max-steps", type=int, default=5000, dest="max_steps")
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--n-steps", type=int, default=128, dest="n_steps")
    p.add_argument("--models-dir", default="rl/models")
    p.add_argument("--logs-dir", default="rl/logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    env = Monitor(
        MindustryEnv(host=args.host, port=args.port, max_steps=args.max_steps),
        filename=f"{args.logs_dir}/monitor",
    )

    model = A2C(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=args.logs_dir,
    )

    callbacks = make_callbacks(save_path=args.models_dir)

    print(f"Starting A2C training for {args.timesteps:,} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    final_path = f"{args.models_dir}/final_model"
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
