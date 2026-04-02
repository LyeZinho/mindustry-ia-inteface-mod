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
import shutil
import socket
import time
from pathlib import Path
from typing import Optional

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from rl.env.mindustry_env import MindustryEnv
from rl.callbacks.training_callbacks import make_callbacks


def _wait_for_port(host: str, port: int, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Mod TCP port {port} not available after {timeout:.0f}s")


def _install_mod(mod_zip: str, server_data_dir: str) -> None:
    src = Path(mod_zip)
    if not src.exists():
        raise FileNotFoundError(f"Mod zip not found: {src}")
    dest_dir = Path(server_data_dir) / "config" / "mods"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "mimi-gateway.zip"
    shutil.copy2(src, dest)
    print(f"Mod installed: {dest}")


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
    p.add_argument(
        "--server-jar",
        default="server-release.jar",
        dest="server_jar",
        help="Path to server-release.jar",
    )
    p.add_argument(
        "--maps",
        default=None,
        help="Comma-separated map names to cycle (default: built-in list)",
    )
    p.add_argument(
        "--no-server",
        action="store_true",
        dest="no_server",
        help="Skip spawning server (connect to already-running server)",
    )
    p.add_argument(
        "--server-data-dir",
        default="rl/server_data",
        dest="server_data_dir",
        help="Directory for Mindustry server data (saves, config)",
    )
    p.add_argument(
        "--mod-zip",
        default="mimi-gateway-v1.0.4.zip",
        dest="mod_zip",
        help="Path to the Mimi Gateway mod zip to install",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    maps = [m.strip() for m in args.maps.split(",")] if args.maps else None

    server: Optional["MindustryServer"] = None
    if not args.no_server:
        _install_mod(args.mod_zip, args.server_data_dir)
        from rl.server.manager import MindustryServer
        server = MindustryServer(jar_path=args.server_jar, data_dir=args.server_data_dir)
        print(f"Starting Mindustry server ({args.server_jar})...")
        server.start()
        print("Waiting for mod TCP port...")
        _wait_for_port(args.host, args.port)
        print("Server ready. Connecting agent...")

    try:
        env = Monitor(
            MindustryEnv(host=args.host, port=args.port, max_steps=args.max_steps, maps=maps),
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

    finally:
        if server is not None:
            print("Stopping server...")
            server.stop()


if __name__ == "__main__":
    main()
