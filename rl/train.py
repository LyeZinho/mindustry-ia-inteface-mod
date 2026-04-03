"""
Training entry point for the Mindustry MaskablePPO agent.

Usage:
    python -m rl.train
    python -m rl.train --timesteps 500000 --host localhost --port 9000

Requires:
    - Mindustry running with Mimi Gateway mod loaded
    - pip install -r rl/requirements.txt
"""
from __future__ import annotations

import argparse
import signal
import shutil
from pathlib import Path
from typing import Callable

from sb3_contrib import MaskablePPO

from rl.env.mindustry_env import MindustryEnv
from rl.callbacks.training_callbacks import make_callbacks


def _make_lr_schedule(lr_start: float, lr_end: float) -> Callable[[float], float]:
    def _schedule(progress_remaining: float) -> float:
        return lr_end + (lr_start - lr_end) * progress_remaining
    return _schedule


def _install_mod(mod_zip: str, server_data_dir: str) -> None:
    src = Path(mod_zip)
    if not src.exists():
        raise FileNotFoundError(f"Mod zip not found: {src}")
    dest_dir = Path(server_data_dir) / "config" / "mods"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "mimi-gateway.zip"
    shutil.copy2(src, dest)
    print(f"Mod installed: {dest}")


def _make_env_factory(host: str, tcp_port: int, max_steps: int, maps) -> Callable:
    """Return a zero-arg callable that constructs one MindustryEnv (for SubprocVecEnv)."""
    def _factory():
        from rl.env.mindustry_env import MindustryEnv
        return MindustryEnv(host=host, tcp_port=tcp_port, max_steps=max_steps, maps=maps)
    return _factory


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Mindustry MaskablePPO agent")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--n-envs", type=int, default=4, dest="n_envs",
                   help="Number of parallel environments (default: 4)")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--max-steps", type=int, default=5000, dest="max_steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=1e-5, dest="lr_end",
                   help="Final learning rate for linear schedule (default: 1e-5)")
    p.add_argument("--n-steps", type=int, default=256, dest="n_steps")
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
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    maps = [m.strip() for m in args.maps.split(",")] if args.maps else None

    servers = []

    def _shutdown():
        for srv in servers:
            try:
                srv.stop()
            except Exception:
                pass

    def _on_sigterm(signum, frame):
        print("\nStopping servers (SIGTERM)...")
        _shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    if not args.no_server:
        from rl.server.manager import start_n_servers
        print(f"Starting {args.n_envs} Mindustry server instance(s)...")
        servers = start_n_servers(
            n=args.n_envs,
            base_tcp_port=args.port,
            base_data_dir=args.server_data_dir,
            jar_path=args.server_jar,
            mod_zip=args.mod_zip,
        )
        print(f"All servers ready. Game starts on first RESET from env.")

    try:
        if args.n_envs == 1:
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(
                MindustryEnv(host=args.host, tcp_port=args.port, max_steps=args.max_steps, maps=maps),
                filename=f"{args.logs_dir}/monitor",
            )
        else:
            from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
            env_fns = [
                _make_env_factory(
                    host=args.host,
                    tcp_port=args.port + i,
                    max_steps=args.max_steps,
                    maps=maps,
                )
                for i in range(args.n_envs)
            ]
            env = VecMonitor(SubprocVecEnv(env_fns), filename=f"{args.logs_dir}/monitor")

        model = MaskablePPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=_make_lr_schedule(args.lr, args.lr_end),
            n_steps=args.n_steps,
            gamma=0.95,
            gae_lambda=0.95,
            ent_coef=0.05,
            verbose=1,
            tensorboard_log=args.logs_dir,
        )

        callbacks = make_callbacks(save_path=args.models_dir, logs_dir=args.logs_dir)

        print(f"Starting MaskablePPO training for {args.timesteps:,} timesteps ({args.n_envs} envs)...")
        model.learn(total_timesteps=args.timesteps, callback=callbacks)

        final_path = f"{args.models_dir}/final_model"
        model.save(final_path)
        print(f"Training complete. Model saved to {final_path}.zip")

    finally:
        _shutdown()


if __name__ == "__main__":
    main()
