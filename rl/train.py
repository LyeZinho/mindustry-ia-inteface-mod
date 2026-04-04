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
import json
import signal
import shutil
from pathlib import Path
from typing import Callable

from sb3_contrib import MaskablePPO

from rl.env.mindustry_env import MindustryEnv
from rl.env.spaces import GRID_CHANNELS, OBS_FEATURES_DIM
from rl.callbacks.training_callbacks import make_callbacks
from rl.models.custom_policy import MindustryActorCriticPolicy


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


def _find_latest_checkpoint(models_dir: str) -> tuple[str | None, str | None]:
    p = Path(models_dir)

    candidates = list(p.glob("mindustry_ppo_*_steps.zip"))
    if candidates:
        def _steps(f: Path) -> int:
            try:
                return int(f.stem.rsplit("_", 2)[-2])
            except (ValueError, IndexError):
                return -1
        latest = max(candidates, key=_steps)
        n = _steps(latest)
        vec = p / f"vecnormalize_{n}_steps.pkl"
        if not vec.exists():
            vec = p / "vecnormalize.pkl"
        return str(latest), str(vec) if vec.exists() else None

    final = p / "final_model.zip"
    if final.exists():
        vec = p / "vecnormalize.pkl"
        return str(final), str(vec) if vec.exists() else None

    return None, None


def _make_env_factory(host: str, tcp_port: int, max_steps: int, maps) -> Callable:
    def _factory():
        from rl.env.mindustry_env import MindustryEnv
        return MindustryEnv(host=host, tcp_port=tcp_port, max_steps=max_steps, maps=maps)
    return _factory


def _save_metadata(models_dir: str, total_timesteps: int) -> None:
    meta = {
        "grid_channels": GRID_CHANNELS,
        "obs_features_dim": OBS_FEATURES_DIM,
        "total_timesteps": total_timesteps,
        "policy": "MindustryActorCriticPolicy",
    }
    path = Path(models_dir) / "metadata_final.json"
    path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved to {path}")


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
    p.add_argument("--n-steps", type=int, default=2048, dest="n_steps")
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
        default="mimi-gateway-v1.0.7.zip",
        dest="mod_zip",
        help="Path to the Mimi Gateway mod zip to install",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="CHECKPOINT",
        dest="resume",
        help="Resume from checkpoint. Omit path to auto-detect latest in --models-dir.",
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
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            venv = DummyVecEnv([lambda: Monitor(
                MindustryEnv(host=args.host, tcp_port=args.port, max_steps=args.max_steps, maps=maps),
                filename=f"{args.logs_dir}/monitor",
            )])
        else:
            from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
            env_fns = [
                _make_env_factory(
                    host=args.host,
                    tcp_port=args.port + i,
                    max_steps=args.max_steps,
                    maps=maps,
                )
                for i in range(args.n_envs)
            ]
            venv = VecMonitor(SubprocVecEnv(env_fns), filename=f"{args.logs_dir}/monitor")

        resume_checkpoint: str | None = None
        resume_vecnorm: str | None = None
        if args.resume is not None:
            if args.resume == "auto":
                resume_checkpoint, resume_vecnorm = _find_latest_checkpoint(args.models_dir)
                if resume_checkpoint is None:
                    raise FileNotFoundError(f"No checkpoint found in {args.models_dir}")
                print(f"Auto-detected checkpoint: {resume_checkpoint}")
            else:
                resume_checkpoint = args.resume
                resume_vecnorm, _ = None, None
                p_cp = Path(resume_checkpoint)
                n_str = p_cp.stem.rsplit("_", 2)
                if len(n_str) >= 2:
                    vec_candidate = Path(args.models_dir) / f"vecnormalize_{n_str[-2]}_steps.pkl"
                    if vec_candidate.exists():
                        resume_vecnorm = str(vec_candidate)
                if resume_vecnorm is None:
                    fallback = Path(args.models_dir) / "vecnormalize.pkl"
                    if fallback.exists():
                        resume_vecnorm = str(fallback)

        from stable_baselines3.common.vec_env import VecNormalize
        if resume_vecnorm is not None:
            print(f"Loading VecNormalize stats: {resume_vecnorm}")
            env = VecNormalize.load(resume_vecnorm, venv=venv)
            env.training = True
        else:
            env = VecNormalize(venv, norm_obs=True, norm_reward=False)

        if resume_checkpoint is not None:
            print(f"Resuming from: {resume_checkpoint}")
            model = MaskablePPO.load(
                resume_checkpoint,
                env=env,
                learning_rate=_make_lr_schedule(args.lr, args.lr_end),
            )
        else:
            model = MaskablePPO(
                policy=MindustryActorCriticPolicy,
                env=env,
                learning_rate=_make_lr_schedule(args.lr, args.lr_end),
                n_steps=args.n_steps,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.05,
                verbose=1,
                tensorboard_log=args.logs_dir,
            )

        callbacks = make_callbacks(save_path=args.models_dir, logs_dir=args.logs_dir)

        reset_num_timesteps = resume_checkpoint is None
        print(f"Starting MaskablePPO training for {args.timesteps:,} timesteps ({args.n_envs} envs)...")
        model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=reset_num_timesteps)

        final_path = f"{args.models_dir}/final_model"
        model.save(final_path)
        env.save(f"{args.models_dir}/vecnormalize.pkl")
        _save_metadata(args.models_dir, args.timesteps)
        print(f"Training complete. Model saved to {final_path}.zip")

    finally:
        _shutdown()


if __name__ == "__main__":
    main()
