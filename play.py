#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Optional
import glob
import os

import numpy as np
from sb3_contrib import MaskablePPO

from rl.env.spaces import GRID_CHANNELS, OBS_FEATURES_DIM
from rl.models.custom_policy import MindustryActorCriticPolicy


def find_latest_model(models_dir: str = "rl/models") -> Optional[str]:
    pattern = os.path.join(models_dir, "mindustry_ppo_*.zip")
    models = glob.glob(pattern)
    if not models:
        return None
    models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return models[0][:-4]


def load_and_validate_model(model_path: str, models_dir: str) -> MaskablePPO:
    meta_path = Path(models_dir) / "metadata_final.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("grid_channels") != GRID_CHANNELS:
            raise ValueError(
                f"Model grid_channels={meta['grid_channels']} != env {GRID_CHANNELS}"
            )
        if meta.get("obs_features_dim") != OBS_FEATURES_DIM:
            raise ValueError(
                f"Model obs_features_dim={meta['obs_features_dim']} != env {OBS_FEATURES_DIM}"
            )
    return MaskablePPO.load(model_path, custom_objects={"policy_class": MindustryActorCriticPolicy})


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Debug mode: Run RL agent continuously on server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python play.py                              # Latest model, continuous play
  python play.py --model mindustry_ppo_360000_steps  # Specific model
  python play.py --max-steps 3000                    # Reset every 3000 steps
  python play.py --port 9001                         # Custom port

Press Ctrl+C to stop.
        """
    )
    
    p.add_argument(
        "--model",
        default=None,
        help="Model name (without .zip). Default: auto-detect latest",
    )
    p.add_argument(
        "--models-dir",
        default="rl/models",
        dest="models_dir",
        help="Directory containing trained models (default: rl/models)",
    )
    p.add_argument(
        "--host",
        default="localhost",
        help="Server host (default: localhost)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Server TCP port (default: 9000)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        dest="max_steps",
        help="Max steps before resetting (default: 5000)",
    )
    p.add_argument(
        "--server-jar",
        default="server-release.jar",
        dest="server_jar",
        help="Path to server-release.jar (default: server-release.jar)",
    )
    p.add_argument(
        "--server-data-dir",
        default="rl/server_data",
        dest="server_data_dir",
        help="Server data directory (default: rl/server_data)",
    )
    p.add_argument(
        "--mod-zip",
        default="mimi-gateway-v1.0.7.zip",
        dest="mod_zip",
        help="Mimi Gateway mod zip path (default: mimi-gateway-v1.0.7.zip)",
    )
    p.add_argument(
        "--no-server",
        action="store_true",
        dest="no_server",
        help="Skip server startup (connect to already-running server)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed logs",
    )
    
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    
    if args.model is None:
        latest = find_latest_model(args.models_dir)
        if latest is None:
            print("❌ No trained models found in", args.models_dir)
            print("   Train a model first: python -m rl.train")
            sys.exit(1)
        args.model = latest
        print(f"📦 Using latest model: {Path(args.model).name}")
    else:
        full_path = Path(args.models_dir) / args.model
        if not full_path.exists() and not (full_path.with_suffix(".zip")).exists():
            print(f"❌ Model not found: {full_path}")
            sys.exit(1)
        args.model = str(full_path)
    
    Path(args.server_data_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🤖 Loading model: {args.model}")
    try:
        model = load_and_validate_model(args.model, args.models_dir)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Server startup
        
    servers = []
    
    def _shutdown():
        for srv in servers:
            try:
                srv.stop()
                print("🛑 Server stopped")
            except Exception as e:
                if args.verbose:
                    print(f"   Warning: {e}")
    
    def _on_sigterm(signum, frame):
        print("\n⏹️  Stopping...")
        _shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)
    
    if not args.no_server:
        print(f"🚀 Starting Mindustry server on port {args.port}...")
        try:
            from rl.server.manager import start_n_servers
            servers = start_n_servers(
                n=1,
                base_tcp_port=args.port,
                base_data_dir=args.server_data_dir,
                jar_path=args.server_jar,
                mod_zip=args.mod_zip,
            )
            print("✅ Server ready\n")
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            sys.exit(1)
    
    try:
        from rl.env.mindustry_env import MindustryEnv
        
        env = MindustryEnv(
            host=args.host,
            tcp_port=args.port,
            max_steps=args.max_steps,
        )
        
        print(f"▶️  AI playing continuously (Ctrl+C to stop)...\n")

        # During inference there is no training callback to set _global_timestep,
        # so the curriculum would stay at Phase 0 (only 3 actions allowed) forever.
        # Force full curriculum by setting a large timestep value.
        env._global_timestep = 10_000_000

        episode_num = 1
        total_steps = 0
        
        while True:
            print(f"📍 Episode {episode_num}", end=" | ")
            
            obs, info = env.reset()
            episode_reward = 0.0
            step_count = 0
            done = False
            
            while not done:
                action_masks = env.action_masks()
                action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                total_steps += 1
                done = terminated or truncated
            
            print(f"Reward: {episode_reward:.1f} | Steps: {step_count} | Total: {total_steps}")
            episode_num += 1
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
