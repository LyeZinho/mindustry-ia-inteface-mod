"""
Training callbacks for Stable Baselines3.

Usage in train.py:
    from rl.callbacks.training_callbacks import make_callbacks
    callbacks = make_callbacks(save_path="rl/models", log_path="rl/logs")
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)


class RewardLoggerCallback(BaseCallback):
    """Logs mean rollout reward to TensorBoard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_rewards: List[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self) -> None:
        if self._episode_rewards:
            mean_r = sum(self._episode_rewards) / len(self._episode_rewards)
            self.logger.record("rollout/mean_episode_reward", mean_r)
            self._episode_rewards.clear()


def make_callbacks(
    save_path: str = "rl/models",
    save_freq: int = 10_000,
    verbose: int = 1,
) -> CallbackList:
    Path(save_path).mkdir(parents=True, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix="mindustry_ppo",
        verbose=verbose,
    )
    reward_logger = RewardLoggerCallback(verbose=verbose)
    return CallbackList([checkpoint_cb, reward_logger])
