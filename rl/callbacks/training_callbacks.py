"""
Training callbacks for Stable Baselines3.

Usage in train.py:
    from rl.callbacks.training_callbacks import make_callbacks
    callbacks = make_callbacks(save_path="rl/models", logs_dir="rl/logs")
"""
from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)

from rl.env.spaces import ACTION_NAMES, NUM_ACTION_TYPES


def _write_metrics_json(path: Path, metrics: Dict[str, Any]) -> None:
    """Atomically write metrics dict to JSON using temp-file rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(metrics), encoding="utf-8")
    tmp.replace(path)


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


class LiveMetricsCallback(BaseCallback):
    """Writes live training metrics to a JSON sidecar file every rollout.

    The file is read by rl/dashboard.py for real-time visualization.
    Written atomically via temp-file rename to avoid partial reads.
    """

    def __init__(
        self,
        metrics_path: str,
        history_len: int = 200,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._metrics_path = Path(metrics_path)
        self._step_latencies: deque = deque(maxlen=history_len)
        self._value_history: List[float] = []
        self._power_history: List[Dict[str, float]] = []
        self._building_history: List[int] = []
        self._last_resources: Dict[str, float] = {}
        self._last_power: Dict[str, float] = {}
        self._last_building_count: int = 0
        self._last_unit_count: int = 0
        self._resources_at_rollout_start: Dict[str, float] = {}
        self._rollout_build_fails: int = 0
        self._build_fail_rate_history: List[float] = []
        self._episode_infos: List[Dict[str, Any]] = []

    def _on_rollout_start(self) -> None:
        if self.training_env is not None:
            self.training_env.set_attr("_global_timestep", self.num_timesteps)
        self._resources_at_rollout_start = dict(self._last_resources)
        self._rollout_build_fails = 0
        self._episode_infos = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            lat = info.get("step_latency_ms")
            if lat is not None:
                self._step_latencies.append(float(lat))
            resources = info.get("resources")
            if resources:
                self._last_resources = {k: float(v) for k, v in resources.items()}
            power = info.get("power")
            if power:
                self._last_power = {k: float(v) for k, v in power.items()}
            bc = info.get("buildings")
            if bc is not None:
                self._last_building_count = int(bc)
            uc = info.get("units")
            if uc is not None:
                self._last_unit_count = int(uc)
            if info.get("build_failed", False):
                self._rollout_build_fails += 1
            self._episode_infos.append(info)
        return True

    def _on_rollout_end(self) -> None:
        metrics = self._compute_metrics()
        try:
            self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
            _write_metrics_json(self._metrics_path, metrics)
        except OSError:
            pass

    def _compute_metrics(self) -> Dict[str, Any]:
        policy_data = self._compute_policy_metrics()
        episode_data = self._compute_episode_metrics()

        if self._last_power:
            self._power_history.append({
                "produced": self._last_power.get("produced", 0.0),
                "consumed": self._last_power.get("consumed", 0.0),
            })
            if len(self._power_history) > 100:
                self._power_history = self._power_history[-100:]

        self._building_history.append(self._last_building_count)
        if len(self._building_history) > 100:
            self._building_history = self._building_history[-100:]

        resource_deltas = {
            k: self._last_resources.get(k, 0.0) - self._resources_at_rollout_start.get(k, 0.0)
            for k in self._last_resources
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "policy": policy_data,
            "episode_metrics": episode_data,
            "world": {
                "resources": self._last_resources,
                "resource_deltas": resource_deltas,
                "power": self._last_power,
                "power_history": self._power_history[-50:],
                "building_count": self._last_building_count,
                "unit_count": self._last_unit_count,
                "building_history": self._building_history[-50:],
            },
            "pipeline": {
                "step_latency_ms_mean": float(np.mean(list(self._step_latencies))) if self._step_latencies else 0.0,
                "step_latency_ms_std": float(np.std(list(self._step_latencies))) if len(self._step_latencies) > 1 else 0.0,
                "step_latency_history": list(self._step_latencies)[-50:],
            },
            "training": {
                "total_timesteps": self.num_timesteps,
            },
        }

    def _compute_episode_metrics(self) -> Dict[str, Any]:
        """Aggregate drill, penalty, and action metrics from episode infos."""
        num_steps = len(self._episode_infos)
        
        drills_built_total = sum(info.get("drills_built_this_step", 0) for info in self._episode_infos)
        drill_build_frequency_pct = (drills_built_total / num_steps * 100) if num_steps > 0 else 0.0
        
        penalty_a_count = sum(1 for info in self._episode_infos if info.get("penalty_a_triggered", False))
        penalty_b_count = sum(1 for info in self._episode_infos if info.get("penalty_b_triggered", False))
        penalty_frequency_pct = ((penalty_a_count + penalty_b_count) / num_steps * 100) if num_steps > 0 else 0.0
        
        action_counts = [0] * NUM_ACTION_TYPES
        for info in self._episode_infos:
            action_idx = info.get("action_taken_index")
            if action_idx is not None and 0 <= action_idx < NUM_ACTION_TYPES:
                action_counts[action_idx] += 1
        
        total_actions = sum(action_counts)
        if total_actions > 0:
            action_dist = {name: count / total_actions for name, count in zip(ACTION_NAMES, action_counts)}
        else:
            action_dist = {name: 1.0 / NUM_ACTION_TYPES for name in ACTION_NAMES}
        
        return {
            "drills_built_total": int(drills_built_total),
            "drill_build_frequency_pct": float(drill_build_frequency_pct),
            "penalty_a_count": int(penalty_a_count),
            "penalty_b_count": int(penalty_b_count),
            "penalty_frequency_pct": float(penalty_frequency_pct),
            "action_dist": action_dist,
        }

    def _compute_policy_metrics(self) -> Dict[str, Any]:
        try:
            buf = self.model.rollout_buffer
            actions = buf.actions.reshape(-1, buf.actions.shape[-1]).astype(int)
            action_types = actions[:, 0]
            action_type_dist = (
                np.bincount(action_types, minlength=NUM_ACTION_TYPES) / max(len(action_types), 1)
            ).tolist()
            values = buf.values.flatten()
            value_mean = float(np.mean(values))
            self._value_history.append(value_mean)
            if len(self._value_history) > 100:
                self._value_history = self._value_history[-100:]
            mask_ratio = 0.0
            if hasattr(buf, "action_masks") and buf.action_masks is not None:
                masks = buf.action_masks.reshape(-1, buf.action_masks.shape[-1])
                mask_ratio = float(1.0 - np.mean(masks.astype(float)))
            build_attempts = int(np.sum((action_types >= 2) & (action_types <= 5)))
            fail_rate = self._rollout_build_fails / max(build_attempts, 1)
            self._build_fail_rate_history.append(fail_rate)
            if len(self._build_fail_rate_history) > 100:
                self._build_fail_rate_history = self._build_fail_rate_history[-100:]
            return {
                "action_type_distribution": action_type_dist,
                "value_mean": value_mean,
                "value_history": self._value_history[-50:],
                "mask_ratio_blocked": mask_ratio,
                "build_fail_rate": fail_rate,
                "build_fail_rate_history": self._build_fail_rate_history[-50:],
            }
        except Exception:
            return {
                "action_type_distribution": [1 / NUM_ACTION_TYPES] * NUM_ACTION_TYPES,
                "value_mean": 0.0,
                "value_history": self._value_history[-50:],
                "mask_ratio_blocked": 0.0,
                "build_fail_rate": 0.0,
                "build_fail_rate_history": self._build_fail_rate_history[-50:],
            }


def make_callbacks(
    save_path: str = "rl/models",
    logs_dir: str = "rl/logs",
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
    live_metrics = LiveMetricsCallback(
        metrics_path=str(Path(logs_dir) / "live_metrics.json"),
        verbose=verbose,
    )
    return CallbackList([checkpoint_cb, reward_logger, live_metrics])
