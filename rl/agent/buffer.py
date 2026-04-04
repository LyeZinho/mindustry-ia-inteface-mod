"""
Trajectory buffer for collecting rollouts during training.
Stores: states, actions, rewards, dones, values, log_probs
"""

from __future__ import annotations

import numpy as np
import torch
from typing import NamedTuple


class Transition(NamedTuple):
    """Single trajectory step"""
    obs_grid: np.ndarray          # (C, 31, 31)
    obs_features: np.ndarray      # (F,)
    action: int
    reward: float
    done: bool
    value: float                  # V(s_t) from critic
    log_prob: float               # log π(a_t | s_t)


class TrajectoryBuffer:
    """
    Collects trajectories and computes returns/advantages.
    
    Usage:
        buffer = TrajectoryBuffer(gamma=0.99, lam=0.95)
        for step in range(n_steps):
            buffer.add(transition)
        returns, advantages = buffer.compute_returns_advantages()
    """
    
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        """
        Args:
            gamma: discount factor
            lam: GAE lambda for advantage estimation
        """
        self.gamma = gamma
        self.lam = lam
        self.transitions: list[Transition] = []
        self.last_value = 0.0
    
    def add(self, transition: Transition) -> None:
        """Add a step to buffer"""
        self.transitions.append(transition)
    
    def set_last_value(self, value: float) -> None:
        """Set bootstrap value for terminal state"""
        self.last_value = value
    
    def compute_returns_advantages(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute discounted returns and advantages using GAE.
        
        Returns:
            returns: (n_steps,) array of discounted returns
            advantages: (n_steps,) array of advantage estimates
        """
        n_steps = len(self.transitions)
        returns = np.zeros(n_steps, dtype=np.float32)
        advantages = np.zeros(n_steps, dtype=np.float32)
        
        # Compute TD residuals (deltas)
        values = np.array([t.value for t in self.transitions] + [self.last_value])
        rewards = np.array([t.reward for t in self.transitions])
        dones = np.array([t.done for t in self.transitions])
        
        deltas = np.zeros(n_steps, dtype=np.float32)
        for t in range(n_steps):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            deltas[t] = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
        
        # Compute GAE advantages
        gae = 0.0
        for t in reversed(range(n_steps)):
            gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Compute returns as advantages + values
        returns = advantages + values[:n_steps]
        
        return returns, advantages
    
    def clear(self) -> None:
        """Clear buffer"""
        self.transitions.clear()
