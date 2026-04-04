"""
A2C (Actor-Critic) trainer with custom training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Optional

from rl.agent.model import PolicyValueNet
from rl.agent.buffer import TrajectoryBuffer, Transition


class A2CTrainer:
    """
    Lightweight A2C trainer.
    
    Per episode:
    1. Collect trajectory with current policy
    2. Compute returns & advantages
    3. Update policy & value networks
    """
    
    def __init__(
        self,
        policy_net: PolicyValueNet,
        learning_rate: float = 3e-4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: str = "cpu",
    ):
        """
        Args:
            policy_net: PolicyValueNet instance
            learning_rate: initial LR
            entropy_coef: entropy bonus coefficient
            value_coef: value loss weight
            max_grad_norm: gradient clipping threshold
            gamma: discount factor
            lam: GAE lambda
            device: "cpu" or "cuda"
        """
        self.policy_net = policy_net.to(device)
        self.device = device
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        
        self.optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        self.buffer = TrajectoryBuffer(gamma=gamma, lam=lam)
        
        self.total_steps = 0
        self.episode_count = 0
    
    def select_action(
        self,
        obs_grid: np.ndarray,
        obs_features: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """
        Select action from policy.
        
        Args:
            obs_grid: (C, 31, 31)
            obs_features: (F,)
            deterministic: if True, take argmax; else sample
        
        Returns:
            action: int in [0, n_actions)
            log_prob: log probability of action
            value: V(s) from critic
        """
        with torch.no_grad():
            grid_t = torch.from_numpy(obs_grid[np.newaxis]).float().to(self.device)
            feat_t = torch.from_numpy(obs_features[np.newaxis]).float().to(self.device)
            
            logits, value_t = self.policy_net(grid_t, feat_t)
            probs = torch.softmax(logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                action = torch.multinomial(probs, 1).item()
            
            log_prob = torch.log(probs[0, action] + 1e-8).item()
            value = value_t.squeeze().item()
        
        return action, log_prob, value
    
    def add_transition(
        self,
        obs_grid: np.ndarray,
        obs_features: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Add experience to buffer"""
        trans = Transition(
            obs_grid=obs_grid,
            obs_features=obs_features,
            action=action,
            reward=reward,
            done=done,
            value=value,
            log_prob=log_prob,
        )
        self.buffer.add(trans)
    
    def update(self, final_value: float = 0.0) -> dict[str, float]:
        """
        Update policy and value networks using collected trajectory.
        
        Args:
            final_value: bootstrap value for terminal state
        
        Returns:
            dict with loss info
        """
        self.buffer.set_last_value(final_value)
        returns, advantages = self.buffer.compute_returns_advantages()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        transitions = self.buffer.transitions
        grids = np.stack([t.obs_grid for t in transitions])
        features = np.stack([t.obs_features for t in transitions])
        actions = np.array([t.action for t in transitions])
        log_probs_old = np.array([t.log_prob for t in transitions])
        
        grids_t = torch.from_numpy(grids).float().to(self.device)
        features_t = torch.from_numpy(features).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        returns_t = torch.from_numpy(returns).float().to(self.device)
        advantages_t = torch.from_numpy(advantages).float().to(self.device)
        
        logits, values = self.policy_net(grids_t, features_t)
        probs = torch.softmax(logits, dim=-1)
        log_probs_new = torch.log(probs.gather(1, actions_t.unsqueeze(1)) + 1e-8).squeeze(1)
        
        policy_loss = -(log_probs_new * advantages_t).mean()
        
        value_loss = ((values.squeeze() - returns_t) ** 2).mean()
        
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.buffer.clear()
        self.episode_count += 1
        self.total_steps += len(transitions)
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }
    
    def save(self, path: str) -> None:
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load model checkpoint"""
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
