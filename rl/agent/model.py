"""
PolicyValueNet: Lightweight A2C policy-value network.

Combines CNN feature extraction with MLP for policy and value heads.
Used for custom A2C training on Mindustry environments.
"""

import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    """
    Actor-Critic network with shared CNN+MLP body and separate policy/value heads.

    Architecture:
        - CNN Branch: Conv2d(grid_channels → 16 → 32) → Flatten
        - Project: CNN_output → 128-dim, features → 64-dim
        - Shared Body: Linear(192 → 256) → ReLU
        - Policy Head: Linear(256 → n_actions)
        - Value Head: Linear(256 → 1)

    Args:
        grid_channels (int): Number of input channels from grid observation.
        features_dim (int): Dimension of flattened features vector (non-grid input).
        n_actions (int): Number of possible actions. Default: 10.
    """

    def __init__(self, grid_channels, features_dim, n_actions=10):
        super(PolicyValueNet, self).__init__()

        # CNN branch for grid observations
        self.conv1 = nn.Conv2d(grid_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        # Compute CNN output dimension via dummy pass (grid is 31x31)
        with torch.no_grad():
            dummy_grid = torch.zeros(1, grid_channels, 31, 31)
            dummy_out = self.relu(self.conv1(dummy_grid))
            dummy_out = self.relu(self.conv2(dummy_out))
            cnn_output_dim = dummy_out.view(1, -1).shape[1]

        # Project CNN output to 128-dim
        self.cnn_proj = nn.Linear(cnn_output_dim, 128)

        # Project features to 64-dim
        self.feat_proj = nn.Linear(features_dim, 64)

        # Shared body (192 = 128 + 64)
        self.body = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
        )

        # Policy head: outputs logits over actions
        self.policy_head = nn.Linear(256, n_actions)

        # Value head: outputs scalar value estimate
        self.value_head = nn.Linear(256, 1)

    def forward(self, grid, features):
        """
        Forward pass.

        Args:
            grid (torch.Tensor): Grid observation, shape (batch_size, grid_channels, H, W).
            features (torch.Tensor): Feature vector, shape (batch_size, features_dim).

        Returns:
            tuple: (action_logits, value)
                - action_logits: (batch_size, n_actions)
                - value: (batch_size, 1)
        """
        # CNN branch
        x = self.relu(self.conv1(grid))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.cnn_proj(x)

        # Feature branch
        feat = self.feat_proj(features)

        # Concatenate and pass through shared body
        combined = torch.cat([x, feat], dim=1)
        body_out = self.body(combined)

        # Separate heads
        action_logits = self.policy_head(body_out)
        value = self.value_head(body_out)

        return action_logits, value
