from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl.env.spaces import GRID_CHANNELS, GRID_SIZE, OBS_FEATURES_DIM


class MindustryFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, GRID_CHANNELS, GRID_SIZE, GRID_SIZE)
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU(),
        )
        self.cnn_norm = nn.LayerNorm(512)

        self.mlp = nn.Sequential(
            nn.Linear(OBS_FEATURES_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mlp_norm = nn.LayerNorm(256)

        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        grid = observations["grid"]
        feats = observations["features"]
        cnn_out = self.cnn_norm(self.cnn_linear(self.cnn(grid)))
        mlp_out = self.mlp_norm(self.mlp(feats))
        fused = torch.cat([cnn_out, mlp_out], dim=1)
        return self.fusion(fused)


class MultiHeadCritic(nn.Module):
    def __init__(self, features_dim: int) -> None:
        super().__init__()
        self.head_survival = nn.Linear(features_dim, 1)
        self.head_economy = nn.Linear(features_dim, 1)
        self.head_defense = nn.Linear(features_dim, 1)
        self.head_build = nn.Linear(features_dim, 1)
        self.weights = nn.Parameter(torch.tensor([0.3, 1.0, 0.1, 0.2]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        v_surv = self.head_survival(features)
        v_econ = self.head_economy(features)
        v_def = self.head_defense(features)
        v_build = self.head_build(features)
        w = torch.softmax(self.weights, dim=0)
        return w[0] * v_surv + w[1] * v_econ + w[2] * v_def + w[3] * v_build

    def head_values(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.head_survival(features),
            self.head_economy(features),
            self.head_defense(features),
            self.head_build(features),
        ], dim=1)


try:
    from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy

    class MindustryActorCriticPolicy(MaskableActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            kwargs["features_extractor_class"] = MindustryFeatureExtractor
            kwargs["features_extractor_kwargs"] = {"features_dim": 256}
            kwargs["net_arch"] = []
            super().__init__(*args, **kwargs)

        def _build(self, lr_schedule) -> None:
            super()._build(lr_schedule)
            self.value_net = MultiHeadCritic(self.features_dim)

        def predict_values(self, obs) -> torch.Tensor:
            features = self.extract_features(obs, self.vf_features_extractor)
            latent_vf = self.mlp_extractor.forward_critic(features)
            return self.value_net(latent_vf)

        def get_head_values(self, obs) -> torch.Tensor:
            with torch.no_grad():
                features = self.extract_features(obs, self.vf_features_extractor)
                latent_vf = self.mlp_extractor.forward_critic(features)
                return self.value_net.head_values(latent_vf)

except ImportError:
    pass
