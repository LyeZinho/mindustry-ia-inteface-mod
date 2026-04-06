import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, flat_dim: int = 15, spatial_h: int = 32, spatial_w: int = 32):
        super().__init__()
        self.flat_dim = flat_dim
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        
        # Flat stream: flat_obs → Dense layers
        self.flat_stream = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Spatial stream: spatial_obs → Conv layers
        self.spatial_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        # Calculate spatial stream output size after pooling
        self.spatial_out_size = 64 * 4 * 4
        
        # Fusion: concatenate flat (128) + spatial (1024) → shared layers
        fusion_in = 128 + self.spatial_out_size
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Output heads
        self.action_head = nn.Linear(128, 7)
        self.placement_mu_head = nn.Linear(128, 2)
        self.placement_sigma_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, flat_obs: torch.Tensor, spatial_obs: torch.Tensor):
        # Flat stream
        flat_feat = self.flat_stream(flat_obs)
        
        # Spatial stream
        spatial_feat = self.spatial_stream(spatial_obs)
        spatial_feat = spatial_feat.view(spatial_feat.size(0), -1)
        
        # Fusion
        combined = torch.cat([flat_feat, spatial_feat], dim=1)
        fused = self.fusion(combined)
        
        # Output heads
        action_logits = self.action_head(fused)
        placement_mu = self.placement_mu_head(fused)
        placement_sigma = F.softplus(self.placement_sigma_head(fused)) + 1e-5
        value = self.value_head(fused)
        
        return action_logits, placement_mu, placement_sigma, value
