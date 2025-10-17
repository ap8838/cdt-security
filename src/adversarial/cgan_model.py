"""
Simple conditional GAN (cGAN) for tabular feature vectors.
Generator: noise + condition -> feature vector (D)
Discriminator: feature vector + condition -> real/fake score
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim: int, cond_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, out_dim),
            nn.Tanh(),  # assuming scaled features roughly in -1..1
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # z: (B, z_dim), cond: (B, cond_dim)
        x = torch.cat([z, cond], dim=1)
        out = self.net(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
            # Output: single logit per sample (for BCEWithLogitsLoss)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim), cond: (B, cond_dim)
        x_in = torch.cat([x, cond], dim=1)
        out = self.net(x_in)
        # ✅ Keep shape as (B, 1) instead of squeezing
        return out
