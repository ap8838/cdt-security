"""
Simple conditional GAN (cGAN) for tabular feature vectors.
Generator: noise + condition -> feature vector (D)
Discriminator: feature vector + condition -> real/fake score
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim: int, cond_dim: int, out_dim: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim),
            nn.Tanh(),  # assuming scaled features in roughly -1..1 or 0..1 depending on scaler
        )

    def forward(self, z, cond):
        # z: (B, z_dim), cond: (B, cond_dim)
        x = torch.cat([z, cond], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
            # output raw score (use BCEWithLogitsLoss)
        )

    def forward(self, x, cond):
        x_in = torch.cat([x, cond], dim=1)
        return self.net(x_in).squeeze(1)
