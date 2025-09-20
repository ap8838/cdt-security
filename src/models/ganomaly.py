import torch
import torch.nn as nn
import torch.nn.functional as f


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_dim=100):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, z):
        return self.decoder(z)


class Generator(nn.Module):
    """Encoder → Decoder → Encoder"""

    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.encoder1 = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.encoder2 = Encoder(input_dim, latent_dim)

    def forward(self, x):
        z = self.encoder1(x)
        recon = self.decoder(z)
        z_hat = self.encoder2(recon)
        return recon, z, z_hat


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class GANomaly(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.generator = Generator(input_dim, latent_dim)
        self.discriminator = Discriminator(input_dim)

    def forward(self, x):
        recon, z, z_hat = self.generator(x)
        d_real = self.discriminator(x)
        d_fake = self.discriminator(recon)
        return recon, z, z_hat, d_real, d_fake

    @staticmethod
    def loss_function(x, recon, z, z_hat, d_real, d_fake):
        recon_loss = f.mse_loss(recon, x)
        latent_loss = f.mse_loss(z_hat, z)
        adv_loss = torch.mean(torch.abs(d_real - d_fake))
        total = recon_loss + latent_loss + adv_loss
        return total, recon_loss, latent_loss, adv_loss
