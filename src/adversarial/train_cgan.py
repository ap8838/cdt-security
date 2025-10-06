import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.adversarial.cgan_model import Discriminator, Generator
from src.adversarial.dataset import TabularCGANDataset


# Renamed 'args' to 'parsed_args' to avoid shadowing
def train(parsed_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TabularCGANDataset(
        parquet_path=parsed_args.parquet,
        features_file=parsed_args.features,
        scaler_path=parsed_args.scaler,
        # RE-ADDED: encoders_path is needed for categorical feature processing
        encoders_path=parsed_args.encoders,
        condition="asset",
        train_on=parsed_args.train_on,
    )
    # Renamed D_in to d_in
    d_in = dataset.X.shape[1]
    cond_dim = dataset.conds.shape[1]
    z_dim = parsed_args.z_dim

    # Updated d_in usage
    g = Generator(
        z_dim=z_dim, cond_dim=cond_dim, out_dim=d_in, hidden=parsed_args.hidden
    ).to(device)
    d = Discriminator(in_dim=d_in, cond_dim=cond_dim, hidden=parsed_args.hidden).to(
        device
    )

    opt_g = optim.Adam(g.parameters(), lr=parsed_args.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(d.parameters(), lr=parsed_args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        dataset, batch_size=parsed_args.batch_size, shuffle=True, drop_last=True
    )

    for epoch in range(1, parsed_args.epochs + 1):
        g.train()
        d.train()
        # Renamed total_d_loss and total_g_loss to snake_case
        total_d_loss = 0.0
        total_g_loss = 0.0
        for real_x, cond in loader:
            real_x = real_x.to(device)
            cond = cond.to(device)
            bs = real_x.shape[0]

            # Train discriminator
            z = torch.randn(bs, z_dim, device=device)
            fake_x = g(z, cond).detach()
            real_logits = d(real_x, cond)
            fake_logits = d(fake_x, cond)

            loss_d = bce(real_logits, torch.ones_like(real_logits)) + bce(
                fake_logits, torch.zeros_like(fake_logits)
            )
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # Train generator
            z = torch.randn(bs, z_dim, device=device)
            fake_x = g(z, cond)
            fake_logits = d(fake_x, cond)
            loss_g = bce(fake_logits, torch.ones_like(fake_logits))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            total_d_loss += loss_d.item()
            total_g_loss += loss_g.item()

        if epoch % parsed_args.log_interval == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{parsed_args.epochs}  D_loss={total_d_loss/len(loader):.4f}  G_loss={total_g_loss/len(loader):.4f}"
            )

    # Save model and metadata
    out_dir = Path(parsed_args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gen_state": g.state_dict(),
            "disc_state": d.state_dict(),
            "z_dim": z_dim,
            "cond_dim": cond_dim,
            # Updated d_in key name
            "D_in": d_in,
        },
        parsed_args.out,
    )
    print("Saved cGAN to", parsed_args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--scaler", default="artifacts/preproc/iot_fridge_scaler.pkl")
    # RE-ADDED: encoders path is necessary for categorical features
    p.add_argument("--encoders", default="artifacts/preproc/iot_fridge_encoders.pkl")
    p.add_argument("--out", default="artifacts/adversarial/iot_fridge_cgan.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument(
        "--train-on", choices=["anomaly", "normal", "all"], default="anomaly"
    )
    args = p.parse_args()
    train(args)
