import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.adversarial.cgan_model import Discriminator, Generator
from src.adversarial.dataset import TabularCGANDataset


def train(parsed_args):
    """Train cGAN for a specific dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = parsed_args.dataset

    #  Use provided paths if passed, otherwise infer automatically
    parquet_path = parsed_args.parquet or f"data/processed/{dataset_name}_train.parquet"
    features_path = (
        parsed_args.features or f"artifacts/preproc/{dataset_name}_features.json"
    )
    scaler_path = parsed_args.scaler or f"artifacts/preproc/{dataset_name}_scaler.pkl"
    encoders_path = (
        parsed_args.encoders or f"artifacts/preproc/{dataset_name}_encoders.pkl"
    )
    out_path = parsed_args.out or f"artifacts/adversarial/{dataset_name}_cgan.pt"

    print(f"\n Training cGAN for dataset: {dataset_name}")
    print(f" Data: {parquet_path}")

    dataset = TabularCGANDataset(
        parquet_path=parquet_path,
        features_file=features_path,
        scaler_path=scaler_path,
        encoders_path=encoders_path,
        condition="asset",
        train_on=parsed_args.train_on,
    )

    d_in = dataset.X.shape[1]
    cond_dim = dataset.conds.shape[1]
    z_dim = parsed_args.z_dim

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
        total_d_loss = 0.0
        total_g_loss = 0.0

        for real_x, cond in loader:
            real_x, cond = real_x.to(device), cond.to(device)
            bs = real_x.shape[0]

            # Train Discriminator
            z = torch.randn(bs, z_dim, device=device)
            fake_x = g(z, cond).detach()
            loss_d = bce(d(real_x, cond), torch.ones(bs, 1, device=device)) + bce(
                d(fake_x, cond), torch.zeros(bs, 1, device=device)
            )
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # Train Generator
            z = torch.randn(bs, z_dim, device=device)
            fake_x = g(z, cond)
            loss_g = bce(d(fake_x, cond), torch.ones(bs, 1, device=device))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            total_d_loss += loss_d.item()
            total_g_loss += loss_g.item()

        if epoch % parsed_args.log_interval == 0 or epoch == 1:
            print(
                f"[{dataset_name}] Epoch {epoch}/{parsed_args.epochs} | "
                f"D={total_d_loss/len(loader):.4f} G={total_g_loss/len(loader):.4f}"
            )

    #  Save model
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gen_state": g.state_dict(),
            "disc_state": d.state_dict(),
            "z_dim": z_dim,
            "cond_dim": cond_dim,
            "D_in": d_in,
        },
        out_path,
    )

    print(f" Saved cGAN for {dataset_name} -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train Conditional GAN for dataset anomalies."
    )
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. iot_fridge)")

    # ✅ Optional overrides for batch scripts
    p.add_argument("--parquet", help="Path to parquet file")
    p.add_argument("--features", help="Path to features.json")
    p.add_argument("--scaler", help="Path to scaler.pkl")
    p.add_argument("--encoders", help="Path to encoders.pkl")
    p.add_argument("--out", help="Output model path")

    # Hyperparameters
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
