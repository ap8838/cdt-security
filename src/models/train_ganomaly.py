import argparse
import json
import os

import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

from src.models.ganomaly import GANomaly
from src.utils.seed import set_seed


def train_ganomaly(
    dataset: str,
    features_file=None,
    epochs=20,
    lr=1e-4,
    seed=42,
    lambda_adv=1.0,
    lambda_latent=1.0,
):
    set_seed(seed)

    # paths
    train_file = f"data/processed/{dataset}_train.parquet"
    model_path = f"artifacts/models/{dataset}_ganomaly.pt"
    threshold_path = f"artifacts/models/{dataset}_ganomaly_threshold.json"
    log_path = f"artifacts/reports/{dataset}_ganomaly_train_log.json"

    if features_file is None:
        features_file = f"artifacts/preproc/{dataset}_features.json"

    # load features
    with open(features_file) as f:
        features = json.load(f)
    cols = [c for c in features["all"] if c not in ("asset_id", "timestamp", "label")]

    # load train parquet and filter for Normal data (label=0)
    df = pd.read_parquet(train_file)
    df = df[df["label"] == 0].copy()
    x = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float32")
        .values
    )

    # split
    x_train, x_val = train_test_split(x, test_size=0.2, random_state=seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)

    # model
    model = GANomaly(input_dim=x_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_log = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon, z, z_hat, d_real, d_fake = model(x_train)

        # --- Loss Calculation with Robust Fallback (from second code) ---
        # --- Loss Calculation with Robust Fallback ---
        used_lambdas = True  # We consider them "used" because we apply them manually
        try:
            # Get the raw component losses
            _, recon_loss, latent_loss, adv_loss = GANomaly.loss_function(
                x_train, recon, z, z_hat, d_real, d_fake,
            )
        except TypeError:
            # Fallback if the signature is different
            _, recon_loss, latent_loss, adv_loss = GANomaly.loss_function(
                x_train, recon, z, z_hat, d_real, d_fake
            )

        # ALWAYS apply weights manually here to ensure consistency
        total_loss = recon_loss + (lambda_latent * latent_loss) + (lambda_adv * adv_loss)
        total_loss.backward()
        optimizer.step()

        # --- Stability Checks (from first code) ---
        if not torch.isfinite(total_loss.detach()):
            print("❗ Non-finite loss detected — stopping early")
            break
        if total_loss.item() > 1e6:
            print("❗ Loss exploded >1e6 — stopping early")
            break
        # ------------------------------------------

        # val
        model.eval()
        with torch.no_grad():
            recon_v, z_v, z_hat_v, d_real_v, d_fake_v = model(x_val)

            # Use same robust method for validation loss
            try:
                _, val_recon, val_latent, val_adv = GANomaly.loss_function(
                    x_val, recon_v, z_v, z_hat_v, d_real_v, d_fake_v,
                )
            except TypeError:
                _, val_recon, val_latent, val_adv = GANomaly.loss_function(
                    x_val, recon_v, z_v, z_hat_v, d_real_v, d_fake_v
                )

            # Apply weights manually for validation total loss
            val_loss = val_recon + (lambda_latent * val_latent) + (lambda_adv * val_adv)

        print(
            f"[{dataset}] Epoch {epoch + 1}/{epochs}, "
            f"Train Loss={total_loss.item():.6f}, Val Loss={val_loss.item():.6f}"
            + (
                f"  (using lambdas adv={lambda_adv}, latent={lambda_latent})"
                if used_lambdas
                else ""
            )
        )

        train_log.append(
            {
                "epoch": epoch + 1,
                "train_total": float(total_loss.item()),
                "val_total": float(val_loss.item()),
                "train_recon": float(recon_loss.item()),
                "train_latent": float(latent_loss.item()),
                "train_adv": float(adv_loss.item()),
                "used_lambdas": used_lambdas,
                "lambda_adv": lambda_adv,
                "lambda_latent": lambda_latent,
            }
        )

    # save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # compute val errors for threshold (recon + latent as score)
    with torch.no_grad():
        recon, z, z_hat, _, _ = model(x_val)
        recon_err = torch.mean((x_val - recon) ** 2, dim=1)
        latent_err = torch.mean((z - z_hat) ** 2, dim=1)
        errors = (recon_err + latent_err).cpu().numpy()

    # Calculate 99th percentile for threshold
    threshold = float(pd.Series(errors).quantile(0.99))

    # save threshold + logs
    with open(threshold_path, "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)

    print(f" [{dataset}] GANomaly model saved, threshold={threshold:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--features_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # new tunable loss weights
    parser.add_argument(
        "--lambda-adv",
        type=float,
        default=1.0,
        help="Weight for adversarial loss term (GAN part)",
    )
    parser.add_argument(
        "--lambda-latent",
        type=float,
        default=1.0,
        help="Weight for latent consistency loss term",
    )

    args = parser.parse_args()

    train_ganomaly(
        dataset=args.dataset,
        features_file=args.features_file,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        lambda_adv=args.lambda_adv,
        lambda_latent=args.lambda_latent,
    )
