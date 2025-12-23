import argparse
import json
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from src.utils.seed import set_seed

from .autoencoder import Autoencoder


def train_autoencoder(dataset: str, features_file=None, epochs=20, lr=1e-3, seed=42):
    # ensure reproducibility
    set_seed(seed)

    # paths per dataset
    train_file = f"data/processed/{dataset}_train.parquet"
    model_path = f"artifacts/models/{dataset}_ae.pt"
    threshold_path = f"artifacts/models/{dataset}_threshold.json"
    reports_dir = "artifacts/reports"
    report_path = f"{reports_dir}/{dataset}_ae_eval.json"
    log_path = f"{reports_dir}/{dataset}_ae_train_log.json"

    # choose correct features.json if not provided
    if features_file is None:
        features_file = f"artifacts/preproc/{dataset}_features.json"

    # 1. Load feature list
    with open(features_file) as f:
        features = json.load(f)
    cols = features["all"]
    cols = [c for c in cols if c not in ("asset_id", "timestamp", "label")]

    # 2. Load train parquet (normal only)
    df = pd.read_parquet(train_file)

    # Keep only numeric data
    x = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float32")
        .values
    )

    # 3. Train/val split
    x_train, x_val = train_test_split(x, test_size=0.2, random_state=seed)

    # 4. Torch tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)

    # 5. Model setup
    model = Autoencoder(input_dim=x_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logs = []  # Collect per-epoch losses

    # 6. Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, x_train)
        loss.backward()
        optimizer.step()

        # validation loss
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), x_val).item()

        # record log
        logs.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss),
            }
        )

        print(
            f"[{dataset}] Epoch {epoch+1}/{epochs}, "
            f"Train Loss={loss.item():.6f}, Val Loss={val_loss:.6f}"
        )

    # 7. Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # 8. Compute reconstruction errors on validation set
    with torch.no_grad():
        recon = model(x_val)
        errors = torch.mean((x_val - recon) ** 2, dim=1).cpu().numpy()

    threshold = float(pd.Series(errors).quantile(0.99))

    # 9. Save threshold
    with open(threshold_path, "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    # 10. Save small evaluation report
    os.makedirs(reports_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({"val_loss": val_loss, "threshold": threshold}, f, indent=2)

    # 11. Save per-epoch training logs
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)

    print(f" [{dataset}] Model saved to {model_path}, threshold={threshold:.6f}")
    print(f" Training log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g. iot_fridge, iot_gps, linux_disk1)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--features_file",
        type=str,
        default=None,
        help="Path to features.json (default: artifacts/preproc/{dataset}_features.json)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_autoencoder(
        args.dataset,
        features_file=args.features_file,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
