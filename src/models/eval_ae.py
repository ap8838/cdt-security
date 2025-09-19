import argparse
import json
import os

import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.utils.seed import set_seed

from .autoencoder import Autoencoder


def evaluate_autoencoder(dataset: str, features_file=None, seed=42):
    # ensure reproducibility
    set_seed(seed)

    # paths per dataset
    test_file = f"data/processed/{dataset}_test.parquet"
    model_path = f"artifacts/models/{dataset}_ae.pt"
    threshold_path = f"artifacts/models/{dataset}_threshold.json"
    report_path = f"artifacts/reports/{dataset}_ae_eval.json"

    # choose correct features.json if not provided
    if features_file is None:
        features_file = f"artifacts/preproc/{dataset}_features.json"

    # 1. Load features
    with open(features_file) as f:
        features = json.load(f)
    cols = features["all"]
    cols = [c for c in cols if c not in ("asset_id", "timestamp", "label")]

    # 2. Load test parquet (normal + malicious)
    df = pd.read_parquet(test_file)

    # Force numeric conversion
    x = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float32")
        .values
    )
    y_true = df["label"].values

    # 3. Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder(input_dim=x.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Load threshold
    with open(threshold_path) as f:
        threshold = json.load(f)["threshold"]

    # 5. Compute reconstruction errors
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        recon = model(x_tensor)
        errors = torch.mean((x_tensor - recon) ** 2, dim=1).cpu().numpy()

    y_pred = (errors > threshold).astype(int)

    # 6. Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, errors)

    report = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "threshold": threshold,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✅ [{dataset}] Evaluation complete:", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g. iot_fridge, iot_gps, linux_disk1)",
    )
    parser.add_argument(
        "--features_file",
        type=str,
        default=None,
        help="Path to features.json (default: artifacts/preproc/{dataset}_features.json)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    evaluate_autoencoder(args.dataset, features_file=args.features_file, seed=args.seed)
