import argparse
import json
import os

import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.models.ganomaly import GANomaly


def evaluate_ganomaly(dataset: str, features_file=None):
    # paths
    test_file = f"data/processed/{dataset}_test.parquet"
    model_path = f"artifacts/models/{dataset}_ganomaly.pt"
    threshold_path = f"artifacts/models/{dataset}_ganomaly_threshold.json"
    report_path = f"artifacts/reports/{dataset}_ganomaly_eval.json"

    if features_file is None:
        features_file = f"artifacts/preproc/{dataset}_features.json"

    # load features
    with open(features_file) as f:
        features = json.load(f)
    cols = [c for c in features["all"] if c not in ("asset_id", "timestamp", "label")]

    # load test parquet
    df = pd.read_parquet(test_file)
    y_true = df["label"].astype(int).values
    x = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float32")
        .values
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

    # load model
    model = GANomaly(input_dim=x.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # load threshold
    with open(threshold_path) as f:
        threshold = json.load(f)["threshold"]

    # forward
    with torch.no_grad():
        recon, z, z_hat, _, _ = model(x_tensor)
        recon_err = torch.mean((x_tensor - recon) ** 2, dim=1)
        latent_err = torch.mean((z - z_hat) ** 2, dim=1)
        scores = (recon_err + latent_err).cpu().numpy()

    y_pred = (scores > threshold).astype(int)

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, scores)) if len(set(y_true)) > 1 else None
        ),
        "threshold": threshold,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ [{dataset}] GANomaly evaluation complete: {metrics}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features_file", type=str, default=None)
    args = parser.parse_args()

    evaluate_ganomaly(dataset=args.dataset, features_file=args.features_file)
