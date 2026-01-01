import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from src.utils.seed import set_seed
from .autoencoder import Autoencoder


def evaluate_autoencoder(dataset, features_file=None, seed=42):
    set_seed(seed)
    test_file = f"data/processed/{dataset}_test.parquet"
    model_path = f"artifacts/models/{dataset}_ae.pt"
    threshold_path = f"artifacts/models/{dataset}_threshold.json"
    report_path = f"artifacts/reports/{dataset}_ae_eval.json"

    if not os.path.exists(test_file):
        print(f"⚠️ Skipping {dataset}: test parquet not found.")
        return

    if features_file is None:
        features_file = f"artifacts/preproc/{dataset}_features.json"

    with open(features_file) as f:
        features = json.load(f)
    cols = [c for c in features["all"] if c not in ("asset_id", "timestamp", "label")]

    df = pd.read_parquet(test_file)
    x = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float32")
        .values
    )
    y_true = df["label"].values

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder(input_dim=x.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(threshold_path) as f:
        threshold = json.load(f)["threshold"]

    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        recon = model(x_tensor)
        errors = torch.mean((x_tensor - recon) ** 2, dim=1).cpu().numpy()

    y_pred = (errors > threshold).astype(int)

    # Check if we have both classes (0 and 1) to calculate ROC AUC
    has_both_classes = len(np.unique(y_true)) > 1

    report = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, errors)) if has_both_classes else None,
        "threshold": threshold,
        "test_samples": len(y_true),
        "test_anomalies": int(np.sum(y_true))
    }

    if not has_both_classes:
        print(f" [!] {dataset}: Test set contains ONLY class {np.unique(y_true)}. ROC-AUC is undefined.")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f" [{dataset}] AE evaluation complete:", report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all")
    p.add_argument("--features_file", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.dataset == "all":
        for f in glob.glob("data/processed/*_test.parquet"):
            dataset = os.path.basename(f).replace("_test.parquet", "")
            evaluate_autoencoder(
                dataset, features_file=args.features_file, seed=args.seed
            )
    else:
        evaluate_autoencoder(
            args.dataset, features_file=args.features_file, seed=args.seed
        )


if __name__ == "__main__":
    main()
