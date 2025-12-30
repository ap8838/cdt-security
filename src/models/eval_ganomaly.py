import argparse
import glob
import json
import os
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from src.models.ganomaly import GANomaly
from src.utils.temporal import make_sliding_windows  # Added import


def evaluate_ganomaly(dataset, features_file=None, window=1):
    test_file = f"data/processed/{dataset}_test.parquet"
    if not os.path.exists(test_file):
        print(f" Skipping {dataset}: no test parquet.")
        return

    model_path = f"artifacts/models/{dataset}_ganomaly.pt"
    threshold_path = f"artifacts/models/{dataset}_ganomaly_threshold.json"
    report_path = f"artifacts/reports/{dataset}_ganomaly_eval.json"

    if features_file is None:
        features_file = f"artifacts/preproc/{dataset}_features.json"

    with open(features_file) as f:
        features = json.load(f)
    cols = [c for c in features["all"] if c not in ("asset_id", "timestamp", "label")]

    df = pd.read_parquet(test_file)

    # --- Modified Data Loading & Windowing ---
    x = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float32")
        .values
    )
    y = df["label"].astype(int).values

    #  TEMPORAL WINDOWING
    x, y_true = make_sliding_windows(x, y, window=window)
    # ------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The input_dim must match the training window (x.shape[1] handles this)
    model = GANomaly(input_dim=x.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(threshold_path) as f:
        threshold = json.load(f)["threshold"]

    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        recon, z, z_hat, _, _ = model(x_tensor)

        # Calculate scores based on reconstruction and latent error
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
        "window_size": window
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f" [{dataset}] GANomaly evaluation complete: {metrics}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all")
    p.add_argument("--features_file", default=None)
    # Added window argument
    p.add_argument(
        "--window",
        type=int,
        default=1,
        help="Temporal sliding window size (must match training window)",
    )
    args = p.parse_args()

    if args.dataset == "all":
        for f in glob.glob("data/processed/*_test.parquet"):
            dataset = os.path.basename(f).replace("_test.parquet", "")
            evaluate_ganomaly(dataset, features_file=args.features_file, window=args.window)
    else:
        evaluate_ganomaly(args.dataset, features_file=args.features_file, window=args.window)


if __name__ == "__main__":
    main()