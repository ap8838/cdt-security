import argparse
import glob
import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from src.models.autoencoder import Autoencoder
from src.utils.seed import set_seed


def infer_event(dataset: str, event_dict: dict, seed=42):
    """Score a single JSON event for anomaly detection with proper preprocessing."""
    set_seed(seed)

    # paths
    model_path = f"artifacts/models/{dataset}_ae.pt"
    threshold_path = f"artifacts/models/{dataset}_threshold.json"
    features_file = f"artifacts/preproc/{dataset}_features.json"
    encoders_file = f"artifacts/preproc/{dataset}_encoders.pkl"
    scaler_file = f"artifacts/preproc/{dataset}_scaler.pkl"

    # load features (drop metadata cols!)
    with open(features_file, encoding="utf-8") as feat_file:
        features = json.load(feat_file)
    cols = [c for c in features["all"] if c not in ("asset_id", "timestamp", "label")]

    # load encoder + scaler
    encoders = joblib.load(encoders_file)
    scaler = joblib.load(scaler_file)

    # build dataframe
    df = pd.DataFrame([event_dict])

    # apply encoders (categorical -> one-hot or freq)
    for col, (enc_type, encoder) in encoders.items():
        if col in df.columns:
            if enc_type == "onehot":
                df[col] = df[col].astype(str)
                transformed = encoder.transform(df[[col]])
                new_cols = encoder.get_feature_names_out([col])
                df[new_cols] = transformed
                df.drop(columns=[col], inplace=True)
            elif enc_type == "freq":
                df[f"{col}_freq"] = df[col].astype(str).map(encoder).fillna(0)
                df.drop(columns=[col], inplace=True)

    # apply scaler (numeric -> scaled)
    numeric_cols = scaler.feature_names_in_
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # align to feature order
    for col in cols:
        if col not in df.columns:
            df[col] = 0
    df = df[cols]

    # numpy array
    x = df.astype("float32").values

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder(input_dim=x.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # load threshold
    with open(threshold_path, encoding="utf-8") as thr_file:
        threshold = json.load(thr_file)["threshold"]

    # forward pass
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        recon = model(x_tensor)
        error = torch.mean((x_tensor - recon) ** 2, dim=1).cpu().item()

    return {
        "dataset": dataset,
        "score": float(error),
        "threshold": float(threshold),
        "is_anomaly": bool(error > threshold),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", help="Dataset name (e.g. iot_fridge). If omitted, runs all."
    )
    parser.add_argument(
        "--event", required=True, help="Either JSON string or path to JSON file."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    event_input = args.event.strip()
    event_path = Path(event_input)

    if event_path.is_file():
        with event_path.open(encoding="utf-8") as ev_file:
            payload = json.load(ev_file)
    else:
        if event_input.startswith("'") and event_input.endswith("'"):
            event_input = event_input[1:-1]
        event_input = event_input.replace("'", '"')
        payload = json.loads(event_input)

    datasets = []
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [
            Path(f).stem.replace("_ae", "")
            for f in glob.glob("artifacts/models/*_ae.pt")
        ]

    for ds in datasets:
        result_dict = infer_event(ds, payload, seed=args.seed)
        print(json.dumps(result_dict, indent=2))
