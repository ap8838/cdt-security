# scripts/diagnose_ae.py
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.models.autoencoder import Autoencoder

# --- Setup project root ---
ROOT = os.path.dirname(os.path.dirname(__file__))
os.chdir(ROOT)
sys.path.append(ROOT)

# --- Select dataset ---


parser = argparse.ArgumentParser(
    description="Diagnose Autoencoder reconstruction errors."
)
parser.add_argument(
    "--dataset",
    type=str,
    default="iot_fridge",
    help="Dataset name (e.g., iot_fridge, win10, iot_garage)",
)
args = parser.parse_args()
dataset = args.dataset

print(f"\n=== Diagnosing AE for dataset: {dataset} ===")

# --- Load features and data ---
features_path = f"artifacts/preproc/{dataset}_features.json"
test_path = f"data/processed/{dataset}_test.parquet"
model_path = f"artifacts/models/{dataset}_ae.pt"

if not (
    os.path.exists(features_path)
    and os.path.exists(test_path)
    and os.path.exists(model_path)
):
    print(f"❌ Missing files for {dataset}. Check preprocessing or training.")
    sys.exit(1)

features = json.load(open(features_path))
cols = [c for c in features["all"] if c not in ("asset_id", "timestamp", "label")]
df = pd.read_parquet(test_path)

x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32").values
y = df["label"].astype(int).values

# --- Load model ---
device = "cpu"
model = Autoencoder(input_dim=x.shape[1]).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Compute reconstruction errors ---
with torch.no_grad():
    xt = torch.tensor(x, dtype=torch.float32).to(device)
    recon = model(xt).cpu().numpy()

errors = np.mean((x - recon) ** 2, axis=1)
norm_errors = errors[y == 0]
anom_errors = errors[y == 1]

# --- Plot distribution ---
plt.figure(figsize=(8, 4))
plt.hist(norm_errors, bins=200, alpha=0.6, label="normal")
plt.hist(anom_errors, bins=200, alpha=0.6, label="anomaly")
plt.xlim(0, np.percentile(errors, 99))
plt.legend()
plt.title(f"{dataset} AE reconstruction error")
plt.tight_layout()
plt.show()

# --- Print stats ---
print(f"\n[{dataset}] Statistics:")
print(
    "Normal mean/median/max:",
    norm_errors.mean(),
    np.median(norm_errors),
    norm_errors.max(),
)
print(
    "Anomaly mean/median/min:",
    anom_errors.mean(),
    np.median(anom_errors),
    anom_errors.min(),
)
print("Difference (anom - norm mean):", anom_errors.mean() - norm_errors.mean())
