# scripts/diagnose_ganomaly.py
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.models.ganomaly import GANomaly

ROOT = os.path.dirname(os.path.dirname(__file__))
os.chdir(ROOT)
sys.path.append(ROOT)


dataset = "iot_motion"  # <- change per dataset

feat = json.load(open(f"artifacts/preproc/{dataset}_features.json"))
cols = [c for c in feat["all"] if c not in ("asset_id", "timestamp", "label")]
df = pd.read_parquet(f"data/processed/{dataset}_test.parquet")
x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32").values
y = df["label"].astype(int).values

device = "cpu"
model = GANomaly(input_dim=x.shape[1]).to(device)
model.load_state_dict(
    torch.load(f"artifacts/models/{dataset}_ganomaly.pt", map_location=device)
)
model.eval()

with torch.no_grad():
    xt = torch.tensor(x, dtype=torch.float32).to(device)
    recon, z, z_hat, _, _ = model(xt)
    recon_err = torch.mean((xt - recon) ** 2, dim=1).cpu().numpy()
    latent_err = torch.mean((z - z_hat) ** 2, dim=1).cpu().numpy()
    scores = recon_err + latent_err

# plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(recon_err[y == 0], bins=200, alpha=0.6, label="norm")
plt.hist(recon_err[y == 1], bins=200, alpha=0.6, label="anom")
plt.title("recon_err")
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(latent_err[y == 0], bins=200, alpha=0.6, label="norm")
plt.hist(latent_err[y == 1], bins=200, alpha=0.6, label="anom")
plt.title("latent_err")
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(scores[y == 0], bins=200, alpha=0.6, label="norm")
plt.hist(scores[y == 1], bins=200, alpha=0.6, label="anom")
plt.title("total score")
plt.legend()

plt.show()

print("recon mean (norm/anom):", recon_err[y == 0].mean(), recon_err[y == 1].mean())
print("latent mean (norm/anom):", latent_err[y == 0].mean(), latent_err[y == 1].mean())
print("score mean (norm/anom):", scores[y == 0].mean(), scores[y == 1].mean())
