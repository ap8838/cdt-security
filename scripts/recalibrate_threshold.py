import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve

from src.models.autoencoder import Autoencoder

dataset = "iot_motion"
features = json.load(open(f"artifacts/preproc/{dataset}_features.json"))
cols = [c for c in features["all"] if c not in ("asset_id", "timestamp", "label")]
df = pd.read_parquet(f"data/processed/{dataset}_test.parquet")
x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32").values
y = df["label"].astype(int).values

device = "cpu"
model = Autoencoder(input_dim=x.shape[1]).to(device)
model.load_state_dict(
    torch.load(f"artifacts/models/{dataset}_ae.pt", map_location=device)
)
model.eval()

with torch.no_grad():
    recon = model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()

errors = np.mean((x - recon) ** 2, axis=1)
prec, rec, thr = precision_recall_curve(y, errors)
f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
ix = np.nanargmax(f1)
best_thr = float(thr[ix])

print(
    f"✅ F1-optimal threshold: {best_thr:.6f} (F1={f1[ix]:.4f}, Precision={prec[ix]:.4f}, Recall={rec[ix]:.4f})"
)

# save it
os.makedirs("artifacts/models", exist_ok=True)
with open(f"artifacts/models/{dataset}_threshold.json", "w") as fh:
    json.dump({"threshold": best_thr}, fh)
