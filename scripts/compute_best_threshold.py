# scripts/compute_best_threshold.py
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

ROOT = os.path.dirname(os.path.dirname(__file__))
os.chdir(ROOT)
sys.path.append(ROOT)


def get_datasets():
    # find *_test.parquet names
    import glob

    ds = [
        os.path.basename(p).replace("_test.parquet", "")
        for p in glob.glob("data/processed/*_test.parquet")
    ]
    return sorted(ds)


def load_ae_scores(dataset):
    # compute AE reconstruction MSE per sample
    from src.models.autoencoder import Autoencoder

    features = json.load(open(f"artifacts/preproc/{dataset}_features.json"))
    cols = [c for c in features["all"] if c not in ("asset_id", "timestamp", "label")]
    df = pd.read_parquet(f"data/processed/{dataset}_test.parquet")
    x = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float32")
        .values
    )
    y = df["label"].astype(int).values
    device = "cpu"
    model = Autoencoder(input_dim=x.shape[1]).to(device)
    model.load_state_dict(
        torch.load(f"artifacts/models/{dataset}_ae.pt", map_location=device)
    )
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32).to(device)
        recon = model(xt).cpu().numpy()
    errors = np.mean((x - recon) ** 2, axis=1)
    return errors, y


def load_ganomaly_scores(dataset):
    # compute GANomaly score used in eval_ganomaly (recon_err + latent_err)
    from src.models.ganomaly import GANomaly

    features = json.load(open(f"artifacts/preproc/{dataset}_features.json"))
    cols = [c for c in features["all"] if c not in ("asset_id", "timestamp", "label")]
    df = pd.read_parquet(f"data/processed/{dataset}_test.parquet")
    x = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float32")
        .values
    )
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
        recon_err = torch.mean((xt.to(device) - recon) ** 2, dim=1)
        latent_err = torch.mean((z - z_hat) ** 2, dim=1)
        scores = (recon_err + latent_err).cpu().numpy()
    return scores, y


def choose_threshold(scores, y):
    # default: maximize tpr - fpr
    fpr, tpr, thr = roc_curve(y, scores)
    ix = np.argmax(tpr - fpr)
    return float(thr[ix]), float(tpr[ix]), float(fpr[ix])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all", help="dataset name or all")
    p.add_argument("--model", choices=("ae", "ganomaly"), default="ae")
    args = p.parse_args()

    datasets = [args.dataset] if args.dataset != "all" else get_datasets()
    os.makedirs("artifacts/models", exist_ok=True)

    for ds in datasets:
        print(f"\n=== Processing {ds} ===")
        try:
            if args.model == "ae":
                scores, y = load_ae_scores(ds)
            else:
                scores, y = load_ganomaly_scores(ds)
        except Exception as e:
            print("⚠️  Skipping", ds, "— error loading model/data:", e)
            continue

        if len(np.unique(y)) < 2:
            print("⚠️  No positive class in test set for", ds, "- skipping ROC")
            continue

        best, tpr, fpr = choose_threshold(scores, y)
        fname = f"artifacts/models/{ds}_{args.model}_threshold.json"
        with open(fname, "w") as fh:
            json.dump({"threshold": best}, fh)
        print(
            f"✅ Best threshold {best:.6f} | TPR={tpr:.3f}, FPR={fpr:.3f} -> saved to {fname}"
        )

    print("\n✅ All done — thresholds saved to artifacts/models/*.json")


if __name__ == "__main__":
    main()
