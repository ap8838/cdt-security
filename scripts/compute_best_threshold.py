import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from src.utils.temporal import make_sliding_windows  # Added import

# Add project root to path and import temporal logic
ROOT = os.path.dirname(os.path.dirname(__file__))
os.chdir(ROOT)
sys.path.append(ROOT)

from src.utils.temporal import make_sliding_windows


def get_datasets():
    import glob
    ds = [
        os.path.basename(p).replace("_test.parquet", "")
        for p in glob.glob("data/processed/*_test.parquet")
    ]
    return sorted(ds)


def load_common_data(dataset):
    features_path = f"artifacts/preproc/{dataset}_features.json"
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing features file for {dataset}")

    features = json.load(open(features_path))
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
    return x, y, features


def load_ae_scores(dataset):
    from src.models.autoencoder import Autoencoder
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
    x, y, features = load_common_data(dataset)

    # AE is POINTWISE (No windows)
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


def load_ganomaly_scores(dataset, window=5):
    from src.models.ganomaly import GANomaly
<<<<<<< Updated upstream

    # Load original point-wise data
    x_raw, y_raw, features = load_common_data(dataset)

    #  TEMPORAL WINDOWING (Commit A)
    # Hardcoded window=5 for consistency in v1
    x, y = make_sliding_windows(x_raw, y_raw, window=5)
=======
    x, y, features = load_common_data(dataset)
>>>>>>> Stashed changes

    # Apply sliding window logic to match GANomaly V1 training
    if window > 1:
        x, y = make_sliding_windows(x, y, window=window)

    device = "cpu"
    # input_dim is automatically handled by the windowed shape[1]
    model = GANomaly(input_dim=x.shape[1]).to(device)
    model.load_state_dict(
        torch.load(f"artifacts/models/{dataset}_ganomaly.pt", map_location=device)
    )
    model.eval()

    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32).to(device)
        recon, z, z_hat, _, _ = model(xt)
<<<<<<< Updated upstream

        # Calculate scores using windowed reconstruction and latent error
        recon_err = torch.mean((xt.to(device) - recon) ** 2, dim=1)
=======
        recon_err = torch.mean((xt - recon) ** 2, dim=1)
>>>>>>> Stashed changes
        latent_err = torch.mean((z - z_hat) ** 2, dim=1)
        scores = (recon_err + latent_err).cpu().numpy()

    return scores, y


def choose_threshold_roc(scores, y):
    fpr, tpr, thr = roc_curve(y, scores)
    ix = np.argmax(tpr - fpr)
    return float(thr[ix]), float(tpr[ix]), float(fpr[ix])


def choose_threshold_f1(scores, y):
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    ix = np.argmax(f1)
    # Precision-recall curve thresholds are 1 shorter than prec/rec
    safe_index = min(int(ix), int(len(thr) - 1))
    best_thr = float(thr[safe_index])

    preds = (scores >= best_thr).astype(int)
    tpr = np.mean(preds[y == 1]) if any(y == 1) else 0.0
    fpr = np.mean(preds[y == 0])
    return best_thr, float(tpr), float(fpr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all", help="dataset name or all")
    p.add_argument("--model", choices=("ae", "ganomaly"), default="ae")
    p.add_argument("--window", type=int, default=5, help="Window size (GANomaly only)")
    args = p.parse_args()

    datasets = sorted(get_datasets()) if args.dataset == "all" else [args.dataset]
    os.makedirs("artifacts/models", exist_ok=True)

    for ds in datasets:
        print(f"\n=== Processing {ds} ({args.model}) ===")
        try:
            if args.model == "ae":
                # AE remains pointwise
                scores, y = load_ae_scores(ds)
                best, tpr, fpr = choose_threshold_roc(scores, y)
                method = "ROC"
            else:
                # GANomaly uses windows
                scores, y = load_ganomaly_scores(ds, window=args.window)
                best, tpr, fpr = choose_threshold_f1(scores, y)
                method = "Max-F1"
        except Exception as e:
            print(f"  Skipping {ds} — error: {e}")
            continue

        if len(np.unique(y)) < 2:
            print(f"  No positive class in test set for {ds} - skipping threshold optimization")
            continue

        fname = f"artifacts/models/{ds}_{args.model}_threshold.json"
        threshold_value = float(best)
        with open(fname, "w") as fh:
            json.dump({"threshold": threshold_value}, fh)

        print(f" [{method}] Threshold {threshold_value:.6f} | TPR={tpr:.3f}, FPR={fpr:.3f} -> {fname}")

    print("\n All done — thresholds saved to artifacts/models/*.json")


if __name__ == "__main__":
    main()