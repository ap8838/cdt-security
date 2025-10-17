"""
Generate synthetic samples from trained cGAN and optionally POST them to API
(so streamer/score pipeline can be tested).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

from src.adversarial.cgan_model import Generator


def load_generator(path, device):
    ckpt = torch.load(path, map_location=device)
    info = {
        "z_dim": ckpt["z_dim"],
        "cond_dim": ckpt["cond_dim"],
        "D_in": ckpt["D_in"],
    }
    gen = Generator(
        z_dim=info["z_dim"], cond_dim=info["cond_dim"], out_dim=info["D_in"], hidden=256
    )
    gen.load_state_dict(ckpt["gen_state"])
    gen.to(device).eval()
    return gen, info


def generate(parsed_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen, info = load_generator(parsed_args.model, device)

    # load features.json to know cols
    with open(parsed_args.features, "r") as f:
        features = json.load(f)

    # Build columns the same way TabularCGANDataset expects:
    # exclude metadata: asset_id, asset, timestamp, label
    cols = [
        c
        for c in features["all"]
        if c not in ("asset_id", "asset", "timestamp", "label")
    ]

    expected_dim = int(info["D_in"])
    if len(cols) != expected_dim:
        print(
            f"⚠️ generate_samples: mismatch between features.json cols ({len(cols)}) "
            f"and model.D_in ({expected_dim}). Attempting to reconcile."
        )
        if len(cols) > expected_dim:
            # Trim extra column names (preserve order)
            print(f"⚠️ Trimming {len(cols) - expected_dim} columns from features list.")
            cols = cols[:expected_dim]
        else:
            # Pad with synthetic column names so DataFrame creation succeeds
            miss = expected_dim - len(cols)
            pad_names = [f"gen_feature_pad_{i}" for i in range(miss)]
            print(f"⚠️ Appending {miss} placeholder feature names: {pad_names}")
            cols = cols + pad_names

    # simple condition strategy: use asset list from dataset parquet if present
    if parsed_args.cond_asset:
        df_cond = pd.read_parquet(parsed_args.cond_asset)
        assets = sorted(df_cond["asset_id"].fillna("").unique().tolist())
        if parsed_args.asset not in assets:
            raise RuntimeError("asset not present in provided cond_asset parquet")
        cond_dim = len(assets)
        asset_idx = assets.index(parsed_args.asset)
        cond = np.zeros((parsed_args.n, cond_dim), dtype=np.float32)
        cond[:, asset_idx] = 1.0
    else:
        # Use generator's cond_dim from the checkpoint
        cond = np.zeros((parsed_args.n, info["cond_dim"]), dtype=np.float32)

    z = torch.randn(parsed_args.n, info["z_dim"], device=device)
    cond_t = torch.from_numpy(cond).to(device)

    with torch.no_grad():
        fake = gen(z, cond_t).cpu().numpy()

    # Debug print
    print(f"Generated raw fake shape: {fake.shape}  (expected columns = {len(cols)})")

    # map back from [-1,1] -> [0,1] if necessary (heuristic)
    if fake.min() >= -1.0 and fake.max() <= 1.0:
        fake = (fake + 1.0) / 2.0

    # Ensure shape matches columns length before constructing DataFrame
    if fake.shape[1] != len(cols):
        # last-resort safety: try trim/pad on the *generated* array to match cols
        if fake.shape[1] > len(cols):
            print(
                f"⚠️ Trimming generated values from {fake.shape[1]} -> {len(cols)} to match column names."
            )
            fake = fake[:, : len(cols)]
        elif fake.shape[1] < len(cols):
            print(
                f"⚠️ Padding generated values from {fake.shape[1]} -> {len(cols)} with zeros."
            )
            pad = np.zeros((fake.shape[0], len(cols) - fake.shape[1]), dtype=fake.dtype)
            fake = np.hstack([fake, pad])

    df_out = pd.DataFrame(fake, columns=cols)
    df_out["asset_id"] = parsed_args.asset if parsed_args.asset else "synthetic"
    df_out["timestamp"] = datetime.now(timezone.utc).isoformat()
    df_out["label"] = 1  # synthetic anomalies

    out_path = Path(parsed_args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    print("Wrote", out_path)

    if parsed_args.post:
        for _, row in df_out.iterrows():
            payload = {
                "asset_id": row["asset_id"],
                "timestamp": row["timestamp"],
                "features": {c: float(row[c]) for c in cols},
            }
            try:
                resp = requests.post(parsed_args.post, json=payload, timeout=5.0)
                print("POST", resp.status_code)
            except Exception as e:
                print("POST failed:", e)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", default="artifacts/adversarial/generated.parquet")
    p.add_argument("--asset", default="synthetic")
    p.add_argument(
        "--cond-asset",
        help="path to parquet to extract asset list for conditioning (optional)",
    )
    p.add_argument("--post", help="POST endpoint to send generated events (optional)")
    args = p.parse_args()
    generate(args)
