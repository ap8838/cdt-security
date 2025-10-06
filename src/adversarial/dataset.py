"""
Dataset loader for Conditional GAN.
- Loads processed parquet file.
- Automatically encodes categorical columns using saved encoders (if available),
  or basic LabelEncoding fallback.
- Ensures numeric tensors for generator/discriminator.
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class TabularCGANDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        features_file: str,
        scaler_path: Optional[str] = None,
        encoders_path: Optional[str] = None,
        condition="asset",
        train_on="anomaly",
    ):
        self.df = pd.read_parquet(parquet_path)

        with open(features_file, "r") as f:
            features = json.load(f)
        all_cols = [
            c for c in features["all"] if c not in ("asset_id", "timestamp", "label")
        ]

        # Filter based on label
        if train_on == "anomaly":
            self.df = self.df[self.df["label"] == 1]
        elif train_on == "normal":
            self.df = self.df[self.df["label"] == 0]

        # Load encoders if available
        self.encoders = {}
        if encoders_path and Path(encoders_path).exists():
            try:
                self.encoders = joblib.load(encoders_path)
                print(f"✅ Loaded encoders from {encoders_path}")
            except Exception as e:
                print(f"⚠️ Failed to load encoders from {encoders_path}: {e}")

        # Load scaler if available
        self.scaler = None
        if scaler_path and Path(scaler_path).exists():
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception as e:
                print(f"⚠️ Failed to load scaler: {e}")

        # Prepare dataframe
        X_df = self.df[all_cols].copy()

        # Apply encoders safely
        for col in X_df.columns:
            if X_df[col].dtype == "object" or str(X_df[col].dtype).startswith(
                "category"
            ):
                try:
                    if col in self.encoders and hasattr(
                        self.encoders[col], "transform"
                    ):
                        enc = self.encoders[col]
                        X_df[col] = enc.transform(X_df[col].astype(str))
                    else:
                        # fallback if no encoder exists for this col
                        le = LabelEncoder()
                        X_df[col] = le.fit_transform(X_df[col].astype(str))
                        print(f"⚠️ Used fallback LabelEncoder for {col}")
                except Exception as e:
                    print(f"⚠️ Skipping {col}: could not encode ({e})")
                    X_df[col] = 0.0  # replace with zeros if cannot encode

        # Convert to numpy
        self.cols = list(X_df.columns)
        self.X = X_df.fillna(0).to_numpy(dtype=np.float32)

        # Build condition vector
        if condition == "asset" and "asset_id" in self.df.columns:
            assets = sorted(self.df["asset_id"].fillna("").unique().tolist())
            self.asset_to_idx = {a: i for i, a in enumerate(assets)}
            conds = []
            for a in self.df["asset_id"].fillna("").astype(str):
                vec = np.zeros(len(assets), dtype=np.float32)
                vec[self.asset_to_idx[a]] = 1.0
                conds.append(vec)
            self.conds = np.stack(conds)
        else:
            self.conds = np.zeros((len(self.X), 1), dtype=np.float32)

        # Optional rescale 0-1 → -1..1
        self._map_scale()

    def _map_scale(self):
        if self.X.size == 0:
            return
        mn, mx = float(self.X.min()), float(self.X.max())
        if 0.0 <= mn and mx <= 1.0:
            self.X = (self.X * 2.0) - 1.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.conds[idx])
