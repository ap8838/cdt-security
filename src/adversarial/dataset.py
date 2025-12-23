"""
Dataset loader for Conditional GAN.

- Loads processed parquet file.
- Assumes preprocessing already encoded categorical columns and created numeric feature columns.
- Builds condition vectors (one-hot) from 'asset' or 'asset_id' column when available.
- Falls back gracefully if `train_on` yields an empty dataset (uses whole DF).
"""

import json
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularCGANDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        features_file: str,
        scaler_path: Optional[str] = None,
        encoders_path: Optional[str] = None,
        condition: str = "asset",
        train_on: str = "anomaly",
    ):
        # Load processed parquet (already encoded & scaled by preprocess.py)
        self.df = pd.read_parquet(parquet_path)

        # Load features.json to know which columns to use (preprocess wrote this)
        with open(features_file, "r") as f:
            features = json.load(f)

        # Exclude all metadata columns from feature matrix
        protected = {"asset_id", "asset", "timestamp", "label"}
        all_cols = [c for c in features["all"] if c not in protected]

        # Keep copy for fallback
        original_df = self.df.copy()

        # Filter rows depending on anomaly/normal/all
        if train_on == "anomaly":
            self.df = self.df[self.df["label"] == 1].copy()
        elif train_on == "normal":
            self.df = self.df[self.df["label"] == 0].copy()

        if len(self.df) == 0:
            print(
                f" TabularCGANDataset: no rows after applying train_on='{train_on}'. Falling back to full dataset."
            )
            self.df = original_df.copy()

        # Ensure all expected columns exist
        missing = [c for c in all_cols if c not in self.df.columns]
        if missing:
            print(
                f" TabularCGANDataset: missing expected feature columns: {missing}. Filling with zeros."
            )
            for c in missing:
                self.df[c] = 0.0

        # Select only numeric columns (drop any accidental string/object)
        X_df = self.df[all_cols].copy()
        X_df = X_df.select_dtypes(include=["number"])
        if X_df.empty:
            raise ValueError(
                " TabularCGANDataset: No numeric columns found after filtering!"
            )

        # Convert to float32 numpy
        self.cols = list(X_df.columns)
        self.X = X_df.fillna(0).to_numpy(dtype=np.float32)

        # -------------------------------
        #  Build conditional vector
        # -------------------------------
        cond_col = None
        if condition and condition in self.df.columns:
            cond_col = condition
        elif "asset" in self.df.columns:
            cond_col = "asset"
        elif "asset_id" in self.df.columns:
            cond_col = "asset_id"

        if cond_col is not None:
            assets = sorted(self.df[cond_col].fillna("").astype(str).unique().tolist())
            if len(assets) == 0:
                print(
                    " TabularCGANDataset: condition column present but empty, using dummy condition."
                )
                self.conds = np.zeros((len(self.X), 1), dtype=np.float32)
            else:
                self.asset_to_idx = {a: i for i, a in enumerate(assets)}
                conds = []
                for a in self.df[cond_col].fillna("").astype(str):
                    vec = np.zeros(len(assets), dtype=np.float32)
                    vec[self.asset_to_idx[a]] = 1.0
                    conds.append(vec)
                self.conds = np.vstack(conds).astype(np.float32)
        else:
            print(
                " TabularCGANDataset: no condition column found — using dummy condition vector."
            )
            self.conds = np.zeros((len(self.X), 1), dtype=np.float32)

        # Optional rescale 0–1 → –1..1 for GAN training stability
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
