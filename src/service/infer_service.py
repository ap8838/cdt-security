import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from src.models.autoencoder import Autoencoder
from src.models.ganomaly import GANomaly


class InferenceService:
    def __init__(self, dataset: str, model_type: str = "ae"):
        """
        dataset: dataset name, e.g., 'iot_fridge'
        model_type: 'ae' (Autoencoder) or 'ganomaly'
        """
        self.dataset = dataset
        self.model_type = model_type.lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # paths
        base = Path("artifacts")
        self.features_file = Path(f"artifacts/preproc/{dataset}_features.json")
        self.encoders_file = Path(f"artifacts/preproc/{dataset}_encoders.pkl")
        self.scaler_file = Path(f"artifacts/preproc/{dataset}_scaler.pkl")

        if self.model_type == "ae":
            self.model_path = base / f"models/{dataset}_ae.pt"
            self.threshold_file = base / f"models/{dataset}_threshold.json"
            self.model = Autoencoder(input_dim=self._get_input_dim()).to(self.device)

        elif self.model_type == "ganomaly":
            self.model_path = base / f"models/{dataset}_ganomaly.pt"
            self.threshold_file = base / f"models/{dataset}_ganomaly_threshold.json"
            self.model = GANomaly(input_dim=self._get_input_dim()).to(self.device)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # load preprocessing
        with open(self.features_file, "r") as f:
            features = json.load(f)
        self.cols = [
            c for c in features["all"] if c not in ("asset_id", "timestamp", "label")
        ]

        self.encoders = joblib.load(self.encoders_file)
        self.scaler = joblib.load(self.scaler_file)

        # load trained model + threshold
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()

        with open(self.threshold_file, "r") as f:
            self.threshold = json.load(f)["threshold"]

    def _get_input_dim(self) -> int:
        """Figure out feature dimension from features.json."""
        with open(self.features_file) as f:
            features = json.load(f)
        cols = [
            c for c in features["all"] if c not in ("asset_id", "timestamp", "label")
        ]
        return len(cols)

    def preprocess(self, event: dict) -> torch.Tensor:
        # Flatten features
        if "features" in event:
            flat = {
                "asset_id": event.get("asset_id"),
                "timestamp": event.get("timestamp"),
                **event["features"],
            }
        else:
            flat = event
        df = pd.DataFrame([flat])

        # Apply encoders
        for col, (enc_type, encoder) in self.encoders.items():
            if col in df.columns:
                if enc_type == "onehot":
                    df[col] = df[col].astype(str).str.strip()
                    transformed = encoder.transform(df[[col]])
                    if hasattr(transformed, "toarray"):
                        transformed = transformed.toarray()
                    try:
                        new_cols = encoder.get_feature_names_out([col])
                    except AttributeError:  # for older sklearn
                        new_cols = [f"{col}_{i}" for i in range(transformed.shape[1])]
                    df[new_cols] = transformed
                    df.drop(columns=[col], inplace=True)
                elif enc_type == "freq":
                    df[f"{col}_freq"] = (
                        df[col].astype(str).str.strip().map(encoder).fillna(0)
                    )
                    df.drop(columns=[col], inplace=True)

        # Apply scaler (robust)
        if hasattr(self.scaler, "feature_names_in_"):
            for col in self.scaler.feature_names_in_:
                if col not in df.columns:
                    df[col] = 0
            df[list(self.scaler.feature_names_in_)] = self.scaler.transform(
                df[list(self.scaler.feature_names_in_)]
            )
        elif isinstance(self.scaler, dict) and "feature_names" in self.scaler:
            for col in self.scaler.get("feature_names", []):
                if col not in df.columns:
                    df[col] = 0

        # Fill missing expected columns
        for col in self.cols:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.cols]

        return torch.tensor(df.astype("float32").values, dtype=torch.float32).to(
            self.device
        )

    def score(self, event: dict):
        """Run model inference on single event."""
        x_tensor = self.preprocess(event)

        with torch.no_grad():
            if self.model_type == "ae":
                recon = self.model(x_tensor)
                error = torch.mean((x_tensor - recon) ** 2, dim=1).item()
                score = error

            elif self.model_type == "ganomaly":
                recon, z, z_hat, _, _ = self.model(x_tensor)
                recon_err = torch.mean((x_tensor - recon) ** 2, dim=1)
                latent_err = torch.mean((z - z_hat) ** 2, dim=1)
                score = (recon_err + latent_err).item()

        return {
            "asset_id": event.get("asset_id"),
            "ts": event.get("timestamp"),
            "score": float(score),
            "threshold": float(self.threshold),
            "is_anomaly": bool(score > self.threshold),
            "model": self.model_type,
        }
