# src/service/infer_service.py
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
        dataset: dataset name (e.g. 'iot_fridge')
        model_type: 'ae' | 'ganomaly'
        """
        self.dataset = dataset
        self.model_type = model_type.lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        base = Path("artifacts")
        self.features_file = Path(f"artifacts/preproc/{dataset}_features.json")
        self.encoders_file = Path(f"artifacts/preproc/{dataset}_encoders.pkl")
        self.scaler_file = Path(f"artifacts/preproc/{dataset}_scaler.pkl")

        # sanity checks
        for p in (self.features_file, self.encoders_file, self.scaler_file):
            if not p.exists():
                raise FileNotFoundError(f"Required preproc file not found: {p}")

        # Choose model + threshold
        if self.model_type == "ae":
            self.model_path = base / f"models/{dataset}_ae.pt"
            self.threshold_file = base / f"models/{dataset}_threshold.json"
            self.model = Autoencoder(input_dim=self._get_input_dim()).to(self.device)
        elif self.model_type == "ganomaly":
            self.model_path = base / f"models/{dataset}_ganomaly.pt"
            self.threshold_file = base / f"models/{dataset}_ganomaly_threshold.json"
            self.model = GANomaly(input_dim=self._get_input_dim()).to(self.device)
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        # load feature info
        with open(self.features_file, "r") as f:
            features = json.load(f)
        self.cols = [
            c for c in features["all"] if c not in ("asset_id", "timestamp", "label")
        ]

        # load preproc objects
        self.encoders = joblib.load(self.encoders_file)
        self.scaler = joblib.load(self.scaler_file)

        # load model weights + threshold
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()

        if not self.threshold_file.exists():
            raise FileNotFoundError(f"Threshold file not found: {self.threshold_file}")
        with open(self.threshold_file, "r") as f:
            self.threshold = json.load(f)["threshold"]

    def _get_input_dim(self) -> int:
        """Infer feature input dimension from features.json."""
        with open(self.features_file) as f:
            data = json.load(f)
        return len(
            [c for c in data["all"] if c not in ("asset_id", "timestamp", "label")]
        )

    def _apply_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply encoders safely, supporting sparse/dense outputs."""
        for col, (enc_type, encoder) in self.encoders.items():
            if col not in df.columns:
                continue
            if enc_type == "onehot":
                # ensure string and strip
                df[col] = df[col].astype(str).str.strip()
                transformed = encoder.transform(df[[col]])
                if hasattr(transformed, "toarray"):
                    transformed = transformed.toarray()
                # feature name handling fallback
                get_names = getattr(encoder, "get_feature_names_out", None)
                if callable(get_names):
                    new_cols = get_names([col])
                else:
                    # fallback create names 0..N-1
                    new_cols = [f"{col}_{i}" for i in range(transformed.shape[1])]
                # attach to df
                transformed_df = pd.DataFrame(
                    transformed, index=df.index, columns=new_cols
                )
                df = pd.concat([df.drop(columns=[col]), transformed_df], axis=1)
            elif enc_type == "freq":
                # encoder is a mapping dict
                df[f"{col}_freq"] = df[col].astype(str).map(encoder).fillna(0)
                df = df.drop(columns=[col])
            else:
                # unknown encoder type: try to call transform and put back single column
                try:
                    df[col] = encoder.transform(df[[col]])
                except Exception:
                    # leave as-is
                    pass
        return df

    def _apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaler safely. If scaler doesn't expose feature_names_in_, assume columns align."""
        if hasattr(self.scaler, "feature_names_in_"):
            expected = list(self.scaler.feature_names_in_)
            # ensure expected cols exist
            for col in expected:
                if col not in df.columns:
                    df[col] = 0.0
            # select and transform (ensuring float)
            subset = df[expected].astype(float)
            transformed = self.scaler.transform(subset)
            df[expected] = transformed
        else:
            # fallback: try to transform all numeric columns (if scaler can transform)
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                try:
                    transformed = self.scaler.transform(df[numeric_cols].astype(float))
                    df[numeric_cols] = transformed
                except Exception:
                    # give up and leave as-is
                    pass
        return df

    def preprocess(self, event: dict) -> torch.Tensor:
        """Apply encoding + scaling to match training format."""
        # flatten
        flat = event.get("features", event)
        flat = {
            "asset_id": event.get("asset_id"),
            "timestamp": event.get("timestamp"),
            **flat,
        }
        df = pd.DataFrame([flat])

        # Apply encoders
        df = self._apply_encoders(df)

        # Ensure numeric columns present before scaling
        df = self._apply_scaler(df)

        # ensure required columns exist (fill missing with 0.0)
        missing = [c for c in self.cols if c not in df.columns]
        if missing:
            zeros = pd.DataFrame(0.0, index=df.index, columns=missing)
            df = pd.concat([df, zeros], axis=1)

        # reorder columns and force float32
        df = df.reindex(columns=self.cols, fill_value=0.0)
        # coerce everything to float32 - this avoids numpy.object_ errors
        for c in df.columns:
            # if any non-numeric values sneaked in, coerce to numeric with NaN->0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float32")

        arr = df.values.astype("float32")
        return torch.tensor(arr, dtype=torch.float32).to(self.device)

    def score(self, event: dict):
        """Run single-event inference and return anomaly info."""
        x_tensor = self.preprocess(event)
        with torch.no_grad():
            if self.model_type == "ae":
                recon = self.model(x_tensor)
                score = torch.mean((x_tensor - recon) ** 2, dim=1).item()
            else:  # ganomaly
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
