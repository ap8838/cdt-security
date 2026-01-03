import argparse
import json
import logging
import os
import traceback
from typing import Dict
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def make_onehot(**kwargs):
    try:
        return OneHotEncoder(**kwargs)
    except TypeError:
        if "sparse_output" in kwargs:
            v = kwargs.pop("sparse_output")
            kwargs["sparse"] = v
        return OneHotEncoder(**kwargs)


def _is_numeric_like(series: pd.Series, thresh: float = 0.9) -> bool:
    """Return True if ≥ thresh fraction of values parse as numeric."""
    coerced = pd.to_numeric(series, errors="coerce")
    if len(coerced) == 0:
        return False
    return coerced.notna().mean() >= thresh


def preprocess_dataset(
    csv_path: str,
    dataset_name: str,
    output_dir: str = "data/processed",
    artifacts_dir: str = "artifacts/preproc",
) -> Dict[str, str]:
    """
    Preprocess a single CSV file and produce:
      - data/processed/{dataset}_train.parquet (includes anomalies)
      - data/processed/{dataset}_test.parquet
      - artifacts/preproc/{dataset}_scaler.pkl
      - artifacts/preproc/{dataset}_encoders.pkl
      - artifacts/preproc/{dataset}_features.json
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    logging.info("Processing %s -> %s", csv_path, dataset_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    try:
        # 1. Load dataset
        df = pd.read_csv(csv_path, low_memory=False)

        # Clean up string columns
        for c in df.select_dtypes(include=["object", "string"]).columns:
            df[c] = df[c].astype(str).str.strip()

        # 2. Create timestamp column
        if "date" in df.columns and "time" in df.columns:
            combined = (
                df["date"].astype(str).str.strip()
                + " "
                + df["time"].astype(str).str.strip()
            )
            try:
                df["timestamp"] = pd.to_datetime(
                    combined, format="%d-%b-%y %H:%M:%S", errors="coerce"
                )
                if df["timestamp"].isna().any():
                    df["timestamp"] = pd.to_datetime(combined, errors="coerce")
            except (ValueError, TypeError):
                df["timestamp"] = pd.to_datetime(combined, errors="coerce")
            df.drop(columns=["date", "time"], inplace=True, errors="ignore")
        elif "ts" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
            df.drop(columns=["ts"], inplace=True, errors="ignore")

        # 3. Add dataset identifier
        df["asset_id"] = dataset_name

        # 3.5 Add "asset" column (used for cGAN conditioning)
        df["asset"] = dataset_name

        # 4. Require label
        label_col = "label"
        if label_col not in df.columns:
            raise ValueError(f"Expected a '{label_col}' column in {csv_path}")

        # 5. Determine numeric vs categorical robustly
        protected = {"asset_id", "asset", "timestamp", label_col}
        numeric_cols = []
        cat_candidates = []
        for c in df.columns:
            if c in protected:
                continue
            ser = df[c]
            if pd.api.types.is_numeric_dtype(ser):
                numeric_cols.append(c)
            elif pd.api.types.is_bool_dtype(ser):
                cat_candidates.append(c)
            elif _is_numeric_like(ser, 0.9):
                df[c] = pd.to_numeric(ser, errors="coerce")
                numeric_cols.append(c)
            else:
                cat_candidates.append(c)

<<<<<<< Updated upstream
        logging.info("Numeric cols: %s", numeric_cols)
        logging.info("Categorical candidates: %s", cat_candidates)
=======
        # --- Version 1: Linux-only chronological split fix ---
        if dataset_name.startswith("linux"):
            split_ratio = 0.6
        else:
            split_ratio = 0.8

        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
>>>>>>> Stashed changes

        # 6. Fit scaler on normal rows only (label=0)
        normal_df = df[df[label_col] == 0].copy()
        if numeric_cols:
            scaler = MinMaxScaler()
            scaler.fit(normal_df[numeric_cols])
            df[numeric_cols] = scaler.transform(df[numeric_cols])
        else:
            scaler = {"feature_names": []}

        # 7. Encode categoricals
        encoded_parts = []
        encoders = {}
        encoded_cols = []

        for col in cat_candidates:
            if col not in df.columns:
                continue
            train_values = normal_df[col].fillna("").astype(str).str.strip()
            unique_vals = train_values.nunique()

            if unique_vals <= 50:
                ohe = make_onehot(sparse_output=False, handle_unknown="ignore")
                ohe.fit(train_values.to_frame())
                transformed = ohe.transform(
                    df[col].fillna("").astype(str).str.strip().to_frame()
                )
                if hasattr(transformed, "toarray"):
                    transformed = transformed.toarray()
                cats = [str(x) for x in ohe.categories_[0]]
                col_names = [f"{col}_{c}" for c in cats]
                df_encoded = pd.DataFrame(
                    transformed, columns=col_names, index=df.index
                )
                encoded_parts.append(df_encoded)
                encoders[col] = ("onehot", ohe)
                encoded_cols.extend(col_names)
            else:
                freq_map = train_values.value_counts(normalize=True).to_dict()
                encoded_series = (
                    df[col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .map(freq_map)
                    .fillna(0)
                    .rename(f"{col}_freq")
                )
                encoded_parts.append(encoded_series.to_frame())
                encoders[col] = ("freq", freq_map)
                encoded_cols.append(f"{col}_freq")

        # 8. Re-assemble DataFrame
        non_numeric_cols = [
            c for c in df.columns if df[c].dtype == "object" and c not in protected
        ]
        df = df.drop(columns=non_numeric_cols, errors="ignore")

        if encoded_parts:
            df = pd.concat([df] + encoded_parts, axis=1)

        df = df.fillna(0)

        # 9. Split train/test (this time train includes anomalies)
        train = df.copy()
        test = df.copy()

        logging.info("Train class distribution:\n%s", train[label_col].value_counts())
        logging.info("Test class distribution:\n%s", test[label_col].value_counts())

        # 10. Save to parquet
        cols_order = ["asset_id", "asset", "timestamp"] + [
            c for c in df.columns if c not in ("asset_id", "asset", "timestamp")
        ]
        train = train[cols_order]
        test = test[cols_order]

        train_path = os.path.join(output_dir, f"{dataset_name}_train.parquet")
        test_path = os.path.join(output_dir, f"{dataset_name}_test.parquet")
        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)

        # Save preprocessing artifacts
        joblib.dump(scaler, os.path.join(artifacts_dir, f"{dataset_name}_scaler.pkl"))
        joblib.dump(
            encoders, os.path.join(artifacts_dir, f"{dataset_name}_encoders.pkl")
        )

        feature_list = {
            "numeric": numeric_cols,
            "categorical": cat_candidates,
            "encoded": encoded_cols,
            "all": [c for c in df.columns if c != label_col],
        }
        with open(
            os.path.join(artifacts_dir, f"{dataset_name}_features.json"), "w"
        ) as fh:
            json.dump(feature_list, fh, indent=2)

        logging.info("%s processed", dataset_name)
        logging.info("Train: %s, Test: %s", train.shape, test.shape)

        return {
            "train_path": train_path,
            "test_path": test_path,
            "features_file": os.path.join(
                artifacts_dir, f"{dataset_name}_features.json"
            ),
        }

    except Exception as e:
        logging.error("Failed processing %s: %s", dataset_name, e)
        logging.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Process a single dataset id (e.g. iot_fridge)",
    )
    args = parser.parse_args()

    datasets = {
        # IoT
        "iot_fridge": "data/raw/IoT_Fridge.csv",
        "iot_garage": "data/raw/IoT_Garage_Door.csv",
        "iot_gps": "data/raw/IoT_GPS_Tracker.csv",
        "iot_modbus": "data/raw/IoT_Modbus.csv",
        "iot_motion": "data/raw/IoT_Motion_Light.csv",
        "iot_thermo": "data/raw/IoT_Thermostat.csv",
        "iot_weather": "data/raw/IoT_Weather.csv",
        # Linux
        "linux_disk1": "data/raw/linux_disk_1.csv",
        "linux_disk2": "data/raw/linux_disk_2.csv",
        "linux_mem1": "data/raw/linux_memory1.csv",
        "linux_mem2": "data/raw/linux_memory2.csv",
        "linux_proc1": "data/raw/linux_process_1.csv",
        "linux_proc2": "data/raw/linux_process_2.csv",
        # Windows
        "win7": "data/raw/windows7_dataset.csv",
        "win10": "data/raw/windows10_dataset.csv",
    }

    if args.dataset:
        if args.dataset not in datasets:
            logging.error(
                "Unknown dataset id '%s'. Available: %s",
                args.dataset,
                list(datasets.keys()),
            )
        else:
            path = datasets[args.dataset]
            if not os.path.exists(path):
                logging.error("File not found: %s", path)
            else:
                preprocess_dataset(path, args.dataset)
    else:
        for dataset_id, path in datasets.items():
            if os.path.exists(path):
                try:
                    preprocess_dataset(path, dataset_id)
                except Exception as exc:
                    logging.error("Processing %s failed: %s", dataset_id, exc)
            else:
                logging.warning("Skipping %s (file not found: %s)", dataset_id, path)
