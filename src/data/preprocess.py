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
            # ---  SPECIFY FORMAT TO REMOVE WARNING AND ENSURE SORT ACCURACY ---
            # Format %d-%b-%y matches "01-Feb-23"
            df["timestamp"] = pd.to_datetime(
                combined, format="%d-%b-%y %H:%M:%S", errors="coerce"
            )

            # Fallback for non-matching rows (if any)
            mask = df["timestamp"].isna()
            if mask.any():
                df.loc[mask, "timestamp"] = pd.to_datetime(combined[mask], errors="coerce")

            df.drop(columns=["date", "time"], inplace=True, errors="ignore")
        elif "ts" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
            df.drop(columns=["ts"], inplace=True, errors="ignore")

        #  Chronological Sort ---
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 3. Add identifiers
        df["asset_id"] = dataset_name
        df["asset"] = dataset_name

        # 4. Determine numeric vs categorical
        label_col = "label"
        protected = {"asset_id", "asset", "timestamp", label_col}
        numeric_cols = []
        cat_candidates = []
        for c in df.columns:
            if c in protected:
                continue
            ser = df[c]
            if pd.api.types.is_numeric_dtype(ser):
                numeric_cols.append(c)
            elif _is_numeric_like(ser, 0.9):
                df[c] = pd.to_numeric(ser, errors="coerce")
                numeric_cols.append(c)
            else:
                cat_candidates.append(c)

        #  Chronological Split (80/20) ---
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # 5. Fit scaler on normal rows within the TRAINING set only
        normal_train = train_df[train_df[label_col] == 0].copy()
        if numeric_cols and not normal_train.empty:
            scaler = MinMaxScaler()
            scaler.fit(normal_train[numeric_cols])
            # Apply to both
            train_df[numeric_cols] = scaler.transform(train_df[numeric_cols].fillna(0))
            test_df[numeric_cols] = scaler.transform(test_df[numeric_cols].fillna(0))
        else:
            scaler = {"feature_names": []}

        # 6. Encode categoricals (Fit on train-normal, transform both)
        encoded_parts_train = []
        encoded_parts_test = []
        encoders = {}
        encoded_cols = []

        for col in cat_candidates:
            train_values = normal_train[col].fillna("").astype(str).str.strip()
            if train_values.nunique() <= 50:
                ohe = make_onehot(sparse_output=False, handle_unknown="ignore")
                ohe.fit(train_values.to_frame())

                tr_enc = ohe.transform(
                    train_df[col].fillna("").astype(str).str.strip().to_frame()
                )
                te_enc = ohe.transform(
                    test_df[col].fillna("").astype(str).str.strip().to_frame()
                )

                cats = [f"{col}_{str(x)}" for x in ohe.categories_[0]]
                encoded_parts_train.append(
                    pd.DataFrame(tr_enc, columns=cats, index=train_df.index)
                )
                encoded_parts_test.append(
                    pd.DataFrame(te_enc, columns=cats, index=test_df.index)
                )
                encoders[col] = ("onehot", ohe)
                encoded_cols.extend(cats)
            else:
                freq_map = train_values.value_counts(normalize=True).to_dict()
                encoded_parts_train.append(
                    train_df[col]
                    .fillna("")
                    .astype(str)
                    .map(freq_map)
                    .fillna(0)
                    .rename(f"{col}_freq")
                )
                encoded_parts_test.append(
                    test_df[col]
                    .fillna("")
                    .astype(str)
                    .map(freq_map)
                    .fillna(0)
                    .rename(f"{col}_freq")
                )
                encoders[col] = ("freq", freq_map)
                encoded_cols.append(f"{col}_freq")

        # 7. Re-assemble and Drop objects
        train_df = train_df.drop(columns=cat_candidates, errors="ignore")
        test_df = test_df.drop(columns=cat_candidates, errors="ignore")

        if encoded_parts_train:
            train_df = pd.concat([train_df] + encoded_parts_train, axis=1)
            test_df = pd.concat([test_df] + encoded_parts_test, axis=1)

        train_df = train_df.fillna(0)
        test_df = test_df.fillna(0)

        # 8. Save to parquet
        cols_order = ["asset_id", "asset", "timestamp"] + [
            c for c in train_df.columns if c not in ("asset_id", "asset", "timestamp")
        ]
        train_df = train_df[cols_order]
        test_df = test_df[cols_order]

        train_path = os.path.join(output_dir, f"{dataset_name}_train.parquet")
        test_path = os.path.join(output_dir, f"{dataset_name}_test.parquet")
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)

        # Save preprocessing artifacts
        joblib.dump(scaler, os.path.join(artifacts_dir, f"{dataset_name}_scaler.pkl"))
        joblib.dump(
            encoders, os.path.join(artifacts_dir, f"{dataset_name}_encoders.pkl")
        )

        feature_list = {
            "numeric": numeric_cols,
            "categorical": cat_candidates,
            "encoded": encoded_cols,
            "all": [
                c
                for c in train_df.columns
                if c not in ("asset_id", "asset", "timestamp", label_col)
            ],
        }
        with open(
                os.path.join(artifacts_dir, f"{dataset_name}_features.json"), "w"
        ) as fh:
            json.dump(feature_list, fh, indent=2)

        logging.info(
            "%s processed. Train: %s, Test: %s",
            dataset_name,
            train_df.shape,
            test_df.shape,
        )

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
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    datasets = {
        "iot_fridge": "data/raw/IoT_Fridge.csv",
        "iot_garage": "data/raw/IoT_Garage_Door.csv",
        "iot_gps": "data/raw/IoT_GPS_Tracker.csv",
        "iot_modbus": "data/raw/IoT_Modbus.csv",
        "iot_motion": "data/raw/IoT_Motion_Light.csv",
        "iot_thermo": "data/raw/IoT_Thermostat.csv",
        "iot_weather": "data/raw/IoT_Weather.csv",
        "linux_disk1": "data/raw/linux_disk_1.csv",
        "linux_disk2": "data/raw/linux_disk_2.csv",
        "linux_mem1": "data/raw/linux_memory1.csv",
        "linux_mem2": "data/raw/linux_memory2.csv",
        "linux_proc1": "data/raw/linux_process_1.csv",
        "linux_proc2": "data/raw/linux_process_2.csv",
        "win7": "data/raw/windows7_dataset.csv",
        "win10": "data/raw/windows10_dataset.csv",
    }

    if args.dataset and args.dataset in datasets:
        preprocess_dataset(datasets[args.dataset], args.dataset)
    else:
        for d_id, path in datasets.items():
            if os.path.exists(path):
                try:
                    preprocess_dataset(path, d_id)
                except Exception as exc:
                    logging.error("Processing %s failed: %s", d_id, exc)
            else:
                logging.warning("Skipping %s (file not found: %s)", d_id, path)