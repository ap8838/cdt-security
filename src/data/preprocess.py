import json
import os
import traceback

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def make_onehot(**kwargs):
    try:
        return OneHotEncoder(**kwargs)
    except TypeError:
        # replace sparse_output -> sparse for older sklearn
        if "sparse_output" in kwargs:
            v = kwargs.pop("sparse_output")
            kwargs["sparse"] = v
        return OneHotEncoder(**kwargs)


def preprocess_dataset(
    csv_path,
    dataset_name,
    output_dir="data/processed",
    artifacts_dir="artifacts/preproc",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    try:
        # 1. Load dataset
        df = pd.read_csv(csv_path, low_memory=False)

        # 2. Create timestamp column
        if "date" in df.columns and "time" in df.columns:
            combined = (
                df["date"].astype(str).str.strip()
                + " "
                + df["time"].astype(str).str.strip()
            )
            # try a strict format first to reduce warnings, fallback if needed
            try:
                df["timestamp"] = pd.to_datetime(
                    combined, format="%d-%b-%y %H:%M:%S", errors="coerce"
                )
                if df["timestamp"].isna().any():
                    df["timestamp"] = pd.to_datetime(combined, errors="coerce")
            except Exception:
                df["timestamp"] = pd.to_datetime(combined, errors="coerce")
            df.drop(columns=["date", "time"], inplace=True, errors="ignore")
        elif "ts" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
            df.drop(columns=["ts"], inplace=True, errors="ignore")

        # 3. Add dataset identifier
        df["asset_id"] = dataset_name

        # 4. Define label and potential categorical columns
        label_col = "label"
        cat_candidates = [
            c for c in df.columns if df[c].dtype == "object" or df[c].dtype == "bool"
        ]
        if "type" in df.columns and "type" not in cat_candidates:
            cat_candidates.append("type")

        # 5. Separate numeric vs categorical
        drop_cols = [c for c in ["timestamp", label_col, "asset_id"] if c in df.columns]
        numeric_cols = [c for c in df.columns if c not in drop_cols + cat_candidates]

        # 6. Create train-only view for fitting (normal only)
        if label_col not in df.columns:
            raise ValueError(f"Expected a '{label_col}' column in {csv_path}")
        train_mask = df[label_col] == 0
        train_df = df[train_mask].copy()

        # 7. Fit scaler on train-only, then transform all
        scaler = None
        if numeric_cols:
            scaler = MinMaxScaler()
            # attempt fit on train numeric columns
            scaler.fit(train_df[numeric_cols])
            df[numeric_cols] = scaler.transform(df[numeric_cols])

        # 8. Fit encoders on train-only, transform all (hybrid)
        encoded_parts = []
        encoders = {}
        encoded_cols = []

        for col in cat_candidates:
            if col not in df.columns:
                continue
            # fit on train (cast to str to avoid mixed-type issues)
            train_values = train_df[col].astype(str)
            unique_vals = train_values.nunique(dropna=True)

            if unique_vals <= 50:
                # One-hot for low-cardinality (safe)
                ohe = make_onehot(sparse_output=False, handle_unknown="ignore")
                ohe.fit(train_values.to_frame())
                transformed = ohe.transform(df[col].astype(str).to_frame())
                # category names from training fit
                try:
                    cats = ohe.categories_[0]
                except Exception:
                    cats = [str(i) for i in range(transformed.shape[1])]
                col_names = [f"{col}_{c}" for c in cats]
                df_encoded = pd.DataFrame(
                    transformed, columns=col_names, index=df.index
                )
                encoded_parts.append(df_encoded)
                encoders[col] = ("onehot", ohe)
                encoded_cols.extend(col_names)
            else:
                # Frequency encoding for high-cardinality columns
                freq_map = train_values.value_counts(normalize=True).to_dict()
                encoded_series = (
                    df[col].astype(str).map(freq_map).fillna(0).rename(f"{col}_freq")
                )
                encoded_parts.append(encoded_series.to_frame())
                encoders[col] = ("freq", freq_map)
                encoded_cols.append(f"{col}_freq")

        # 9. Concatenate encoded parts into df
        if encoded_parts:
            df = pd.concat(
                [df.drop(columns=cat_candidates, errors="ignore"), *encoded_parts],
                axis=1,
            )
        else:
            df = df.drop(columns=cat_candidates, errors="ignore")

        # 10. Fill any remaining NaNs (safety net)
        df = df.fillna(0)

        # 11. Split into train/test (train must be normal-only)
        train = df[df[label_col] == 0].copy()
        test = df.copy()

        # 12. Save processed files
        train_path = os.path.join(output_dir, f"{dataset_name}_train.parquet")
        test_path = os.path.join(output_dir, f"{dataset_name}_test.parquet")
        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)

        # 13. Save artifacts
        if scaler is not None:
            joblib.dump(
                scaler, os.path.join(artifacts_dir, f"{dataset_name}_scaler.pkl")
            )
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
        ) as f:
            json.dump(feature_list, f, indent=2)

        # 14. Report
        print(f"✅ {dataset_name} processed")
        print(f"Train: {train.shape}, Test: {test.shape}")
        print(
            f"Class distribution (test):\n{test[label_col].value_counts(normalize=True)}\n"
        )

    except Exception as e:
        print(f"⚠️ Failed processing {dataset_name}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    datasets = {
        # IoT
        "iot_fridge": "data/raw/IoT_Fridge.csv",
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

    for dataset_id, path in datasets.items():
        if os.path.exists(path):
            preprocess_dataset(path, dataset_id)
        else:
            print(f"skipping {dataset_id} (file not found: {path})")
