import json
import os

import joblib
import pandas as pd
import pytest


@pytest.mark.full  # <-- mark this as a full/integration test
def test_preprocessed_files():
    processed_dir = "data/processed"
    artifacts_dir = "artifacts/preproc"

    # Assets to test (all datasets)
    assets = [
        "iot_fridge",
        "iot_gps",
        "iot_modbus",
        "iot_motion",
        "iot_thermo",
        "iot_weather",
        "linux_disk1",
        "linux_disk2",
        "linux_mem1",
        "linux_mem2",
        "linux_proc1",
        "linux_proc2",
        "win7",
        "win10",
    ]

    for asset in assets:
        train_path = os.path.join(processed_dir, f"{asset}_train.parquet")
        test_path = os.path.join(processed_dir, f"{asset}_test.parquet")
        features_path = os.path.join(artifacts_dir, f"{asset}_features.json")

        # 1. Files exist
        assert os.path.exists(train_path), f"{asset} train missing"
        assert os.path.exists(test_path), f"{asset} test missing"
        assert os.path.exists(features_path), f"{asset} features.json missing"

        # 2. Load data
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)

        # 3. Columns consistent with feature list
        with open(features_path) as f:
            features = json.load(f)
        cols_expected = set(features["all"] + ["label"])
        assert set(train.columns) == cols_expected, f"{asset} train cols mismatch"
        assert set(test.columns) == cols_expected, f"{asset} test cols mismatch"

        # 4. Train has only normal samples
        assert (
            train["label"].nunique() == 1 and train["label"].iloc[0] == 0
        ), f"{asset} train has label leakage"

        # 5. Class distribution makes sense (should include anomalies in test)
        assert 0 in test["label"].unique(), f"{asset} test missing normal samples"
        assert 1 in test["label"].unique(), f"{asset} test missing malicious samples"

        # 6. Artifacts loadable
        scaler_path = os.path.join(artifacts_dir, f"{asset}_scaler.pkl")
        if os.path.exists(scaler_path):
            _ = joblib.load(scaler_path)
        encoders_path = os.path.join(artifacts_dir, f"{asset}_encoders.pkl")
        if os.path.exists(encoders_path):
            _ = joblib.load(encoders_path)

        # 7. No NaNs after preprocessing
        assert not train.isnull().any().any(), f"{asset} train has NaNs"
        assert not test.isnull().any().any(), f"{asset} test has NaNs"

        # 8. Consistent column order between train and test
        assert list(train.columns) == list(
            test.columns
        ), f"{asset} column order mismatch"

    print("✅ All Phase 1 checks passed for all assets!")
