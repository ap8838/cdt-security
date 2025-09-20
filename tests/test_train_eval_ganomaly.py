import json
import os
import shutil
import subprocess

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_parquet(tmp_path_factory):
    """
    Convert sample CSV (IoT_Fridge.csv) into parquet train/test for testing.
    """
    csv_path = "tests/sample_data/IoT_Fridge.csv"
    df = pd.read_csv(csv_path)

    # ✅ Merge date+time -> timestamp
    if {"date", "time"}.issubset(df.columns):
        df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
        df = df.drop(columns=["date", "time"])

    # ✅ Add dummy asset_id if missing
    if "asset_id" not in df.columns:
        df["asset_id"] = "iot_fridge"

    # ✅ Add labels if missing
    if "label" not in df.columns:
        df["label"] = [0] * (len(df) // 2) + [1] * (len(df) - len(df) // 2)

    # ✅ Split into train (normal only) and test (all)
    train_df = df[df["label"] == 0].copy()
    test_df = df.copy()

    # ✅ Save parquet into temp dir
    out_dir = tmp_path_factory.mktemp("processed_ganomaly")
    train_file = out_dir / "iot_fridge_train.parquet"
    test_file = out_dir / "iot_fridge_test.parquet"
    train_df.to_parquet(train_file, index=False)
    test_df.to_parquet(test_file, index=False)

    # ✅ Minimal features.json
    features = {"all": [c for c in df.columns if c not in ("label",)]}
    preproc_dir = tmp_path_factory.mktemp("preproc_ganomaly")
    features_file = preproc_dir / "iot_fridge_features.json"
    with open(features_file, "w") as f:
        json.dump(features, f)

    return {
        "train_file": str(train_file),
        "test_file": str(test_file),
        "features_file": str(features_file),
    }


def test_train_and_eval_ganomaly(sample_parquet):
    dataset = "iot_fridge"

    # ✅ Ensure dirs exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts/preproc", exist_ok=True)
    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/reports", exist_ok=True)

    # ✅ Copy parquet + features.json into expected locations
    shutil.copy(sample_parquet["train_file"], f"data/processed/{dataset}_train.parquet")
    shutil.copy(sample_parquet["test_file"], f"data/processed/{dataset}_test.parquet")
    shutil.copy(
        sample_parquet["features_file"],
        f"artifacts/preproc/{dataset}_features.json",
    )

    # ✅ Train via CLI
    subprocess.run(
        [
            "python",
            "-m",
            "src.models.train_ganomaly",
            "--dataset",
            dataset,
            "--epochs",
            "1",  # keep fast
        ],
        check=True,
    )

    # ✅ Eval via CLI
    subprocess.run(
        ["python", "-m", "src.models.eval_ganomaly", "--dataset", dataset],
        check=True,
    )

    # ✅ Check expected artifacts
    model_path = f"artifacts/models/{dataset}_ganomaly.pt"
    threshold_path = f"artifacts/models/{dataset}_ganomaly_threshold.json"
    train_log_path = f"artifacts/reports/{dataset}_ganomaly_train_log.json"
    report_path = f"artifacts/reports/{dataset}_ganomaly_eval.json"

    for path in [model_path, threshold_path, train_log_path, report_path]:
        assert os.path.exists(path), f"Missing expected artifact: {path}"

    # ✅ Check eval report has metrics
    with open(report_path) as f:
        report = json.load(f)

    for key in ["precision", "recall", "f1", "roc_auc", "threshold"]:
        assert key in report, f"Missing key in GANomaly eval report: {key}"
