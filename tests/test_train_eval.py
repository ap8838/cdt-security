import json
import os
import shutil

import pandas as pd
import pytest

from src.models.eval_ae import evaluate_autoencoder
from src.models.train_ae import train_autoencoder


@pytest.fixture(scope="session")
def sample_parquet(tmp_path_factory):
    """
    Convert sample CSV (IoT_Fridge.csv) into parquet train/test for testing.
    Handles date/time merge -> timestamp.
    """
    csv_path = "tests/sample_data/IoT_Fridge.csv"
    df = pd.read_csv(csv_path)

    #  Merge date+time -> timestamp
    if {"date", "time"}.issubset(df.columns):
        df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
        df = df.drop(columns=["date", "time"])

    #  Add dummy asset_id if missing
    if "asset_id" not in df.columns:
        df["asset_id"] = "iot_fridge"

    #  Add labels if missing
    if "label" not in df.columns:
        df["label"] = [0] * (len(df) // 2) + [1] * (len(df) - len(df) // 2)

    #  Split into train (normal only) and test (all)
    train_df = df[df["label"] == 0].copy()
    test_df = df.copy()

    #  Save parquet into temp dir
    out_dir = tmp_path_factory.mktemp("processed")
    train_file = out_dir / "iot_fridge_train.parquet"
    test_file = out_dir / "iot_fridge_test.parquet"
    train_df.to_parquet(train_file, index=False)
    test_df.to_parquet(test_file, index=False)

    #  Minimal feature list JSON
    features = {
        "all": [c for c in df.columns if c not in ("label",)],
    }
    preproc_dir = tmp_path_factory.mktemp("preproc")
    features_file = preproc_dir / "iot_fridge_features.json"
    with open(features_file, "w") as f:
        json.dump(features, f)

    return {
        "train_file": str(train_file),
        "test_file": str(test_file),
        "features_file": str(features_file),
    }


def test_train_and_eval(sample_parquet, tmp_path):
    dataset = "iot_fridge"

    #  Create expected project dirs
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts/preproc", exist_ok=True)
    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/reports", exist_ok=True)

    #  Copy parquet + features.json into the locations train/eval expect
    shutil.copy(sample_parquet["train_file"], f"data/processed/{dataset}_train.parquet")
    shutil.copy(sample_parquet["test_file"], f"data/processed/{dataset}_test.parquet")
    shutil.copy(
        sample_parquet["features_file"],
        f"artifacts/preproc/{dataset}_features.json",
    )

    #  Train
    train_autoencoder(
        dataset=dataset,
        features_file=f"artifacts/preproc/{dataset}_features.json",
        epochs=2,  # keep fast
        lr=1e-3,
    )

    #  Eval
    evaluate_autoencoder(
        dataset=dataset,
        features_file=f"artifacts/preproc/{dataset}_features.json",
    )

    # Check that artifacts exist
    model_path = f"artifacts/models/{dataset}_ae.pt"
    report_path = f"artifacts/reports/{dataset}_ae_eval.json"
    assert os.path.exists(model_path)
    assert os.path.exists(report_path)

    # Report should have metrics
    with open(report_path) as f:
        report = json.load(f)
    assert "precision" in report
    assert "recall" in report
    assert "f1" in report
    assert "roc_auc" in report
