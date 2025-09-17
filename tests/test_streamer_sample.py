# tests/test_streamer_sample.py
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

# import the project pieces (must run pytest from repo root)
from src.data.preprocess import preprocess_dataset
from src.stream.streamer import run_streamer


def find_sample_csv(name: str) -> Path | None:
    """Look for sample CSV in likely test folders (handles both `sample_data` and `sample data`)."""
    candidates = [Path("tests") / "sample_data", Path("tests") / "sample data"]
    for d in candidates:
        p = d / name
        if p.exists():
            return p
    return None


def test_streamer_with_sample_csv(tmp_path):
    """
    End-to-end smoke test:
    - Preprocess a sample CSV into processed/test.parquet (in a tmp dir).
    - Subsample the processed test file (so test is fast).
    - Run the streamer on the small parquet into a tmp sqlite DB.
    - Assert the asset_state table was written and contains a snapshot row.
    """
    sample_csv = find_sample_csv("IoT_Fridge.csv")
    if sample_csv is None:
        pytest.skip(
            "No sample CSV found under tests/sample_data or 'tests/sample data'"
        )

    # Directories inside a temporary folder so test does not touch repo files
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1) Run preprocessing to create processed/iot_fridge_test.parquet (train/test + artifacts)
    preprocess_dataset(
        str(sample_csv),
        "iot_fridge",
        output_dir=str(processed_dir),
        artifacts_dir=str(artifacts_dir),
    )

    test_parquet = processed_dir / "iot_fridge_test.parquet"
    assert test_parquet.exists(), "Preprocessor did not create test parquet"

    # 2) Subsample (speed) — keep small number of rows for fast CI
    df_test = pd.read_parquet(test_parquet)
    if len(df_test) > 200:
        df_small = df_test.head(200)
    else:
        df_small = df_test
    small_parquet = processed_dir / "iot_fridge_test_small.parquet"
    df_small.to_parquet(small_parquet, index=False)

    # 3) Run streamer writing to a temporary sqlite DB
    db_path = tmp_path / "stream.db"
    run_streamer(str(small_parquet), str(db_path), rate=10000, speed=10000)

    # 4) Validate DB content
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # ensure table exists
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='asset_state'"
    )
    assert cur.fetchone() is not None, "asset_state table missing"

    cur.execute("SELECT asset_id, data, label FROM asset_state")
    rows = cur.fetchall()
    conn.close()

    assert len(rows) >= 1, "asset_state is empty"
    asset_id, data_json, label = rows[0]
    data = json.loads(data_json)

    # basic sanity checks
    assert asset_id == "iot_fridge", "asset_id mismatch"
    assert isinstance(data, dict), "data column must be JSON string of features"
    assert label in (0, 1), "label must be 0 or 1"
