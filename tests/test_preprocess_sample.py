import json

import joblib
import pandas as pd

from src.data.preprocess import preprocess_dataset


def test_sample_preprocessing(tmp_path):
    # point processed + artifacts to a temp dir
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"

    datasets = {
        "iot_fridge": "tests/sample data/IoT_Fridge.csv",
        "linux_disk1": "tests/sample data/linux_disk_1.csv",
        "linux_mem1": "tests/sample data/linux_memory1.csv",
        "linux_proc1": "tests/sample data/linux_process_1.csv",
        "win7": "tests/sample data/windows7_dataset.csv",
    }

    for name, path in datasets.items():
        preprocess_dataset(
            path,
            name,
            output_dir=processed_dir,
            artifacts_dir=artifacts_dir,
        )

        train_path = processed_dir / f"{name}_train.parquet"
        test_path = processed_dir / f"{name}_test.parquet"
        features_path = artifacts_dir / f"{name}_features.json"

        # 1. Files exist
        assert train_path.exists()
        assert test_path.exists()
        assert features_path.exists()

        # 2. Load data
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)

        # 3. Columns consistent
        with open(features_path) as f:
            features = json.load(f)
        cols_expected = set(features["all"] + ["label"])
        assert set(train.columns) == cols_expected
        assert set(test.columns) == cols_expected

        # 4. Train has only normal
        assert train["label"].nunique() == 1 and train["label"].iloc[0] == 0

        # 5. No NaNs
        assert not train.isnull().any().any()
        assert not test.isnull().any().any()

        # 6. Artifacts loadable
        encoders_path = artifacts_dir / f"{name}_encoders.pkl"
        assert encoders_path.exists()
        _ = joblib.load(encoders_path)
