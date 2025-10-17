import glob
import os
import subprocess
from pathlib import Path


def run_cgan_training(dataset: str):
    """
    Executes the src.adversarial.train_cgan script for a specific dataset
    using subprocess to simulate a command-line call.
    """
    print(f"\n🚀 Training cGAN for dataset: {dataset}")

    # Define paths based on the dataset name
    in_parquet = Path(f"data/processed/{dataset}_train.parquet")
    out_model = Path(f"artifacts/adversarial/{dataset}_cgan.pt")
    features_json = Path(f"artifacts/preproc/{dataset}_features.json")
    scaler_pkl = Path(f"artifacts/preproc/{dataset}_scaler.pkl")
    encoders_pkl = Path(f"artifacts/preproc/{dataset}_encoders.pkl")

    # Skip dataset if required files are missing
    required_files = [in_parquet, features_json, scaler_pkl, encoders_pkl]
    if not all(f.exists() for f in required_files):
        print(
            f"⚠️ Skipping {dataset}: missing one or more required preprocessing files."
        )
        return

    try:
        command = [
            "python",
            "-m",
            "src.adversarial.train_cgan",
            "--dataset",
            dataset,
            "--parquet",
            str(in_parquet),
            "--features",
            str(features_json),
            "--scaler",
            str(scaler_pkl),
            "--encoders",
            str(encoders_pkl),
            "--out",
            str(out_model),
            "--epochs",
            "60",
            "--batch-size",
            "256",
            "--z-dim",
            "64",
            "--train-on",
            "anomaly",
        ]

        subprocess.run(command, check=True)
        print(f"✅ [{dataset}] cGAN model training complete.")

    except subprocess.CalledProcessError as e:
        print(f"❌ [{dataset}] cGAN training failed! Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error while training {dataset}: {e}")


def run_all_cgan():
    """Iterates through all datasets in data/processed and runs the cGAN training."""
    parquet_files = glob.glob("data/processed/*_train.parquet")
    if not parquet_files:
        print(
            "❌ No *_train.parquet files found in data/processed/. Please preprocess datasets first."
        )
        return

    datasets = sorted(
        set(os.path.basename(p).replace("_train.parquet", "") for p in parquet_files)
    )
    print(f"\n🧩 Found {len(datasets)} datasets to train: {datasets}")

    for dataset in datasets:
        run_cgan_training(dataset)

    print("\n🎉 All cGAN training processes completed successfully.")


if __name__ == "__main__":
    run_all_cgan()
