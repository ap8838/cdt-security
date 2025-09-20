import glob
import os
import subprocess


def run_all():
    # 1. Find all *_train.parquet files
    train_files = glob.glob("data/processed/*_train.parquet")

    if not train_files:
        print("❌ No train parquet files found in data/processed/")
        return

    for train_file in train_files:
        dataset = os.path.basename(train_file).replace("_train.parquet", "")
        print(f"\n🚀 Running AE for dataset: {dataset}")

        # 2. Train
        subprocess.run(
            [
                "python",
                "-m",
                "src.models.train_ae",
                "--dataset",
                dataset,
                "--epochs",
                "20",
            ],
            check=True,
        )

        # 3. Evaluate
        subprocess.run(
            ["python", "-m", "src.models.eval_ae", "--dataset", dataset],
            check=True,
        )

    print("\n✅ All datasets processed successfully!")


if __name__ == "__main__":
    run_all()
