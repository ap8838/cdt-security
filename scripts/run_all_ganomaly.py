import glob
import os
import subprocess


def run_all_ganomaly():
    # 1. Find all *_train.parquet files
    train_files = glob.glob("data/processed/*_train.parquet")

    if not train_files:
        print("❌ No train parquet files found in data/processed/")
        return

    for train_file in train_files:
        dataset = os.path.basename(train_file).replace("_train.parquet", "")
        print(f"\n🚀 Running GANomaly for dataset: {dataset}")

        # 2. Train GANomaly
        subprocess.run(
            [
                "python",
                "-m",
                "src.models.train_ganomaly",
                "--dataset",
                dataset,
                "--epochs",
                "20",  # you can tune this
                "--lr",
                "0.0001",  # you can tune this too
            ],
            check=True,
        )

        # 3. Evaluate GANomaly
        subprocess.run(
            ["python", "-m", "src.models.eval_ganomaly", "--dataset", dataset],
            check=True,
        )

    print("\n✅ All datasets processed successfully with GANomaly!")


if __name__ == "__main__":
    run_all_ganomaly()
