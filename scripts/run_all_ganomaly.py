# scripts/run_all_ganomaly.py
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

        # 2️⃣ Train GANomaly with tuned λ values (you can adjust globally here)
        subprocess.run(
            [
                "python",
                "-m",
                "src.models.train_ganomaly",
                "--dataset",
                dataset,
                "--epochs",
                "150",  # More stable convergence
                "--lr",
                "0.0001",  # 1e-4
                "--lambda-adv",
                "0.05",  # Lower weight for adversarial term
                "--lambda-latent",
                "15",  # Stronger latent consistency
            ],
            check=True,
        )

        # 3️⃣ Evaluate GANomaly
        subprocess.run(
            [
                "python",
                "-m",
                "src.models.eval_ganomaly",
                "--dataset",
                dataset,
            ],
            check=True,
        )

    print("\n✅ All datasets processed successfully with GANomaly!")


if __name__ == "__main__":
    run_all_ganomaly()
