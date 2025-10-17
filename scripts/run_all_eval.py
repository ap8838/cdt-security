import glob
import os
import subprocess


def run_all_evals():
    print("\n🔍 Running evaluation for all models...\n")

    # Loop through datasets that have test parquet files
    datasets = [
        os.path.basename(f).replace("_test.parquet", "")
        for f in glob.glob("data/processed/*_test.parquet")
    ]

    for dataset in datasets:
        print(f"📊 Evaluating Autoencoder for {dataset}...")
        subprocess.run(
            ["python", "-m", "src.models.eval_ae", "--dataset", dataset], check=False
        )

        print(f"📊 Evaluating GANomaly for {dataset}...")
        subprocess.run(
            ["python", "-m", "src.models.eval_ganomaly", "--dataset", dataset],
            check=False,
        )

    print("\n✅ All evaluations completed successfully!\n")


if __name__ == "__main__":
    run_all_evals()
