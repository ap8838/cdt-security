import glob
import os
import subprocess
import sys
import argparse

def get_datasets():
    datasets = [
        os.path.basename(f).replace("_train.parquet", "")
        for f in glob.glob("data/processed/*_train.parquet")
    ]
    if not datasets:
        print(" No train parquet files found in data/processed/")
        sys.exit(1)
    return datasets

def run_ae(dataset):
    print(f"    Step: Autoencoder (Train & Eval)")
    subprocess.run([sys.executable, "-m", "src.models.train_ae", "--dataset", dataset], check=True)
    subprocess.run([sys.executable, "-m", "src.models.eval_ae", "--dataset", dataset], check=True)

def run_ganomaly(dataset, window=1):
    print(f"    Step: GANomaly (Train & Eval) | Window Size: {window}")
    # Train
    subprocess.run([
        sys.executable, "-m", "src.models.train_ganomaly",
        "--dataset", dataset,
        "--epochs", "150",
        "--lr", "0.0001",
        "--lambda-adv", "0.05",
        "--lambda-latent", "15",
        "--window", str(window)
    ], check=True)
    # Eval (Window must match training)
    subprocess.run([
        sys.executable, "-m", "src.models.eval_ganomaly",
        "--dataset", dataset,
        "--window", str(window)
    ], check=True)

def main():
    parser = argparse.ArgumentParser(description="Master script to run AI models.")
    parser.add_argument(
        "mode",
        choices=["ae", "ganomaly", "both"],
        help="Which model(s) to run: 'ae', 'ganomaly', or 'both'"
    )
    parser.add_argument("--dataset", default="all", help="Specific dataset name or 'all'")
    parser.add_argument("--window", type=int, default=1, help="Sliding window size for GANomaly")

    args = parser.parse_args()

    all_datasets = get_datasets()
    target_datasets = [args.dataset] if args.dataset != "all" else all_datasets

    print(f" Starting process in mode: {args.mode.upper()}")

    for ds in target_datasets:
        if ds not in all_datasets:
            print(f"  Dataset '{ds}' not found. Skipping...")
            continue

        print(f"\n Processing Dataset: {ds}")

        if args.mode in ["ae", "both"]:
            run_ae(ds)

        if args.mode in ["ganomaly", "both"]:
            run_ganomaly(ds, window=args.window)

    print("\n All requested tasks completed successfully!")

if __name__ == "__main__":
    main()