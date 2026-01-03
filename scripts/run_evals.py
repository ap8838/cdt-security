import argparse
import glob
import os
import subprocess
import sys

def get_datasets():
    ds = [
        os.path.basename(p).replace("_test.parquet", "")
        for p in glob.glob("data/processed/*_test.parquet")
    ]
    if not ds:
        print(" No test datasets found in data/processed/")
        sys.exit(1)
    return sorted(ds)

def eval_ae(dataset):
    print(f" [AE] Evaluating Autoencoder for: {dataset}...")
    # AE remains pointwise
    subprocess.run(
        [sys.executable, "-m", "src.models.eval_ae", "--dataset", dataset],
        check=False
    )

<<<<<<< Updated upstream
def eval_ganomaly(dataset, window=1):
    print(f" [GAN] Evaluating GANomaly for: {dataset} (Window: {window})...")
    subprocess.run(
        [sys.executable, "-m", "src.models.eval_ganomaly", "--dataset", dataset, "--window", str(window)],
=======

def eval_ganomaly(dataset, window=5):
    print(f" [GAN] Evaluating GANomaly for: {dataset} (Window: {window})...")
    subprocess.run(
        [
            sys.executable, "-m", "src.models.eval_ganomaly",
            "--dataset", dataset,
            "--window", str(window)
        ],
>>>>>>> Stashed changes
        check=False
    )

def main():
    parser = argparse.ArgumentParser(description="Master script to evaluate AI models.")
    parser.add_argument(
        "mode",
        choices=["ae", "ganomaly", "both"],
        help="Which model type to evaluate: 'ae', 'ganomaly', or 'both'"
    )
    parser.add_argument("--dataset", default="all", help="Specific dataset name or 'all'")
<<<<<<< Updated upstream
    parser.add_argument("--window", type=int, default=1, help="Window size used during training")
=======
    parser.add_argument("--window", type=int, default=5, help="Window size for GANomaly (V1)")
>>>>>>> Stashed changes

    args = parser.parse_args()

    all_datasets = get_datasets()
    target_datasets = [args.dataset] if args.dataset != "all" else all_datasets

    print(f" Starting EVALUATION in mode: {args.mode.upper()}")

    for ds in target_datasets:
        if ds not in all_datasets:
            print(f"⚠  Dataset '{ds}' not found. Skipping...")
            continue

        if args.mode in ["ae", "both"]:
            eval_ae(ds)

        if args.mode in ["ganomaly", "both"]:
            eval_ganomaly(ds, window=args.window)

    print("\n All requested evaluations completed!")

if __name__ == "__main__":
    main()