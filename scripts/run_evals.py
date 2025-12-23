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
        print("❌ No test datasets found in data/processed/")
        sys.exit(1)
    return sorted(ds)


def eval_ae(dataset):
    print(f"📊 [AE] Evaluating Autoencoder for: {dataset}...")
    subprocess.run(
        [sys.executable, "-m", "src.models.eval_ae", "--dataset", dataset],
        check=False
    )


def eval_ganomaly(dataset):
    print(f"📊 [GAN] Evaluating GANomaly for: {dataset}...")
    subprocess.run(
        [sys.executable, "-m", "src.models.eval_ganomaly", "--dataset", dataset],
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

    args = parser.parse_args()

    all_datasets = get_datasets()
    target_datasets = [args.dataset] if args.dataset != "all" else all_datasets

    print(f"🔍 Starting EVALUATION in mode: {args.mode.upper()}")

    for ds in target_datasets:
        if ds not in all_datasets:
            print(f"⚠️  Dataset '{ds}' not found. Skipping...")
            continue

        if args.mode in ["ae", "both"]:
            eval_ae(ds)

        if args.mode in ["ganomaly", "both"]:
            eval_ganomaly(ds)

    print("\n✅ All requested evaluations completed!")


if __name__ == "__main__":
    main()