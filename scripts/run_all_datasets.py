import glob
import os
import subprocess


def main():
    datasets = [
        os.path.basename(f).replace("_train.parquet", "")
        for f in glob.glob("data/processed/*_train.parquet")
    ]
    for ds in datasets:
        print(f"\n🚀 Processing {ds} ...")
        subprocess.run(
            ["python", "-m", "src.models.train_ae", "--dataset", ds], check=True
        )
        subprocess.run(
            ["python", "-m", "src.models.eval_ae", "--dataset", ds], check=True
        )
        subprocess.run(
            ["python", "-m", "src.models.train_ganomaly", "--dataset", ds], check=True
        )
        subprocess.run(
            ["python", "-m", "src.models.eval_ganomaly", "--dataset", ds], check=True
        )
    print("\n✅ All datasets processed successfully!")


if __name__ == "__main__":
    main()
