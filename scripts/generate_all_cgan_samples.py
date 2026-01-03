import glob
import os
import subprocess


def generate_all():
    models = glob.glob("artifacts/adversarial/*_cgan.pt")
    for model_path in models:
        dataset = os.path.basename(model_path).replace("_cgan.pt", "")
        print(f"\n Generating synthetic samples for {dataset}")
        features = f"artifacts/preproc/{dataset}_features.json"
        out = f"artifacts/adversarial/{dataset}_generated.parquet"

        subprocess.run(
            [
                "python",
                "-m",
                "src.adversarial.generate_samples",
                "--model",
                model_path,
                "--features",
                features,
                "--n",
                "100",
                "--out",
                out,
            ],
            check=True,
        )


if __name__ == "__main__":
    generate_all()
