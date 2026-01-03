import glob
import os
import subprocess


def eval_all_generated():
    gens = glob.glob("artifacts/adversarial/*_generated.parquet")
    for gen_path in gens:
        dataset = os.path.basename(gen_path).replace("_generated.parquet", "")
        print(f"\n Evaluating generated samples for {dataset}")
        out = f"artifacts/adversarial/{dataset}_eval.csv"

        subprocess.run(
            [
                "python",
                "-m",
                "src.adversarial.eval_generated",
                "--parquet",
                gen_path,
                "--dataset",
                dataset,
                "--model",
                "ganomaly",
                "--out",
                out,
            ],
            check=True,
        )


if __name__ == "__main__":
    eval_all_generated()
