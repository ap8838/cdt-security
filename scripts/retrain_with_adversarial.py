import glob
import os
import shutil
import subprocess

import pandas as pd


def retrain_all():
    adv_dir = "artifacts/adversarial"
    processed_dir = "data/processed"

    for gen_file in glob.glob(os.path.join(adv_dir, "*_generated.parquet")):
        dataset = os.path.basename(gen_file).replace("_generated.parquet", "")
        train_path = os.path.join(processed_dir, f"{dataset}_train.parquet")

        if not os.path.exists(train_path):
            print(f"⚠️ Missing train file for {dataset}, skipping.")
            continue

        print(f"\n🔁 Retraining {dataset} with synthetic anomalies included...")

        # Load both DataFrames
        df_train = pd.read_parquet(train_path)
        df_gen = pd.read_parquet(gen_file)

        # Ensure both have same columns
        for col in df_train.columns:
            if col not in df_gen.columns:
                df_gen[col] = None
        for col in df_gen.columns:
            if col not in df_train.columns:
                df_train[col] = None

        # Align columns and convert timestamp dtype
        df_train["timestamp"] = df_train["timestamp"].astype(str)
        df_gen["timestamp"] = df_gen["timestamp"].astype(str)
        df_train = df_train[df_gen.columns]

        # Merge both
        df_combined = pd.concat([df_train, df_gen], ignore_index=True)

        combined_path = os.path.join(
            processed_dir, f"{dataset}_train_augmented.parquet"
        )

        # Save cleanly
        df_combined.to_parquet(combined_path, index=False)
        print(f"💾 Saved augmented train data: {combined_path}")

        # ✅ Overwrite the main training file with augmented version
        shutil.copy(combined_path, train_path)
        print(f"📁 Replaced base training file: {train_path}")

        # Retrain AE (uses the updated train parquet automatically)
        subprocess.run(
            ["python", "-m", "src.models.train_ae", "--dataset", dataset], check=True
        )

        # Retrain GANomaly
        subprocess.run(
            ["python", "-m", "src.models.train_ganomaly", "--dataset", dataset],
            check=True,
        )

        print(f"✅ Retrained AE + GANomaly for {dataset} using augmented data.")


if __name__ == "__main__":
    retrain_all()
