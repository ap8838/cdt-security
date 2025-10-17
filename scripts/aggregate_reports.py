# scripts/aggregate_reports.py
import json
import os
from pathlib import Path

import pandas as pd


def load_reports(folder):
    rows = []
    for file in Path(folder).glob("*_eval.json"):
        parts = file.stem.split("_")
        dataset = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
        model = parts[-2] if len(parts) > 1 else "unknown"

        with open(file) as f:
            metrics = json.load(f)
        row = {"dataset": dataset, "model": model}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def load_adversarial(folder="artifacts/adversarial"):
    rows = []
    for file in Path(folder).glob("*_eval.csv"):
        dataset = file.stem.replace("_eval", "")
        df = pd.read_csv(file)
        df["dataset"] = dataset
        df["model"] = "cgan"
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aggregate_reports():
    out_file = "artifacts/reports/aggregate_metrics.csv"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    ae_gan_df = load_reports("artifacts/reports")
    adv_df = load_adversarial("artifacts/adversarial")

    combined = pd.concat([ae_gan_df, adv_df], ignore_index=True)
    combined.to_csv(out_file, index=False)
    print(f"✅ Aggregated report saved → {out_file}")
    print(combined.head())


if __name__ == "__main__":
    aggregate_reports()
