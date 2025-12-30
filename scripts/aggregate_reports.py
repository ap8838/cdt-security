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


def aggregate_reports():
    out_file = "artifacts/reports/aggregate_metrics.csv"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    ae_gan_df = load_reports("artifacts/reports")

    combined = ae_gan_df

    combined.to_csv(out_file, index=False)
    print(f" Clean aggregated report saved → {out_file}")
    print(combined.head())


if __name__ == "__main__":
    aggregate_reports()
