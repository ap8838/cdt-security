# scripts/compare_reports.py
import json
import os
from pathlib import Path

import pandas as pd


def load_json_reports(folder):
    rows = []
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path.resolve()}")
        return pd.DataFrame()

    files = list(folder_path.glob("*_eval.json"))
    if not files:
        print(f"⚠️ No *_eval.json files found in: {folder_path.resolve()}")

    for f in files:
        try:
            parts = f.stem.split("_")
            dataset = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
            model = parts[-2] if len(parts) > 1 else "unknown"
            with open(f) as j:
                metrics = json.load(j)
            row = {"dataset": dataset, "model": model}
            row.update(metrics)
            rows.append(row)
        except Exception as e:
            print(f"⚠️ Failed to parse {f}: {e}")

    print(f"📄 Loaded {len(rows)} reports from {folder_path.resolve()}")
    return pd.DataFrame(rows)


def compare_reports(
    before="artifacts/reports/baseline",
    after="artifacts/reports",
    out="artifacts/reports/compare_before_after.csv",
):
    print(f"🔍 Comparing reports:\n  BEFORE → {before}\n  AFTER  → {after}\n")

    df_before = load_json_reports(before)
    df_after = load_json_reports(after)

    if df_before.empty or df_after.empty:
        print("⚠️ One or both report folders are empty.")
        print(f"📊 Before count: {len(df_before)} | After count: {len(df_after)}")
        return

    merged = pd.merge(
        df_before, df_after, on=["dataset", "model"], suffixes=("_before", "_after")
    )
    if merged.empty:
        print(
            "⚠️ No overlapping dataset/model pairs found between baseline and new reports."
        )
        print(f"👉 Datasets before: {df_before['dataset'].unique()}")
        print(f"👉 Datasets after:  {df_after['dataset'].unique()}")
        return

    for metric in ["precision", "recall", "f1", "roc_auc"]:
        if f"{metric}_before" in merged and f"{metric}_after" in merged:
            merged[f"{metric}_delta"] = (
                merged[f"{metric}_after"] - merged[f"{metric}_before"]
            )

    os.makedirs(os.path.dirname(out), exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"✅ Comparison saved → {out}")
    print(merged.head())


if __name__ == "__main__":
    compare_reports()
