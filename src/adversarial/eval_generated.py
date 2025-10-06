"""
Take generated.parquet and compute scores using existing inference model.
Outputs a CSV/JSON with scores so you can compare detection rates.
Example:
  python -m src.adversarial.eval_generated --parquet artifacts/adversarial/generated.parquet --out artifacts/adversarial/eval.csv
"""

import argparse

import pandas as pd

from src.service.infer_service import InferenceService


def eval_generated(parsed_args):  # Renamed 'args' to 'parsed_args'
    df = pd.read_parquet(parsed_args.parquet)
    svc = InferenceService(dataset=parsed_args.dataset, model_type=parsed_args.model)
    rows = []
    for _, r in df.iterrows():
        evt = {"asset_id": r["asset_id"], "timestamp": r["timestamp"]}
        # Access columns from the DataFrame using the DataFrame's column names
        features = {
            c: r[c] for c in df.columns if c not in ("asset_id", "timestamp", "label")
        }
        evt.update(features)
        res = svc.score(evt)
        rows.append({**res, "raw_features": features})
    out = pd.DataFrame(rows)
    out.to_csv(parsed_args.out, index=False)
    print("Wrote", parsed_args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--out", default="artifacts/adversarial/eval.csv")
    p.add_argument("--dataset", default="iot_fridge")
    p.add_argument("--model", default="ganomaly")
    args = p.parse_args()
    eval_generated(args)
