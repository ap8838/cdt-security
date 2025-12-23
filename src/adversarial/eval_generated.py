import argparse
import glob
import os
import pandas as pd
from src.service.infer_service import InferenceService


def eval_generated_one(parquet, dataset, model, out):
    df = pd.read_parquet(parquet)
    svc = InferenceService(dataset=dataset, model_type=model)
    rows = []
    for _, r in df.iterrows():
        evt = {
            "asset_id": r.get("asset_id", dataset),
            "timestamp": r.get("timestamp", ""),
        }
        features = {
            c: r[c] for c in df.columns if c not in ("asset_id", "timestamp", "label")
        }
        evt.update(features)
        res = svc.score(evt)
        rows.append({**res, "raw_features": features})
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f" Wrote evaluation for {dataset} → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--parquet", help="Path to generated parquet or directory", default=None
    )
    p.add_argument("--out", default="artifacts/adversarial/eval.csv")
    p.add_argument("--dataset", default="all")
    p.add_argument("--model", default="ganomaly")
    args = p.parse_args()

    if args.dataset == "all":
        files = glob.glob("artifacts/adversarial/*_generated.parquet")
        for parquet in files:
            dataset = os.path.basename(parquet).replace("_generated.parquet", "")
            out = f"artifacts/adversarial/{dataset}_eval.csv"
            eval_generated_one(parquet, dataset, args.model, out)
    else:
        eval_generated_one(args.parquet, args.dataset, args.model, args.out)


if __name__ == "__main__":
    main()
