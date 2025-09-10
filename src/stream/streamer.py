import argparse
import time

import pandas as pd


def main(input_path, rate=1):
    df = pd.read_parquet(input_path)
    for _, row in df.iterrows():
        print(row.to_dict())
        time.sleep(1.0 / rate)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--rate", type=float, default=1.0)
    args = p.parse_args()
    main(args.input, args.rate)
