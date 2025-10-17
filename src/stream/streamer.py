import argparse
import glob
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
from requests.exceptions import RequestException


def _to_python(val):
    if pd.isna(val):
        return None
    if hasattr(val, "item"):
        return val.item()
    return val


def run_streamer(
    input_file,
    db_path="artifacts/stream/stream.db",
    rate=5,
    speed=1.0,
    endpoint="http://127.0.0.1:8000/score",
    post_to_api=True,
    max_events=None,
):
    df = pd.read_parquet(input_file)
    required = {"asset_id", "timestamp", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Dataset missing required columns: {required - set(df.columns)}"
        )

    df = df.sort_values("timestamp")

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS asset_state (
            asset_id TEXT PRIMARY KEY,
            timestamp TEXT,
            data TEXT,
            label INTEGER
        )
        """
    )
    conn.commit()

    count = 0
    for _, row in df.iterrows():
        event = {
            "asset_id": str(_to_python(row["asset_id"])),
            "timestamp": str(_to_python(row["timestamp"])),
            "label": int(_to_python(row["label"])),
            "features": {
                col: _to_python(row[col])
                for col in df.columns
                if col not in ("asset_id", "timestamp", "label")
            },
        }

        print(json.dumps(event))

        cur.execute(
            """
            INSERT OR REPLACE INTO asset_state (asset_id, timestamp, data, label)
            VALUES (?, ?, ?, ?)
            """,
            (
                event["asset_id"],
                event["timestamp"],
                json.dumps(event["features"], default=str),
                event["label"],
            ),
        )
        conn.commit()

        if post_to_api and endpoint:
            try:
                r = requests.post(endpoint, json=event, timeout=5.0)
                r.raise_for_status()
                resp = (
                    r.json()
                    if r.headers.get("content-type") == "application/json"
                    else {"status": "ok (no-json)"}
                )
                print("🔎 Scored:", resp)
            except RequestException as e:
                print(f"⚠️ API call failed: {e}")

        count += 1
        if max_events and count >= max_events:
            print(f"🛑 Stopped after {count} events (max-events reached)")
            break

        time.sleep(1.0 / (rate * speed))

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to processed parquet file.")
    parser.add_argument("--db", default="artifacts/stream/stream.db")
    parser.add_argument("--rate", type=int, default=5)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/score")
    parser.add_argument("--no-post", action="store_true")
    parser.add_argument(
        "--max-events", type=int, default=None, help="Limit number of streamed events."
    )
    args = parser.parse_args()

    if args.input:
        run_streamer(
            args.input,
            db_path=args.db,
            rate=args.rate,
            speed=args.speed,
            endpoint=args.endpoint,
            post_to_api=(not args.no_post),
            max_events=args.max_events,
        )
    else:
        test_files = glob.glob("data/processed/*_test.parquet")
        for fpath in test_files:
            dataset = Path(fpath).stem.replace("_test", "")
            print(f"🚀 Streaming dataset: {dataset}")
            run_streamer(
                fpath,
                db_path=f"artifacts/stream/{dataset}.db",
                rate=args.rate,
                speed=args.speed,
                endpoint=f"http://127.0.0.1:8000/score/{dataset}",
                post_to_api=(not args.no_post),
                max_events=args.max_events,
            )
