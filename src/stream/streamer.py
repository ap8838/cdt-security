import argparse
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
from requests.exceptions import RequestException


def _to_python(val):
    """Convert pandas/numpy scalars to plain Python types for JSON/DB safety."""
    if pd.isna(val):
        return None
    if hasattr(val, "item"):  # numpy scalar
        return val.item()
    return val


def run_streamer(
    input_file,
    db_path="artifacts/stream/stream.db",
    rate=5,
    speed=1.0,
    endpoint="http://127.0.0.1:8000/score",
    post_to_api=True,
):
    # 1. Load processed dataset
    df = pd.read_parquet(input_file)

    # 2. Validate expected columns
    required = {"asset_id", "timestamp", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Dataset missing required columns: {required - set(df.columns)}"
        )

    # 3. Sort chronologically for realistic replay
    df = df.sort_values("timestamp")

    # 4. Setup SQLite (asset_state)
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

    # 5. Stream loop
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

        # Print event (simulating real-time stream)
        print(json.dumps(event))

        # Update DB with latest asset state (local asset_state)
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

        # POST to inference API (if enabled)
        if post_to_api and endpoint:
            api_payload = {
                "asset_id": event["asset_id"],
                "timestamp": event["timestamp"],
                "features": event["features"],
            }
            try:
                r = requests.post(endpoint, json=api_payload, timeout=5.0)
                r.raise_for_status()
                try:
                    resp = r.json()
                except ValueError:
                    resp = {"status": "ok (no-json)"}
                print("🔎 Scored:", resp)
            except RequestException as e:
                print(f"⚠️ API call failed: {e}")

        # Rate control
        time.sleep(1.0 / (rate * speed))

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed parquet file")
    parser.add_argument(
        "--db", default="artifacts/stream/stream.db", help="SQLite DB path"
    )
    parser.add_argument("--rate", type=int, default=5, help="Events per second")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://127.0.0.1:8000/score",
        help="Inference API URL (POST)",
    )
    parser.add_argument(
        "--no-post", action="store_true", help="Don't POST to inference API"
    )
    args = parser.parse_args()

    run_streamer(
        args.input,
        db_path=args.db,
        rate=args.rate,
        speed=args.speed,
        endpoint=args.endpoint,
        post_to_api=(not args.no_post),
    )
