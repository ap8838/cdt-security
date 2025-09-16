import argparse
import json
import sqlite3
import time

import pandas as pd


def _to_python(val):
    """Convert pandas/numpy scalars to plain Python types for JSON/DB safety."""
    if pd.isna(val):
        return None
    if hasattr(val, "item"):  # numpy scalar
        return val.item()
    return val


def run_streamer(input_file, db_path="artifacts/stream/stream.db", rate=5, speed=1.0):
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

    # 4. Setup SQLite
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS asset_state (
            asset_id TEXT,
            timestamp TEXT,
            data TEXT,
            label INTEGER,
            PRIMARY KEY (asset_id)
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

        # Update DB with latest asset state
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
    args = parser.parse_args()

    run_streamer(args.input, args.db, args.rate, args.speed)
