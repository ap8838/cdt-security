"""
Simple DB inspector for artifacts/stream/stream.db
Usage examples:
  python scripts/query_db.py --db artifacts/stream/stream.db --tables
  python scripts/query_db.py --db artifacts/stream/stream.db --asset-state --limit 10
  python scripts/query_db.py --db artifacts/stream/stream.db --alerts --limit 20
  python scripts/query_db.py --db artifacts/stream/stream.db --tail --interval 2
"""

import argparse
import json
import sqlite3
import time
import pandas as pd


def list_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    rows = [r[0] for r in cur.fetchall()]
    print("Tables:", rows)


def show_asset_state(conn, limit=10):
    try:
        df = pd.read_sql_query(f"SELECT * FROM asset_state LIMIT {limit}", conn)
        print(df.to_string(index=False))
        # Show parsed JSON 'data' column for first rows (if present)
        if "data" in df.columns:
            print("\nParsed 'data' for the returned rows:")
            for i, r in df.head(limit).iterrows():
                try:
                    print(r["asset_id"], json.loads(r["data"]))
                except Exception:
                    print(r["asset_id"], "<could not parse data>")
    except Exception as e:
        print("Error reading asset_state:", e)


def show_alerts(conn, limit=10):
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM alerts ORDER BY id DESC LIMIT {limit}", conn
        )
        print(df.to_string(index=False))
    except Exception as e:
        print("Error reading alerts:", e)


def tail_alerts(conn, limit=5, interval=1.0):
    last_seen = None
    print("Tailing alerts (press Ctrl-C to stop)...")
    try:
        while True:
            df = pd.read_sql_query(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", conn, params=(limit,)
            )
            if df.empty:
                print("[no alerts]")
            else:
                max_id = int(df["id"].max())
                if last_seen is None:
                    # show last few first time
                    print(df.to_string(index=False))
                elif max_id > last_seen:
                    new_df = pd.read_sql_query(
                        "SELECT * FROM alerts WHERE id > ? ORDER BY id ASC",
                        conn,
                        params=(last_seen,),
                    )
                    if not new_df.empty:
                        print(new_df.to_string(index=False))
                last_seen = max_id
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


def pragma_table_info(conn, table):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    if not rows:
        print(f"No such table: {table}")
    else:
        print(f"Schema for {table}:")
        for r in rows:
            # cid, name, type, notnull, dflt_value, pk
            print(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="artifacts/stream/stream.db")
    p.add_argument("--tables", action="store_true")
    p.add_argument("--asset-state", action="store_true")
    p.add_argument("--alerts", action="store_true")
    p.add_argument("--tail", action="store_true")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument(
        "--interval", type=float, default=1.0, help="tail poll interval seconds"
    )
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        if args.tables:
            list_tables(conn)
        if args.asset_state:
            show_asset_state(conn, limit=args.limit)
        if args.alerts:
            show_alerts(conn, limit=args.limit)
        if args.tail:
            tail_alerts(conn, limit=args.limit, interval=args.interval)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
