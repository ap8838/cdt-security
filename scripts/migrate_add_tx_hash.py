import sqlite3
from pathlib import Path

DB = Path("artifacts/stream/stream.db")


def column_exists(conn, table, col):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(%s)" % table)
    cols = [r[1] for r in cur.fetchall()]
    return col in cols


def main():
    if not DB.exists():
        print("DB not found:", DB)
        return
    conn = sqlite3.connect(str(DB))
    try:
        if not column_exists(conn, "alerts", "tx_hash"):
            print("Adding tx_hash column to alerts table...")
            conn.execute("ALTER TABLE alerts ADD COLUMN tx_hash TEXT")
            conn.commit()
            print("Done.")
        else:
            print("tx_hash column already exists.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
