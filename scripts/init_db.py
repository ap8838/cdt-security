import sqlite3
import os

# Ensure the directory exists
os.makedirs("artifacts/stream", exist_ok=True)

conn = sqlite3.connect("artifacts/stream/stream.db")
cur = conn.cursor()

# Combined schema with tx_hash included
cur.execute(
    """
CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  asset_id TEXT,
  timestamp TEXT,
  score REAL,
  model TEXT,
  threshold REAL,
  is_anomaly INTEGER,
  tx_hash TEXT,
  raw TEXT
)
"""
)

conn.commit()
conn.close()
print(" Alerts table created .")