import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from src.service.infer_service import InferenceService


# -----------------------
# Request schema
# -----------------------
class Event(BaseModel):
    asset_id: str
    timestamp: str
    features: dict


# -----------------------
# DB setup
# -----------------------
ALERTS_DB = Path("artifacts/stream/stream.db")


def _ensure_alerts_table(conn):
    cur = conn.cursor()
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
            raw TEXT
        )
        """
    )
    conn.commit()


# -----------------------
# Lifespan (startup/shutdown)
# -----------------------
@asynccontextmanager
async def lifespan(app_instance: FastAPI):  # <-- renamed param to avoid shadow
    # Startup: ensure DB + table exist
    ALERTS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ALERTS_DB))
    _ensure_alerts_table(conn)
    conn.close()

    yield  # API runs here

    # Shutdown: nothing extra for now


# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Anomaly Detection API", lifespan=lifespan)

# Initialize model service
service = InferenceService(dataset="iot_fridge", model_type="ganomaly")


# -----------------------
# DB logging
# -----------------------
def log_alert(result: dict, raw_event: dict):
    """Insert an alert row if anomaly was detected"""
    try:
        if not result.get("is_anomaly"):
            return

        conn = sqlite3.connect(str(ALERTS_DB))
        _ensure_alerts_table(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO alerts (asset_id, timestamp, score, model, threshold, is_anomaly, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.get("asset_id"),
                result.get("ts"),
                float(result.get("score", 0.0)),
                result.get("model"),
                float(result.get("threshold", 0.0)),
                int(bool(result.get("is_anomaly"))),
                json.dumps(raw_event, default=str),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("⚠️ Failed to log alert:", e)


# -----------------------
# API endpoint
# -----------------------
@app.post("/score")
def score_event(event: Event):
    payload = {"asset_id": event.asset_id, "timestamp": event.timestamp}
    payload.update(event.features or {})
    result = service.score(payload)

    try:
        log_alert(result, event.model_dump())
    except Exception as e:
        print("⚠️ log_alert failed:", e)

    return result
