import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
ASSET_STATE_DB = ALERTS_DB  # same SQLite file


def _ensure_alerts_table(conn: sqlite3.Connection):
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
async def lifespan(_app: FastAPI):  # `_app` required but unused
    ALERTS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ALERTS_DB))
    _ensure_alerts_table(conn)
    conn.close()
    yield
    # nothing on shutdown


# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Anomaly Detection API", lifespan=lifespan)

# Enable CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, restrict to UI domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve built UI if exists
ui_dist = Path("ui/dist")
if ui_dist.exists():
    app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="ui")

# Initialize inference service
service = InferenceService(dataset="iot_fridge", model_type="ganomaly")


# -----------------------
# DB logging
# -----------------------
def log_alert(result: dict, raw_event: dict):
    """Insert an alert row if anomaly detected."""
    if not result.get("is_anomaly"):
        return

    conn = None
    try:
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
    except sqlite3.Error as db_err:
        print("⚠️ DB logging error:", db_err)
    except Exception as err:
        print("⚠️ Unexpected logging error:", err)
    finally:
        if conn is not None:
            conn.close()


@app.get("/alerts")
def get_alerts(limit: int = Query(100, ge=1, le=1000), since_id: int = 0):
    """Return latest alerts."""
    conn = None
    try:
        conn = sqlite3.connect(str(ALERTS_DB))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM alerts WHERE id > ? ORDER BY id DESC LIMIT ?",
            (since_id, limit),
        )
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            if "raw" in r and r["raw"]:
                try:
                    r["raw"] = json.loads(r["raw"])
                except json.JSONDecodeError:
                    pass
        return rows
    finally:
        if conn is not None:
            conn.close()


@app.get("/assets")
def get_assets():
    """Return asset IDs and their last seen timestamp."""
    conn = None
    try:
        conn = sqlite3.connect(str(ASSET_STATE_DB))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT asset_id, MAX(timestamp) as last_ts FROM asset_state GROUP BY asset_id"
        )
        rows = [dict(r) for r in cur.fetchall()]
        return rows
    except sqlite3.OperationalError:
        return []
    finally:
        if conn is not None:
            conn.close()


# -----------------------
# API endpoints
# -----------------------
@app.post("/score")
def score_event(event: Event):
    """Run inference on a single event and log anomaly if detected."""
    payload = {"asset_id": event.asset_id, "timestamp": event.timestamp}
    payload.update(event.features or {})
    result = service.score(payload)

    try:
        log_alert(result, event.model_dump())
    except Exception as err:
        print("⚠️ log_alert failed:", err)

    return result
