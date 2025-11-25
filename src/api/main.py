# src/api/main.py
import json
import sqlite3
from contextlib import asynccontextmanager
from glob import glob
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.blockchain.client import BlockchainClient
from src.service.infer_service import InferenceService


# -----------------------------
# Base Models
# -----------------------------
class Event(BaseModel):
    asset_id: str
    timestamp: str
    features: dict


# -----------------------------
# Database setup
# -----------------------------
ALERTS_DB = Path("artifacts/stream/stream.db")
ASSET_STATE_DB = ALERTS_DB


def _ensure_alerts_table(conn: sqlite3.Connection):
    """Ensures the alerts table exists."""
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
            raw TEXT,
            tx_hash TEXT
        )
        """
    )
    conn.commit()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """FastAPI lifecycle manager."""
    ALERTS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ALERTS_DB))
    _ensure_alerts_table(conn)
    conn.close()
    yield


# -----------------------------
# App Initialization
# -----------------------------
app = FastAPI(title="Anomaly Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend if built
ui_dist = Path("ui/dist")
if ui_dist.exists():
    app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="ui")

# Blockchain
try:
    bc_client = BlockchainClient()
    print("✅ Blockchain client initialized")
except Exception as e:
    print("⚠️ Blockchain not initialized:", e)
    bc_client = None


# -----------------------------
# Helper Functions
# -----------------------------
def log_alert(result: dict, raw_event: dict):
    """Insert alert row into DB and blockchain (if enabled)."""
    if not result.get("is_anomaly"):
        return

    tx_hash = None
    conn = None

    # Write to blockchain
    if bc_client:
        try:
            tx_hash = bc_client.register_alert(result, result.get("asset_id"))
            print(f"✅ Alert written to blockchain: {tx_hash}")
        except Exception as chain_err:
            print("⚠️ Blockchain write failed:", chain_err)

    try:
        conn = sqlite3.connect(str(ALERTS_DB))
        _ensure_alerts_table(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO alerts (asset_id, timestamp, score, model, threshold, is_anomaly, raw, tx_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.get("asset_id"),
                result.get("ts"),
                float(result.get("score", 0.0)),
                result.get("model"),
                float(result.get("threshold", 0.0)),
                int(bool(result.get("is_anomaly"))),
                json.dumps(raw_event, default=str),
                tx_hash or "",
            ),
        )
        conn.commit()
    except Exception as err:
        print("⚠️ DB logging error:", err)
    finally:
        if conn is not None:
            conn.close()


# -----------------------------
# API ROUTES
# -----------------------------
@app.post("/score/{dataset}")
def score_event_dataset(dataset: str, event: Event):
    payload = {"asset_id": event.asset_id, "timestamp": event.timestamp}
    payload.update(event.features or {})

    dyn_service = InferenceService(dataset=dataset, model_type="ganomaly")
    result = dyn_service.score(payload)
    log_alert(result, event.model_dump())
    return result


@app.post("/score")
def score_event(event: Event, dataset: str = Query(None)):
    payload = {"asset_id": event.asset_id, "timestamp": event.timestamp}
    payload.update(event.features or {})

    if dataset:
        model = InferenceService(dataset, model_type="ganomaly")
        result = model.score(payload)
        log_alert(result, event.model_dump())
        return result

    asset_prefix = event.asset_id.split("_")[0] if "_" in event.asset_id else None
    available_datasets = [
        Path(p).stem.replace("_ae", "") for p in glob("artifacts/models/*_ae.pt")
    ]
    dataset = (
        asset_prefix if asset_prefix in available_datasets else available_datasets[0]
    )

    dyn_service = InferenceService(dataset=dataset, model_type="ganomaly")
    result = dyn_service.score(payload)
    log_alert(result, event.model_dump())
    return result


@app.get("/datasets")
def list_datasets():
    files = glob("artifacts/preproc/*_features.json")
    ds = [Path(f).stem.replace("_features", "") for f in files]
    return {"datasets": sorted(ds)}


@app.get("/alerts")
def get_alerts(
    limit: int = 200, offset: int = 0, dataset: Optional[str] = None, since_id: int = 0
):
    conn = sqlite3.connect(str(ALERTS_DB))
    try:
        base_q = "SELECT * FROM alerts"
        params = []
        where = []

        if dataset:
            where.append("asset_id LIKE ?")
            params.append(f"{dataset}%")

        if since_id and since_id > 0:
            where.append("id > ?")
            params.append(int(since_id))

        where_clause = f" WHERE {' AND '.join(where)}" if where else ""
        order_clause = " ORDER BY id DESC"
        limit_clause = " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        q = base_q + where_clause + order_clause + limit_clause
        df = pd.read_sql_query(q, conn, params=params)

        # Clean NaN
        df = df.where(pd.notnull(df), None)

        # 🔥 ABSOLUTE FIX: Remove duplicate records entirely
        df = df.drop_duplicates(
            subset=["id", "asset_id", "timestamp", "score", "model"], keep="last"
        )

        # 🔥 Sort after dedupe
        df = df.sort_values("id", ascending=False)

        return df.to_dict(orient="records")
    finally:
        conn.close()


@app.get("/metrics")
def get_metrics():
    """
    Return aggregate metrics from CSV or JSON, with NaN cleaned.
    """
    csv_path = Path("artifacts/reports/aggregate_metrics.csv")
    json_path = Path("artifacts/reports/aggregate_metrics.json")

    # Prefer CSV
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df = df.where(pd.notnull(df), None)
            return {"rows": df.to_dict(orient="records")}
        except Exception as e:
            return {"error": f"Failed to parse CSV: {e}"}

    # Fallback to JSON
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())

            def fix_nan(x):
                if isinstance(x, float) and (x != x):
                    return None
                return x

            cleaned = [{k: fix_nan(v) for k, v in row.items()} for row in data]
            return {"rows": cleaned}
        except Exception as e:
            return {"error": f"Failed to parse JSON: {e}"}

    return {"error": "No metrics file found"}


@app.get("/assets")
def get_assets():
    conn = sqlite3.connect("artifacts/stream/stream.db")
    df = pd.read_sql_query("SELECT DISTINCT asset_id FROM alerts", conn)
    conn.close()
    return df.to_dict(orient="records")


@app.post("/adversarial/generate")
def generate_adversarial_samples(
    dataset: str = "iot_fridge", n: int = 50, post_to_api: bool = True
):
    """
    Generate synthetic cGAN attacks and optionally POST them to /score.
    """

    import json
    from datetime import datetime, timezone

    import numpy as np
    import torch

    from src.adversarial.generate_samples import load_generator

    model_path = Path(f"artifacts/adversarial/{dataset}_cgan.pt")
    features_path = Path(f"artifacts/preproc/{dataset}_features.json")

    if not model_path.exists():
        return {"error": f"Model not found: {model_path}"}

    if not features_path.exists():
        return {"error": f"Features not found: {features_path}"}

    # Load cGAN generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen, info = load_generator(model_path, device)

    with open(features_path, "r") as f:
        features = json.load(f)

    cols = [
        c
        for c in features["all"]
        if c not in ("asset_id", "asset", "timestamp", "label")
    ]

    # Condition vector
    cond = np.zeros((n, info["cond_dim"]), dtype=np.float32)

    # Noise
    z = torch.randn(n, info["z_dim"]).to(device)
    cond_t = torch.from_numpy(cond).to(device)

    with torch.no_grad():
        fake = gen(z, cond_t).cpu().numpy()

    # If model outputs [-1,1] → map back to [0,1]
    if fake.min() >= -1 and fake.max() <= 1:
        fake = (fake + 1) / 2

    # Build events
    events = []
    for i in range(n):
        event = {
            "asset_id": dataset,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": {cols[j]: float(fake[i][j]) for j in range(len(cols))},
        }
        events.append(event)

    score_url = f"http://localhost:8000/score?dataset={dataset}"

    posted = 0
    anomalies = 0

    if post_to_api:
        import requests

        for ev in events:
            try:
                resp = requests.post(score_url, json=ev, timeout=5)
                if resp.status_code == 200:
                    posted += 1
                    r = resp.json()
                    if r.get("is_anomaly"):
                        anomalies += 1
                else:
                    print("⚠️ POST failed:", resp.status_code)
            except Exception as e:
                print("⚠️ POST error:", e)

    return {
        "dataset": dataset,
        "generated": n,
        "posted": posted,
        "anomalies_detected": anomalies,
        "message": "cGAN simulation completed",
    }


@app.get("/blockchain/verify")
def verify_blockchain_record(tx_hash: str):
    """
    Return both onchain lookup and DB metadata for the tx_hash (if present).
    """
    result = {}

    # on-chain look-up
    if not bc_client:
        result["onchain_error"] = "Blockchain not connected"
    else:
        try:
            onchain = bc_client.lookup_alert(tx_hash)
            result["onchain"] = onchain
        except Exception as e:
            result["onchain_error"] = str(e)

    # DB record (best-effort)
    conn = sqlite3.connect(str(ALERTS_DB))
    try:
        df = pd.read_sql_query(
            "SELECT * FROM alerts WHERE tx_hash = ? LIMIT 1", conn, params=(tx_hash,)
        )
        if not df.empty:
            row = df.iloc[0].to_dict()
            try:
                row["raw_parsed"] = json.loads(row.get("raw", "{}"))
            except Exception:
                row["raw_parsed"] = row.get("raw")
            result["db"] = row
        else:
            result["db"] = None
    finally:
        conn.close()

    return result


def verify_hash_route(tx: str):
    return verify_blockchain_record(tx_hash=tx)
