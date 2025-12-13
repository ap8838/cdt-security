# src/api/main.py
import json
import sqlite3
from contextlib import asynccontextmanager
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """Ensure alerts table exists (add tx_hash and synthetic columns if missing)."""
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
            tx_hash TEXT,
            synthetic INTEGER DEFAULT 0
        )
        """
    )
    conn.commit()

    # For existing DBs: ensure column exists (safe to run repeatedly).
    cur.execute("PRAGMA table_info(alerts)")
    cols = [r[1] for r in cur.fetchall()]
    if "synthetic" not in cols:
        try:
            cur.execute("ALTER TABLE alerts ADD COLUMN synthetic INTEGER DEFAULT 0")
            conn.commit()
        except sqlite3.Error:
            # if ALTER fails for any reason, ignore
            pass


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
except Exception as bc_err:  # Use specific name for outer scope error
    print("⚠️ Blockchain not initialized:", bc_err)
    bc_client = None


# -----------------------------
# Helper Functions
# -----------------------------
def log_alert(result: dict, raw_event: dict):
    """Insert alert row into DB and blockchain (if enabled)."""
    if not result.get("is_anomaly"):
        return

    # detect synthetic marker inside posted event
    synthetic = False
    try:
        fe = raw_event.get("features", {})
        if isinstance(fe, dict) and fe.get("__synthetic") in (True, "true", "1", 1):
            synthetic = True
    except Exception:
        synthetic = False

    tx_hash = None
    # --- Try blockchain write ---
    if bc_client:
        try:
            # FIX: Use 'raw_event' (the input data) for hashing, not 'result' (the score)
            tx_hash = bc_client.register_alert(
                raw_event,  # ✅ CHANGED: Use raw_event for hashing
                result.get("asset_id"),
                synthetic=synthetic,
            )
            print(f"✅ Alert written to blockchain: {tx_hash}")
        except Exception as chain_err:  # Use specific name
            print("⚠️ Blockchain write failed:", chain_err)

    # --- Always log to DB (include synthetic column) ---
    conn = None
    try:
        conn = sqlite3.connect(str(ALERTS_DB))
        _ensure_alerts_table(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO alerts (asset_id, timestamp, score, model, threshold, is_anomaly, raw, tx_hash, synthetic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                int(bool(synthetic)),
            ),
        )
        conn.commit()
    except sqlite3.Error as db_err:  # Use specific exception type
        print("⚠️ DB logging error:", db_err)
    except Exception as general_err:  # Use specific name
        print("⚠️ Unexpected logging error:", general_err)
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
        params: List[Any] = (
            []
        )  # Explicitly type params as List[Any] to handle mixed types
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

        # Remove duplicate records entirely
        df = df.drop_duplicates(
            subset=["id", "asset_id", "timestamp", "score", "model"], keep="last"
        )

        # Sort after dedupe
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
        except Exception as file_err:
            return {"error": f"Failed to parse CSV: {file_err}"}

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
        except Exception as file_err:
            return {"error": f"Failed to parse JSON: {file_err}"}

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
        # The key change for this function is ensuring the synthetic marker is included
        features_dict = {cols[j]: float(fake[i][j]) for j in range(len(cols))}
        features_dict["__synthetic"] = True  # <-- Add synthetic marker here

        event = {
            "asset_id": dataset,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": features_dict,
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
            except (
                requests.exceptions.RequestException
            ) as req_err:  # Specific exception
                print("⚠️ POST error:", req_err)

    return {
        "dataset": dataset,
        "generated": n,
        "posted": posted,
        "anomalies_detected": anomalies,
        "message": "cGAN simulation completed",
    }


@app.get("/blockchain/verify")
def verify_blockchain_record(tx_hash: str) -> Dict[str, Any]:  # Explicit return type
    """
    Checks DB for the record associated with tx_hash, recomputes the canonical hash,
    and looks up the canonical hash on the blockchain.
    Returns a simplified verdict for the UI.
    """

    if not bc_client:
        return {"error": "Blockchain disabled", "verdict": "Disabled"}

    # -------------------------
    # 1. Lookup DB entry by tx_hash (Crucial step)
    # -------------------------
    conn = sqlite3.connect(str(ALERTS_DB))
    try:
        df = pd.read_sql_query(
            "SELECT * FROM alerts WHERE tx_hash = ? LIMIT 1", conn, params=[tx_hash]
        )
    finally:
        conn.close()

    if df.empty:
        return {
            "verdict": "DB Record Not Found",
            "message": f"The transaction hash '{tx_hash}' does not match any entry in the local database.",
            "tx_hash": tx_hash,
        }

    # Found DB entry
    row = df.iloc[0].to_dict()
    db_record = row
    raw_content = row.get("raw", "{}")
    recomputed_hash = bc_client.compute_alert_hash(raw_content)

    # Ensure the hash is 0x-prefixed for lookup
    if not recomputed_hash.startswith("0x"):
        recomputed_hash = "0x" + recomputed_hash

    # -------------------------
    # 2. Lookup using the CORRECT canonical alert-hash
    # -------------------------
    final_lookup = bc_client.lookup_alert(recomputed_hash)

    # Check if the lookup was successful (i.e., asset_id is not the default empty string)
    if final_lookup and final_lookup.get("asset_id"):
        return {
            "verdict": "Verified",
            "db_record": db_record,
            "onchain_record": final_lookup,
            "verified_hash": recomputed_hash,
            "message": "Record successfully verified on-chain using the canonical alert hash.",
        }
    else:
        # If the canonical hash lookup failed, the transaction failed or the hash calculation is still wrong.
        return {
            "verdict": "Verification Failed",
            "db_record": db_record,
            "verified_hash": recomputed_hash,
            "onchain_record": final_lookup,
            "message": "The canonical hash was not found on the blockchain. The transaction may have reverted or the alert hash computation failed.",
        }


@app.get("/adversarial/models")
def list_adversarial_models():
    """
    Return list of datasets that have saved cGAN generators:
    artifacts/adversarial/{dataset}_cgan.pt
    """
    pattern = "artifacts/adversarial/*_cgan.pt"
    files = glob(pattern)
    models = []
    for f in files:
        name = Path(f).stem  # e.g. 'iot_fridge_cgan'
        if name.endswith("_cgan"):
            ds = name[: -len("_cgan")]
            models.append(ds)
    return {"models": sorted(models)}


# Keeping the last function for completeness, though it seems unused/incomplete
def verify_hash_route(tx: str):
    return verify_blockchain_record(tx_hash=tx)
