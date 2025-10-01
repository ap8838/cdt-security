# CDT Security — Prototype

---

## 🚀 Quickstart

```bash
# create & activate virtual environment
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\activate        # Windows PowerShell

# install dependencies
pip install -r requirements.txt


###1. Preprocess dataset (Phase 1)

Generate processed parquet files + preprocessing artifacts.

python -m src.data.preprocess

2. Train models 

Autoencoder (AE):

python -m src.models.run_all
# or
python scripts/run_all.py


GANomaly:

python scripts/run_all_ganomaly.py

3. Local inference test

Run a single event against a trained model.

python -m src.models.infer --dataset iot_fridge --event event.json

4. Start API 
uvicorn src.api.main:app --reload --port 8000


Endpoint: POST /score

Example input:

{
  "asset_id": "iot_fridge",
  "timestamp": "2025-09-24T12:00:00Z",
  "features": { "fridge_temperature": 999 }
}

5. Test API
python scripts/test_api.py


Manual PowerShell example:

Invoke-RestMethod -Uri http://127.0.0.1:8000/score -Method Post -Headers @{"Content-Type" = "application/json"} -Body '{"asset_id":"iot_fridge","timestamp":"2025-09-24T12:00:00Z","features":{"fridge_temperature":999}}'

6. Run streaming simulator 

Without API (DB + stdout only):

python -m src.stream.streamer --input data/processed/iot_fridge_test.parquet --rate 10 --speed 1.0


With API integration (alerts stored in DB):

python -m src.stream.streamer --input data/processed/iot_fridge_test.parquet --rate 2 --speed 1.0 --endpoint http://127.0.0.1:8000/score

Inspect database 

Alerts + asset states are stored in artifacts/stream/stream.db.

# list tables
python scripts/query_db.py --tables

# view alerts
python scripts/query_db.py --alerts --limit 20

# tail alerts in real-time
python scripts/query_db.py --tail --interval 2

# view asset_state
python scripts/query_db.py --asset-state --limit 10

7. Run tests

pytest
