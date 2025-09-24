# CDT Security — prototype

Quickstart (venv):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.preprocess
pytest
python -m src.stream.streamer --input data/processed/iot_fridge_test.parquet --rate 1000 --speed 10.0
python -m src.models.run_all or python scripts/run_all.py
python -m src.models.infer --dataset iot_fridge --event event.json
python scripts/run_all_ganomaly.py
uvicorn src.api.main:app --reload
python scripts/test_api.py
uvicorn src.api.main:app --reload --port 8000
python -m src.stream.streamer --input data/processed/iot_fridge_test.parquet --rate 2 --speed 1.0 --endpoint http://127.0.0.1:8000/score
python scripts/query_db.py --tables
python scripts/query_db.py --alerts --limit 20
python scripts/query_db.py --tail --interval 2
Invoke-RestMethod -Uri http://127.0.0.1:8000/score -Method Post -Headers @{"Content-Type" = "application/json"} -Body '{"asset_id":"iot_fridge","timestamp":"2025-09-24T12:00:00Z","features":{"fridge_temperature":999}}'




