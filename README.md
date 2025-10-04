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

7. Start Ganache

Run Ganache locally:

npx ganache


It will display accounts and private keys.
Copy one of the private keys and set it in your .env file:

WEB3_PROVIDER=http://127.0.0.1:8545
PRIVATE_KEY=0x6947281498207d796e56b32de979b24ee4e1b2dc4fbe50b663b6f60875e9c257
CHAIN_ID=1337

8. Deploy the smart contract

Deploy the AlertRegistry contract to Ganache:

python src/blockchain/deploy_contract.py


You’ll see output like:

🎉 Contract deployed at: 0x63783aA31b8A226F5E87aEDB5e60395560Ae122f
💾 Wrote contract artifact to: artifacts/blockchain/AlertRegistry.json

9.  Start the API
uvicorn src.api.main:app --reload --port 8000


You should see:

✅ Blockchain client initialized
✅ Alert written to blockchain: <tx_hash>

10. Generate and log an alert
Invoke-RestMethod -Uri http://127.0.0.1:8000/score -Method Post -Headers @{"Content-Type" = "application/json"} -Body '{"asset_id":"iot_fridge","timestamp":"2025-10-06T12:00:00Z","features":{"fridge_temperature":999}}'


Then check alerts:

Invoke-RestMethod -Uri http://127.0.0.1:8000/alerts


You’ll see:

tx_hash : 1cb0f6f09bed182d052c932880f5b39dba253b371119a70ccb2edd7bdd47dc72

11. Verify alert on blockchain

Check the on-chain record for a given hash:

python -m src.blockchain.verify_alert 0x1cb0f6f09bed182d052c932880f5b39dba253b371119a70ccb2edd7bdd47dc72


Expected output:

On-chain record: {
  'alertHash': '0x1cb0f6f09bed182d052c932880f5b39dba253b371119a70ccb2edd7bdd47dc72',
  'timestamp': 0,
  'asset_id': '',
  'submitter': '0x0000000000000000000000000000000000000000'
}


(The default contract stores hashes only; extended metadata storage can be added later.)

12. Run tests

pytest
