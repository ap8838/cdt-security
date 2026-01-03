# CDT Security — Prototype

---

##  Quickstart

```bash
# create & activate virtual environment
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\activate        # Windows PowerShell

# install dependencies
pip install -r requirements.txt
also install ui dependencies from ui/package.json
clean npm installation etc

###1. Preprocess dataset (Phase 1)

Generate processed parquet files + preprocessing artifacts.

python -m src.data.preprocess

2. Train models 

Autoencoder (AE):

python scripts/run_models.py ae --dataset iot_fridge
or
python scripts/run_models.py ae

p.s: Some datasets contain only normal samples in the test split after chronological partitioning. In such cases, ROC-AUC is undefined and anomaly detection metrics are reported as zero. This reflects real-world deployment scenarios where anomalies may be absent during evaluation windows.

GANomaly:

python scripts/run_models.py ganomaly --dataset linux_disk1 --window 5
# or run all
python scripts/run_models.py ganomaly --window 5

setup thresholds for both models with :
python scripts/compute_best_threshold.py --model ae
python scripts/compute_best_threshold.py --model ganomaly --window 5

# Evaluate AE (Pointwise)
python scripts/run_evals.py ae

# Evaluate GANomaly (Temporal Window = 5)
python scripts/run_evals.py ganomaly --window 5

3. Local inference test

Run a single event against a trained model.

python -m src.models.infer --dataset iot_fridge --event event.json

4.Frontend In a new terminal
cd ui
npm run dev

5. Start API 
uvicorn src.api.main:app --reload --port 8000


Endpoint: POST /score

Example input:

{
  "asset_id": "iot_fridge",
  "timestamp": "2025-09-24T12:00:00Z",
  "features": { "fridge_temperature": 999 }
}

6. Test API
python scripts/test_api.py


Manual PowerShell example:

Invoke-RestMethod -Uri http://127.0.0.1:8000/score -Method Post -Headers @{"Content-Type" = "application/json"} -Body '{"asset_id":"iot_fridge","timestamp":"2025-09-24T12:00:00Z","features":{"fridge_temperature":999}}'

7. Run streaming simulator 

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

8. Start Ganache

Run Ganache locally:

Create .env: 

# Backend blockchain connection
RPC_URL=http://127.0.0.1:8545
CONTRACT_ADDRESS=<Your Contact Address>
PRIVATE_KEY=<Your Private Key>
CHAIN_ID=1337

# Frontend API URL
VITE_API_BASE=http://localhost:8000

To run : 
npx ganache

It will display accounts and private keys.
Copy one of the private keys and set it in your .env file:

WEB3_PROVIDER=http://127.0.0.1:8545
PRIVATE_KEY=0x6947281498207d796e56b32de979b24ee4e1b2dc4fbe50b663b6f60875e9c257
CHAIN_ID=1337

9. Deploy the smart contract

Deploy the AlertRegistry contract to Ganache:

python src/blockchain/deploy_contract.py


You’ll see output like:

 Contract deployed at: 0x63783aA31b8A226F5E87aEDB5e60395560Ae122f
 Wrote contract artifact to: artifacts/blockchain/AlertRegistry.json

10.  Start the API
uvicorn src.api.main:app --reload --port 8000


You should see:

 Blockchain client initialized
 Alert written to blockchain: <tx_hash>

11. Generate and log an alert
Invoke-RestMethod -Uri http://127.0.0.1:8000/score -Method Post -Headers @{"Content-Type" = "application/json"} -Body '{"asset_id":"iot_fridge","timestamp":"2025-10-06T12:00:00Z","features":{"fridge_temperature":999}}'


Then check alerts:

Invoke-RestMethod -Uri http://127.0.0.1:8000/alerts


You’ll see:

tx_hash : 1cb0f6f09bed182d052c932880f5b39dba253b371119a70ccb2edd7bdd47dc72

12. Verify alert on blockchain

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

13. Train the cGAN
python scripts/run_all_cgan.py



14. Generate synthetic samples
python scripts/generate_all_cgan_samples.py

 Output:

Wrote artifacts/adversarial/generated.parquet



15. Evaluate generated samples
python scripts/eval_all_generated.py


 Output:

Wrote artifacts/adversarial/eval.csv


To view basic statistics:

import pandas as pd
df = pd.read_parquet("artifacts/adversarial/generated.parquet")
print(df.describe())

16. Run tests

pytest
