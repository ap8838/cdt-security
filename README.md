# CDT Security — prototype

Quickstart (venv):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.preprocess
pytest
python -m src.stream.streamer --input data/processed/iot_fridge_test.parquet --rate 1000 --speed 10.0
python -m src.models.run_all
python -m src.models.infer --dataset iot_fridge --event event.json


