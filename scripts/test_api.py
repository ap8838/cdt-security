import glob
import json
import os

import requests

BASE_URL = "http://127.0.0.1:8000/score"


def main():
    test_files = glob.glob("data/processed/*_test.parquet")
    if not test_files:
        print("❌ No test parquet files found.")
        return

    for test_file in test_files:
        dataset = os.path.basename(test_file).replace("_test.parquet", "")
        print(f"\n🚀 Testing API with dataset: {dataset}")
        payload = {
            "asset_id": dataset,
            "timestamp": "2025-09-22T12:00:00Z",
            "features": {"dummy": 1.0},
        }
        resp = requests.post(f"{BASE_URL}/{dataset}", json=payload)
        if resp.status_code == 200:
            print("✅ Response:", json.dumps(resp.json(), indent=2))
        else:
            print(f"❌ Error {resp.status_code}: {resp.text}")


if __name__ == "__main__":
    main()
