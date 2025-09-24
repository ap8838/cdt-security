import json

import requests

API_URL = "http://127.0.0.1:8000/score"

sample_event = {
    "asset_id": "iot_fridge",
    "timestamp": "2025-09-22T12:00:00Z",
    "features": {
        "fridge_temperature": 5.4,
        "temp_condition": "cold",
        "type": "smart_fridge",
    },
}


def main():
    print(f"Sending request to {API_URL} ...")
    response = requests.post(API_URL, json=sample_event)

    if response.status_code == 200:
        print("✅ Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ Error {response.status_code}: {response.text}")


if __name__ == "__main__":
    main()
