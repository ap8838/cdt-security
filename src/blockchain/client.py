# src/blockchain/client.py

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from web3 import Web3
from web3 import exceptions as w3_exceptions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = PROJECT_ROOT / "artifacts" / "blockchain" / "AlertRegistry.json"

load_dotenv(PROJECT_ROOT / ".env")

WEB3_PROVIDER = os.getenv("WEB3_PROVIDER", "http://127.0.0.1:8545")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
CHAIN_ID = int(os.getenv("CHAIN_ID", "1337"))


class BlockchainClient:
    def __init__(self, provider_url: str = None, private_key: str = None):
        self.provider_url = provider_url or WEB3_PROVIDER
        self.w3 = Web3(Web3.HTTPProvider(self.provider_url))

        if not self.w3.is_connected():
            raise RuntimeError(
                f"Could not connect to Web3 provider at {self.provider_url}"
            )

        if not ARTIFACT.exists():
            raise RuntimeError(f"Contract artifact not found: {ARTIFACT}")

        data = json.loads(ARTIFACT.read_text())

        self.contract_address = data.get("address")
        if not self.contract_address:
            raise RuntimeError(
                "Contract artifact has no address. Deploy contract first."
            )

        self.abi = data["abi"]
        self.contract = self.w3.eth.contract(
            address=self.contract_address,  # runtime-correct
            abi=self.abi,
        )

        self.private_key = private_key or PRIVATE_KEY
        if not self.private_key:
            raise RuntimeError("PRIVATE_KEY not set (required to send txns)")

        self.account = self.w3.eth.account.from_key(self.private_key)

    # ----------------------------------------------------
    @staticmethod
    def compute_alert_hash(raw_event: Dict[str, Any] | str) -> str:
        if isinstance(raw_event, str):
            try:
                raw_event = json.loads(raw_event)
            except Exception:
                return "0x" + "0" * 64

        canonical = json.dumps(raw_event, sort_keys=True, separators=(",", ":"))
        return Web3.keccak(text=canonical).hex()

    # ----------------------------------------------------

    def register_alert(
        self,
        alert_obj: dict,
        asset_id: str,
        synthetic: bool = False,
        dataset: str = None,
    ) -> str:
        asset_tag = f"{dataset}:{asset_id}" if dataset else asset_id

        alert_hash_hex = self.compute_alert_hash(alert_obj)

        # 🔒 Web3 typing bug – runtime value is valid
        alert_hash_bytes = self.w3.to_bytes(
            hexstr=alert_hash_hex  # type: ignore[arg-type]
        )

        ts = int(time.time())
        nonce = self.w3.eth.get_transaction_count(self.account.address)

        txn = self.contract.functions.registerAlert(
            alert_hash_bytes, ts, asset_tag, synthetic
        ).build_transaction(
            {
                "from": self.account.address,
                "nonce": nonce,
                "gas": 300000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": CHAIN_ID,
            }
        )

        signed = self.account.sign_transaction(txn)

        # 🔒 Web3 typing bug – return type is HexBytes at runtime
        tx_hash = self.w3.eth.send_raw_transaction(
            signed.raw_transaction  # type: ignore[arg-type]
        )

        self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        return tx_hash.hex()

    def lookup_alert(self, alert_hash_hex: str) -> Dict[str, Any]:
        if not alert_hash_hex.startswith("0x"):
            alert_hash_hex = "0x" + alert_hash_hex

        try:
            alert_hash_bytes = self.w3.to_bytes(
                hexstr=alert_hash_hex  # type: ignore[arg-type]
            )

            entry = self.contract.functions.getAlert(alert_hash_bytes).call()

            return {
                "alertHash": self.w3.to_hex(entry[0]),
                "timestamp": entry[1],
                "asset_id": entry[2],
                "synthetic": entry[3],
                "submitter": entry[4],
            }

        except w3_exceptions.ContractCustomError as err:
            return {"error": f"Contract Error: {str(err)}"}

        except Exception as general_err:
            return {"error": f"General Lookup Error: {str(general_err)}"}
