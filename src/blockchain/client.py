import json
import os
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = PROJECT_ROOT / "artifacts" / "blockchain" / "AlertRegistry.json"

load_dotenv(PROJECT_ROOT / ".env")

WEB3_PROVIDER = os.getenv("WEB3_PROVIDER", "http://127.0.0.1:8545")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
CHAIN_ID = int(os.getenv("CHAIN_ID", "1337"))


class BlockchainClient:
    def __init__(self, provider_url: str = None, private_key: str = None):
        """Initialize blockchain connection and load smart contract."""
        self.provider_url = provider_url or WEB3_PROVIDER
        self.w3 = Web3(Web3.HTTPProvider(self.provider_url))
        if not self.w3.is_connected():  # type: ignore[attr-defined]
            raise RuntimeError(
                f"Could not connect to Web3 provider at {self.provider_url}"
            )

        if not ARTIFACT.exists():
            raise RuntimeError(f"Contract artifact not found: {ARTIFACT}")

        data = json.loads(ARTIFACT.read_text())
        self.contract_address = data["address"]
        self.abi = data["abi"]
        self.contract = self.w3.eth.contract(
            address=self.contract_address, abi=self.abi
        )

        self.private_key = private_key or PRIVATE_KEY
        if not self.private_key:
            raise RuntimeError("PRIVATE_KEY not set (required to send txns)")

        self.account = self.w3.eth.account.from_key(self.private_key)

    def register_alert(
        self, alert_obj: dict, asset_id: str, dataset: str = None
    ) -> str:
        """
        Stores a deterministic keccak hash of alert_obj on-chain.
        Optionally includes dataset name in the asset_id tag for traceability.
        """
        asset_tag = f"{dataset}:{asset_id}" if dataset else asset_id

        alert_json = json.dumps(alert_obj, sort_keys=True, separators=(",", ":"))
        alert_hash = self.w3.keccak(text=alert_json)  # bytes32

        nonce = self.w3.eth.get_transaction_count(self.account.address)
        txn = self.contract.functions.registerAlert(
            alert_hash, asset_tag
        ).build_transaction(
            {
                "from": self.account.address,
                "nonce": nonce,
                "gas": 200000,
                "gasPrice": self.w3.to_wei("20", "gwei"),  # type: ignore[attr-defined]
                "chainId": CHAIN_ID,
            }
        )

        signed = self.account.sign_transaction(txn)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)

        # Wait for confirmation
        self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        return tx_hash.hex()

    def lookup_alert(self, alert_hash_hex: str):
        """
        Reads the on-chain entry (if any).
        `alert_hash_hex` should be 0x... hex string.
        """
        if not alert_hash_hex.startswith("0x"):
            alert_hash_hex = "0x" + alert_hash_hex

        alert_hash = self.w3.to_bytes(hexstr=alert_hash_hex)  # type: ignore[attr-defined]
        entry = self.contract.functions.getAlert(alert_hash).call()

        return {
            "alertHash": self.w3.to_hex(entry[0]) if entry[0] != b"" else None,  # type: ignore[attr-defined]
            "timestamp": entry[1],
            "asset_id": entry[2],
            "submitter": entry[3],
        }
