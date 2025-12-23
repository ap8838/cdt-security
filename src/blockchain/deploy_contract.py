import json
import os
from pathlib import Path
from dotenv import load_dotenv
from solcx import compile_standard, install_solc
from web3 import Web3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "src" / "blockchain" / "contracts" / "AlertRegistry.sol"
OUT_DIR = PROJECT_ROOT / "artifacts" / "blockchain"

load_dotenv(PROJECT_ROOT / ".env")

WEB3_PROVIDER = os.getenv("WEB3_PROVIDER", "http://127.0.0.1:8545")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
CHAIN_ID = int(os.getenv("CHAIN_ID", "1337"))


def compile_contract():
    print("Installing solc and compiling contract...")
    install_solc("0.8.17")
    source = CONTRACT_PATH.read_text()
    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {"AlertRegistry.sol": {"content": source}},
            "settings": {
                "outputSelection": {"*": {"*": ["abi", "evm.bytecode.object"]}}
            },
        },
        solc_version="0.8.17",
    )
    contract_key = compiled["contracts"]["AlertRegistry.sol"]["AlertRegistry"]
    contract_abi = contract_key["abi"]
    contract_bytecode = contract_key["evm"]["bytecode"]["object"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "AlertRegistry.compiled.json").write_text(json.dumps(compiled))
    (OUT_DIR / "AlertRegistry.abi.json").write_text(json.dumps(contract_abi))

    return contract_abi, contract_bytecode


def deploy(contract_abi, contract_bytecode):
    if not PRIVATE_KEY:
        raise RuntimeError(
            "PRIVATE_KEY not set in .env (needed to send deployment txn)"
        )

    w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
    acct = w3.eth.account.from_key(PRIVATE_KEY)
    contract = w3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)

    nonce = w3.eth.get_transaction_count(acct.address)
    tx = contract.constructor().build_transaction(
        {
            "from": acct.address,
            "nonce": nonce,
            "gas": 2000000,
            "gasPrice": w3.to_wei("20", "gwei"),
            "chainId": CHAIN_ID,
        }
    )
    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print("Sent deployment tx:", tx_hash.hex())

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    address = receipt["contractAddress"]
    print("Contract deployed at:", address)

    # Save artifact
    artifact = {
        "address": address,
        "abi": contract_abi,
    }
    (OUT_DIR / "AlertRegistry.json").write_text(json.dumps(artifact, indent=2))
    print("Wrote contract artifact to:", OUT_DIR / "AlertRegistry.json")
    return address


if __name__ == "__main__":
    abi, bytecode = compile_contract()
    deploy(abi, bytecode)
