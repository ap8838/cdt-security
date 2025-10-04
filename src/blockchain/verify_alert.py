import sys

from src.blockchain.client import BlockchainClient


def main(alert_hash_hex: str):
    bc = BlockchainClient()
    info = bc.lookup_alert(alert_hash_hex)
    print("On-chain record:", info)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.blockchain.verify_alert <0xalert_hash>")
    else:
        main(sys.argv[1])
