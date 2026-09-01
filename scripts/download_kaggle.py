"""Download the public Kaggle dataset without committing the 1M-row payload.

Requires Kaggle CLI credentials configured by the developer or CI environment.
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

DATASET = "harshjain123/synthetic-credit-card-fraud-interpretable"
OUT = Path(os.getenv("KAGGLE_DATA_DIR", "data/raw"))

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", str(OUT), "--unzip"], check=True)
    print(f"Downloaded {DATASET} into {OUT}; inspect columns before training.")

if __name__ == "__main__":
    main()
