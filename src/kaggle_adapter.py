"""Adapter from the Kaggle 27-feature dataset to the v1 API feature contract."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

KAGGLE_COLUMNS = [
    "transaction_id", "customer_id", "timestamp_seconds", "hour_of_day", "day_of_week",
    "is_weekend", "is_night", "amount", "avg_amount_30d", "amount_to_avg_ratio",
    "customer_age", "customer_tenure_days", "account_balance", "income_band",
    "merchant_category", "merchant_id", "transaction_type", "card_present", "device_type",
    "num_transactions_last_1h", "num_transactions_last_24h", "minutes_since_last_transaction",
    "distance_from_home_km", "distance_from_last_transaction_km", "is_foreign_transaction",
    "ip_address_risk_score", "failed_pin_attempts_24h", "is_fraud",
]
TARGET = "is_fraud"


def load_kaggle(path: Path, sample_rows: int | None = 100_000, seed: int = 42) -> tuple[list[dict], pd.Series]:
    """Load and map Kaggle rows. Sampling bounds local training memory and is deterministic."""
    frame = pd.read_csv(path)
    if sample_rows is not None and len(frame) > sample_rows:
        frame = frame.sample(sample_rows, random_state=seed)
    required = set(KAGGLE_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Kaggle dataset missing columns: {sorted(missing)}")
    mapped = pd.DataFrame({
        "transaction_amount": frame["amount"].astype(float),
        "merchant_category": frame["merchant_category"].astype(str),
        "country_risk_score": frame["ip_address_risk_score"].astype(float).clip(0, 1),
        "account_age_days": frame["customer_tenure_days"].astype(int).clip(lower=0),
        "transaction_count_24h": frame["num_transactions_last_24h"].astype(int).clip(lower=0),
        "failed_auth_attempts_24h": frame["failed_pin_attempts_24h"].astype(int).clip(lower=0),
        "is_new_device": frame["customer_tenure_days"].astype(int).lt(30),
        "hour_utc": frame["hour_of_day"].astype(int).mod(24),
    })
    return mapped.to_dict(orient="records"), frame[TARGET].astype(int)
