"""Feature contract shared by training and inference."""

from __future__ import annotations

import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:
    pa = None

FEATURE_COLUMNS = [
    "transaction_amount",
    "merchant_category",
    "country_risk_score",
    "account_age_days",
    "transaction_count_24h",
    "failed_auth_attempts_24h",
    "is_new_device",
    "hour_utc",
]


def validate_features(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing feature columns: {sorted(missing)}")
    if (frame["country_risk_score"].lt(0) | frame["country_risk_score"].gt(1)).any():
        raise ValueError("country_risk_score must be between 0 and 1")
    if (frame["hour_utc"].lt(0) | frame["hour_utc"].gt(23)).any():
        raise ValueError("hour_utc must be between 0 and 23")
    frame = frame[FEATURE_COLUMNS]
    schema = pandera_schema()
    return schema.validate(frame) if schema is not None else frame


def pandera_schema():
    """Return the optional Pandera schema for the mapped v1 contract."""
    if pa is None:
        return None
    return pa.DataFrameSchema(
        {
            "transaction_amount": pa.Column(float, checks=pa.Check.ge(0)),
            "country_risk_score": pa.Column(float, checks=pa.Check.in_range(0, 1)),
            "account_age_days": pa.Column(int, checks=pa.Check.ge(0)),
            "transaction_count_24h": pa.Column(int, checks=pa.Check.ge(0)),
            "failed_auth_attempts_24h": pa.Column(int, checks=pa.Check.ge(0)),
            "hour_utc": pa.Column(int, checks=pa.Check.in_range(0, 23)),
        },
        checks=None,
        coerce=True,
    )
