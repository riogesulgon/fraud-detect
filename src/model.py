from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

FEATURES = [
    "transaction_amount",
    "merchant_category",
    "country_risk_score",
    "account_age_days",
    "transaction_count_24h",
    "failed_auth_attempts_24h",
    "is_new_device",
    "hour_utc",
]
CATEGORICAL = ["merchant_category"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
MODEL_PATH = Path("models/risk-model.joblib")


@dataclass(frozen=True)
class ModelInfo:
    model_version: str
    feature_schema_version: str
    decision_threshold: float
    training_rows: int
    metrics: dict[str, float]


def generate_data(n: int = 1000, seed: int = 42) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(seed)
    categories = np.array(["electronics", "grocery", "travel", "fashion", "gaming"])
    rows = []
    for _ in range(n):
        row = {
            "transaction_amount": round(float(rng.lognormal(4.2, 1.0)), 2),
            "merchant_category": str(rng.choice(categories)),
            "country_risk_score": round(float(rng.random()), 4),
            "account_age_days": int(rng.integers(1, 2500)),
            "transaction_count_24h": int(rng.poisson(4)),
            "failed_auth_attempts_24h": int(rng.poisson(0.4)),
            "is_new_device": bool(rng.integers(0, 2)),
            "hour_utc": int(rng.integers(0, 24)),
        }
        rows.append(row)
    scores = np.array(
        [
            r["country_risk_score"] * 1.8
            + r["is_new_device"] * 1.2
            + r["failed_auth_attempts_24h"] * 0.7
            + r["transaction_count_24h"] * 0.08
            + (r["transaction_amount"] > 250) * 0.7
            + (r["account_age_days"] < 30) * 0.8
            for r in rows
        ]
    )
    return rows, (scores > 1.8).astype(int)


def train(
    rows: list[dict], labels: np.ndarray, path: Path = MODEL_PATH, dataset_name: str = "synthetic"
) -> ModelInfo:
    pre = DictVectorizer(sparse=False)
    pipe = Pipeline(
        [
            ("preprocess", pre),
            ("classifier", HistGradientBoostingClassifier(max_iter=80, random_state=42)),
        ]
    )
    pipe.fit(rows, labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    version = dataset_name + "-" + hashlib.sha1(json.dumps(FEATURES).encode()).hexdigest()[:8]
    return ModelInfo(version, "1.0", 0.65, len(rows), {})


def load(path: Path = MODEL_PATH):
    if not path.exists():
        rows, labels = generate_data()
        train(rows, labels, path)
    return joblib.load(path)
