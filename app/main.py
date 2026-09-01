from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "risk-model.joblib"
EVALUATION_PATH = ROOT / "reports" / "evaluation.json"
THRESHOLD = 0.65
FEATURE_SCHEMA_VERSION = "1.0"


class RiskRequest(BaseModel):
    transaction_amount: float = Field(ge=0)
    merchant_category: str = Field(min_length=1, max_length=64)
    country_risk_score: float = Field(ge=0, le=1)
    account_age_days: int = Field(ge=0)
    transaction_count_24h: int = Field(ge=0)
    failed_auth_attempts_24h: int = Field(ge=0)
    is_new_device: bool
    hour_utc: int = Field(ge=0, le=23)


class RiskResponse(BaseModel):
    risk_score: float
    risk_band: str
    model_version: str
    decision_threshold: float


app = FastAPI(
    title="MLOps Risk Platform",
    description="Demonstrator only; not suitable for live financial decisions.",
)
_requests = 0
_model: Any = None


def _load_model() -> Any:
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model


def _metadata() -> dict[str, Any]:
    if EVALUATION_PATH.exists():
        return json.loads(EVALUATION_PATH.read_text())
    return {"dataset": "unknown", "model_version": "untrained", "metrics": {}}


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok" if _load_model() is not None else "degraded",
        "model_loaded": _load_model() is not None,
    }


@app.get("/v1/model-info")
def model_info() -> dict[str, Any]:
    meta = _metadata()
    metrics = meta.get("metrics", meta.get("selected_model", {}))
    return {
        "model_version": meta["model_version"],
        "training_timestamp": None,
        "metrics": metrics,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset": meta["dataset"],
        "decision_threshold": meta.get("decision_threshold", THRESHOLD),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    version = _metadata()["model_version"]
    return PlainTextResponse(
        f'# HELP risk_score_requests_total Total risk scoring requests.\n# TYPE risk_score_requests_total counter\nrisk_score_requests_total {_requests}\nmodel_version{{version="{version}"}} 1\n'
    )


@app.post("/v1/risk-score", response_model=RiskResponse)
def risk_score(req: RiskRequest) -> RiskResponse:
    global _requests
    model = _load_model()
    if model is None:
        raise RuntimeError("model artifact is not available")
    _requests += 1
    score = float(model.predict_proba([req.model_dump()])[0, 1])
    score = round(min(max(score, 0.0), 1.0), 6)
    threshold = float(_metadata().get("decision_threshold", THRESHOLD))
    band = "high" if score >= threshold else "medium" if score >= 0.35 else "low"
    return RiskResponse(
        risk_score=score,
        risk_band=band,
        model_version=_metadata()["model_version"],
        decision_threshold=threshold,
    )


FULL_MODEL_PATH = ROOT / "models" / "full-feature-calibrated.joblib"
_full_artifact: dict[str, Any] | None = None


class RiskV2Request(BaseModel):
    transaction_id: str | int
    customer_id: str | int
    timestamp_seconds: int = Field(ge=0)
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: int = Field(ge=0, le=1)
    is_night: int = Field(ge=0, le=1)
    amount: float = Field(ge=0)
    avg_amount_30d: float = Field(ge=0)
    amount_to_avg_ratio: float = Field(ge=0)
    customer_age: int = Field(ge=0)
    customer_tenure_days: int = Field(ge=0)
    account_balance: float = Field(ge=0)
    income_band: int = Field(ge=0)
    merchant_category: str = Field(min_length=1, max_length=64)
    merchant_id: str | int
    transaction_type: str = Field(min_length=1, max_length=64)
    card_present: int = Field(ge=0, le=1)
    device_type: str = Field(min_length=1, max_length=64)
    num_transactions_last_1h: int = Field(ge=0)
    num_transactions_last_24h: int = Field(ge=0)
    minutes_since_last_transaction: float = Field(ge=0)
    distance_from_home_km: float = Field(ge=0)
    distance_from_last_transaction_km: float = Field(ge=0)
    is_foreign_transaction: int = Field(ge=0, le=1)
    ip_address_risk_score: float = Field(ge=0, le=1)
    failed_pin_attempts_24h: int = Field(ge=0)


def _load_full_artifact() -> dict[str, Any] | None:
    global _full_artifact
    if _full_artifact is None and FULL_MODEL_PATH.exists():
        _full_artifact = joblib.load(FULL_MODEL_PATH)
    return _full_artifact


@app.post("/v2/risk-score", response_model=RiskResponse)
def risk_score_v2(req: RiskV2Request) -> RiskResponse:
    global _requests
    artifact = _load_full_artifact()
    if artifact is None:
        raise RuntimeError("calibrated full-feature model artifact is not available")
    frame = pd.DataFrame([req.model_dump()]).drop(columns=["transaction_id", "customer_id"])
    frame = pd.get_dummies(
        frame, columns=frame.select_dtypes(include=["object"]).columns, dtype=float
    ).fillna(0)
    frame = frame.reindex(columns=artifact["feature_columns"], fill_value=0)
    raw = artifact["base_model"].predict_proba(frame)[:, 1]
    score = round(float(artifact["calibrator"].predict_proba(raw.reshape(-1, 1))[0, 1]), 6)
    threshold = float(artifact["threshold"])
    _requests += 1
    band = "high" if score >= threshold else "medium" if score >= threshold / 2 else "low"
    return RiskResponse(
        risk_score=score,
        risk_band=band,
        model_version="kaggle-full-calibrated",
        decision_threshold=threshold,
    )
