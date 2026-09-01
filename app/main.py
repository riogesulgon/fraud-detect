from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import joblib
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

app = FastAPI(title="MLOps Risk Platform", description="Demonstrator only; not suitable for live financial decisions.")
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
    return {"status": "ok" if _load_model() is not None else "degraded", "model_loaded": _load_model() is not None}

@app.get("/v1/model-info")
def model_info() -> dict[str, Any]:
    meta = _metadata()
    return {"model_version": meta["model_version"], "training_timestamp": None, "metrics": meta["metrics"], "feature_schema_version": FEATURE_SCHEMA_VERSION, "dataset": meta["dataset"], "decision_threshold": meta.get("decision_threshold", THRESHOLD)}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    version = _metadata()["model_version"]
    return PlainTextResponse(f'# HELP risk_score_requests_total Total risk scoring requests.\n# TYPE risk_score_requests_total counter\nrisk_score_requests_total {_requests}\nmodel_version{{version="{version}"}} 1\n')

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
    return RiskResponse(risk_score=score, risk_band=band, model_version=_metadata()["model_version"], decision_threshold=threshold)
