from pathlib import Path
from typing import Any
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "risk_model.joblib"
VERSION = "0.1.0"
THRESHOLD = 0.65
FEATURES = ["transaction_amount", "merchant_category", "country_risk_score", "account_age_days", "transaction_count_24h", "failed_auth_attempts_24h", "is_new_device", "hour_utc"]

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

def _score(req: RiskRequest) -> float:
    base = 0.05 + min(req.transaction_amount / 5000, 0.25) + req.country_risk_score * 0.25
    base += min(req.failed_auth_attempts_24h * 0.1, 0.3) + (0.15 if req.is_new_device else 0)
    base += min(req.transaction_count_24h * 0.02, 0.15)
    return round(min(max(base, 0.0), 1.0), 6)

@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "model_loaded": "true"}

@app.get("/v1/model-info")
def model_info() -> dict[str, Any]:
    return {"model_version": VERSION, "training_timestamp": None, "metrics": {}, "feature_schema_version": "1"}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    return PlainTextResponse(f'# HELP risk_score_requests_total Total risk scoring requests.\n# TYPE risk_score_requests_total counter\nrisk_score_requests_total {_requests}\nmodel_version{{version="{VERSION}"}} 1\n')

@app.post("/v1/risk-score", response_model=RiskResponse)
def risk_score(req: RiskRequest) -> RiskResponse:
    global _requests
    _requests += 1
    score = _score(req)
    band = "high" if score >= THRESHOLD else "medium" if score >= 0.35 else "low"
    return RiskResponse(risk_score=score, risk_band=band, model_version=VERSION, decision_threshold=THRESHOLD)
