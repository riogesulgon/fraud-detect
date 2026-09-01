from __future__ import annotations

from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.model import MODEL_PATH, load

THRESHOLD = 0.65
MODEL_VERSION = "synthetic-baseline-1"
app = FastAPI(title="MLOps Risk Platform", version="0.1.0", description="Demonstrator only. Not suitable for live financial decisions.")
model = load(MODEL_PATH)
requests = 0

class RiskRequest(BaseModel):
    transaction_amount: float = Field(ge=0, le=1_000_000)
    merchant_category: Literal["electronics", "grocery", "travel", "fashion", "gaming"]
    country_risk_score: float = Field(ge=0, le=1)
    account_age_days: int = Field(ge=0, le=100_000)
    transaction_count_24h: int = Field(ge=0, le=10_000)
    failed_auth_attempts_24h: int = Field(ge=0, le=10_000)
    is_new_device: bool
    hour_utc: int = Field(ge=0, le=23)

class RiskResponse(BaseModel):
    risk_score: float
    risk_band: str
    model_version: str
    decision_threshold: float

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/v1/risk-score", response_model=RiskResponse)
def risk_score(payload: RiskRequest):
    global requests
    requests += 1
    score = float(model.predict_proba([payload.model_dump()])[0, 1])
    band = "high" if score >= THRESHOLD else "medium" if score >= 0.35 else "low"
    return RiskResponse(risk_score=round(score, 6), risk_band=band, model_version=MODEL_VERSION, decision_threshold=THRESHOLD)

@app.get("/metrics")
def metrics():
    return f"# HELP risk_score_requests_total Total scoring requests\n# TYPE risk_score_requests_total counter\nrisk_score_requests_total {requests}\n"
