from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

def test_health():
    response = client.get("/healthz")
    assert response.status_code == 200 and response.json()["model_loaded"]

def test_score():
    response = client.post("/v1/risk-score", json={"transaction_amount":145.5,"merchant_category":"electronics","country_risk_score":.42,"account_age_days":730,"transaction_count_24h":3,"failed_auth_attempts_24h":0,"is_new_device":False,"hour_utc":14})
    body = response.json()
    assert response.status_code == 200 and 0 <= body["risk_score"] <= 1 and body["risk_band"] in {"low","medium","high"}

def test_invalid_payload():
    assert client.post("/v1/risk-score", json={}).status_code == 422
