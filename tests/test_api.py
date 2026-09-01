from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_score_contract() -> None:
    response = client.post(
        "/v1/risk-score",
        json={
            "transaction_amount": 145.5,
            "merchant_category": "electronics",
            "country_risk_score": 0.42,
            "account_age_days": 730,
            "transaction_count_24h": 3,
            "failed_auth_attempts_24h": 0,
            "is_new_device": False,
            "hour_utc": 14,
        },
    )
    assert response.status_code == 200
    assert set(response.json()) == {
        "risk_score",
        "risk_band",
        "model_version",
        "decision_threshold",
    }


def test_invalid_payload() -> None:
    assert client.post("/v1/risk-score", json={}).status_code == 422
