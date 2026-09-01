from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PAYLOAD = {
    "transaction_id": 1,
    "customer_id": 99,
    "timestamp_seconds": 1700000000,
    "hour_of_day": 14,
    "day_of_week": 2,
    "is_weekend": 0,
    "is_night": 0,
    "amount": 145.5,
    "avg_amount_30d": 100.0,
    "amount_to_avg_ratio": 1.455,
    "customer_age": 35,
    "customer_tenure_days": 730,
    "account_balance": 5000.0,
    "income_band": 3,
    "merchant_category": "electronics",
    "merchant_id": 123,
    "transaction_type": "online",
    "card_present": 0,
    "device_type": "web_desktop",
    "num_transactions_last_1h": 1,
    "num_transactions_last_24h": 3,
    "minutes_since_last_transaction": 120.0,
    "distance_from_home_km": 4.0,
    "distance_from_last_transaction_km": 2.0,
    "is_foreign_transaction": 0,
    "ip_address_risk_score": 0.42,
    "failed_pin_attempts_24h": 0,
}


def test_full_feature_score_contract() -> None:
    response = client.post("/v2/risk-score", json=PAYLOAD)
    assert response.status_code == 200
    assert set(response.json()) == {
        "risk_score",
        "risk_band",
        "model_version",
        "decision_threshold",
    }
    assert response.json()["model_version"] == "kaggle-full-calibrated"


def test_full_feature_invalid_range() -> None:
    assert client.post("/v2/risk-score", json=PAYLOAD | {"hour_of_day": 24}).status_code == 422
