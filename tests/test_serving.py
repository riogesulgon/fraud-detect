import logging

from fastapi.testclient import TestClient

from app import main
from app.main import app, logger

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


def test_request_id_header_and_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MINUTE", 1)
    monkeypatch.setattr(main, "_rate_buckets", {})
    first = client.post("/v2/risk-score", json=PAYLOAD)
    assert first.status_code == 200
    assert first.headers["X-Request-ID"]
    custom = client.post("/v2/risk-score", json=PAYLOAD, headers={"X-Request-ID": "abc-123"})
    assert custom.headers["X-Request-ID"] == "abc-123"
    limited = client.post("/v2/risk-score", json=PAYLOAD)
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "60"


def test_structured_log_excludes_payload_and_healthz() -> None:
    records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    capture = Capture()
    logger.addHandler(capture)
    try:
        response = client.post("/v2/risk-score", json=PAYLOAD)
        assert response.status_code == 200
        assert client.get("/healthz").status_code == 200
    finally:
        logger.removeHandler(capture)
    request_events = [r for r in records if '"event": "request"' in r]
    assert len(request_events) == 1  # /healthz logging is suppressed
    joined = "\n".join(records)
    assert "145.5" not in joined  # request bodies never reach the logs
    assert "electronics" not in joined
    assert "customer_age" not in joined
