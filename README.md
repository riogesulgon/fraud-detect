# MLOps Risk Platform

A deterministic synthetic transaction-risk scoring demonstrator. **It is not suitable for live financial decisions.**

## Five-minute quickstart

```bash
make install
make train
make test
make run
curl -X POST http://localhost:8000/v1/risk-score -H 'content-type: application/json' -d '{"transaction_amount":145.5,"merchant_category":"electronics","country_risk_score":0.42,"account_age_days":730,"transaction_count_24h":3,"failed_auth_attempts_24h":0,"is_new_device":false,"hour_utc":14}'
```

The baseline uses seeded synthetic data, schema-validated FastAPI inputs, and a scikit-learn classifier. `/healthz`, `/metrics`, and `/v1/risk-score` are included in milestone 1. Model artifacts are generated locally and ignored by git.
