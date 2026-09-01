# MLOps Risk Platform

A synthetic transaction-risk scoring demonstrator. It is not suitable for live financial decisions.

## Quick start

```bash
make install
make data
make test
make run
curl -X POST http://localhost:8000/v1/risk-score -H 'content-type: application/json' -d '{"transaction_amount":145.5,"merchant_category":"electronics","country_risk_score":0.42,"account_age_days":730,"transaction_count_24h":3,"failed_auth_attempts_24h":0,"is_new_device":false,"hour_utc":14}'
```

See `SPEC.md` for the complete delivery plan.

## Model tuning benchmark

Run `python scripts/benchmark_full.py` to evaluate an imbalance-weighted HistGradientBoosting model on the full Kaggle feature set using a chronological 70/15/15 split. The benchmark is separate from the stable eight-field API adapter until its richer inference contract is explicitly versioned.
