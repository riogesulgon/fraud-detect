# Data

The committed `sample.csv` is deterministic synthetic fixture data used for local tests.

The recommended larger dataset is [Synthetic Credit Card Fraud (Interpretable 1M)](https://www.kaggle.com/datasets/harshjain123/synthetic-credit-card-fraud-interpretable), by `harshjain123`. It contains 1M synthetic transactions, 27 named features, and approximately 1.9% fraud. Download it locally with:

```bash
make kaggle-data
```

The Kaggle page reports the dataset license as **CC0-1.0**. The downloaded payload is still intentionally not committed. Keep `kaggle-token.txt` outside the repository where possible; it is ignored as a safety net.

## Adapter mapping

The Kaggle loader maps `amount`, `merchant_category`, `ip_address_risk_score`, `customer_tenure_days`, `num_transactions_last_24h`, `failed_pin_attempts_24h`, and `hour_of_day` into the stable v1 API feature contract. `is_new_device` is derived from tenure under 30 days. Identifiers are excluded.
