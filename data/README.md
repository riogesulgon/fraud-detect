# Data

The committed `sample.csv` is deterministic synthetic fixture data used for local tests.

The recommended larger dataset is [Synthetic Credit Card Fraud (Interpretable 1M)](https://www.kaggle.com/datasets/harshjain123/synthetic-credit-card-fraud-interpretable), by `harshjain123`. It contains 1M synthetic transactions, 27 named features, and approximately 1.9% fraud. Download it locally with:

```bash
make kaggle-data
```

The Kaggle payload is intentionally not committed. Review the dataset's current Kaggle license and terms before redistribution or publication. No real customer or payment-rail data may be added to this repository.
