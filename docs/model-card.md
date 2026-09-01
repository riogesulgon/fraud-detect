# Model card

This project is a synthetic demonstrator and must not be used for live financial decisions.

## Intended use

Explore reproducible model training, probability calibration, drift monitoring, and safe API delivery for transaction-risk scoring. The served models return a risk score and band for triage-style review queues; they never return an automated approve/decline outcome.

## Data and model

Training uses the public [Synthetic Credit Card Fraud (Interpretable 1M)](https://www.kaggle.com/datasets/harshjain123/synthetic-credit-card-fraud-interpretable) dataset (CC0-1.0). It contains 1M transactions with 27 named features and approximately 1.95% fraud.

Two model paths are provided:

- **v1 adapter model** (`models/risk-model.joblib`): trained through an explicit adapter that maps Kaggle columns onto the stable eight-field API contract, with a validation-tuned threshold.
- **v2 calibrated model** (`models/full-feature-calibrated.joblib`): an imbalance-weighted scikit-learn HistGradientBoostingClassifier using all non-identifier features, calibrated with Platt sigmoid on a held-out chronological set.

## Evaluation and operating point

Calibrated full-feature model on a chronological 60/20/20 split:

- ROC-AUC: 0.8314
- PR-AUC: 0.1857 (fraud prevalence ~0.0195, so roughly 9.6x random ranking)
- Brier score: 0.0171
- Expected calibration error: 0.0013
- Calibration method: Platt sigmoid
- Operating threshold: 0.0772 (validation objective: minimize expected cost, false negative weighted 20x a false alert; a high-recall alternative at threshold 0.0110 is preserved in `reports/calibration.json`)
- At the operating point: recall 40.6%, precision 15.27%, F1 22.2% (~6.5 alerts per confirmed fraud)
- High-recall alternative (threshold 0.0110): recall 81.3%, precision 4.56% (~22 alerts per confirmed fraud)

The default point prioritizes reviewer efficiency under the documented 20:1 cost assumption; teams that prize recall over review capacity can raise it. Both points come from the same calibrated probabilities, so switching is a configuration change, not a retrain. This is a review-queue operating point, not an approve/decline decision. Threshold selection trades false positives for missed frauds; the numbers above are the measured cost of that trade-off. The false-positive/false-negative balance should be re-derived from real review capacity and fraud costs before any production use.

## Limitations

- Calibrated on synthetic Kaggle data only; single chronological split, no cross-validation.
- Calibration improves probability reliability but not ranking quality.
- At the default operating point (recall 40.6%), roughly 59% of fraud goes undetected; at the high-recall alternative (81.3%), roughly 19% does. Neither point is safe for automation.
- One-hot encoding and sampling choices are tuned for this dataset; schema changes require retraining.
- Metrics are not evidence of production accuracy. A real system would require governance, bias analysis, human review, calibration monitoring, privacy controls, and extensive pre-production validation.