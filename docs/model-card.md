# Model card

This project is a synthetic demonstrator and must not be used for live financial decisions.

## Intended use

Explore reproducible model training and safe API delivery for transaction-risk scoring.

## Data and model

The training data is deterministic synthetic data. The baseline uses a scikit-learn gradient boosting classifier and an explicit 0.65 operating threshold.

## Limitations

Synthetic labels do not represent real fraud. Metrics are not evidence of production accuracy. A real system would require governance, bias analysis, human review, calibration, privacy controls, and extensive pre-production validation.
