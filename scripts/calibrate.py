from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DATA = Path("data/raw/creditcard_fraud_synthetic.csv")
REPORT = Path("reports/calibration.json")


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p >= low) & (p < high if high < 1 else p <= high)
        if mask.any():
            error += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(error)


def main() -> None:
    frame = pd.read_csv(DATA).sort_values("timestamp_seconds").reset_index(drop=True)
    frame = frame.iloc[:: max(1, len(frame) // 150_000)].copy()
    y = frame.pop("is_fraud").astype(int).to_numpy()
    x = frame.drop(columns=["transaction_id", "customer_id"], errors="ignore")
    x = pd.get_dummies(x, columns=x.select_dtypes(include=["object"]).columns, dtype=float)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    n = len(x)
    train_end, calibration_end = int(n * 0.6), int(n * 0.8)
    x_train, x_cal, x_test = (
        x.iloc[:train_end],
        x.iloc[train_end:calibration_end],
        x.iloc[calibration_end:],
    )
    y_train, y_cal, y_test = y[:train_end], y[train_end:calibration_end], y[calibration_end:]
    weights = np.where(y_train == 1, (y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)
    base = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=42
    )
    base.fit(x_train, y_train, sample_weight=weights)
    raw_cal = base.predict_proba(x_cal)[:, 1]
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(raw_cal.reshape(-1, 1), y_cal)
    raw_test = base.predict_proba(x_test)[:, 1]
    calibrated_test = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    result = {
        "dataset": "kaggle",
        "split": "chronological 60/20/20",
        "rows": n,
        "calibration_method": "platt_sigmoid",
        "raw": {
            "brier": float(brier_score_loss(y_test, raw_test)),
            "ece": expected_calibration_error(y_test, raw_test),
            "roc_auc": float(roc_auc_score(y_test, raw_test)),
            "pr_auc": float(average_precision_score(y_test, raw_test)),
        },
        "calibrated": {
            "brier": float(brier_score_loss(y_test, calibrated_test)),
            "ece": expected_calibration_error(y_test, calibrated_test),
            "roc_auc": float(roc_auc_score(y_test, calibrated_test)),
            "pr_auc": float(average_precision_score(y_test, calibrated_test)),
        },
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2))
    joblib.dump(
        {"base_model": base, "calibrator": calibrator}, "models/full-feature-calibrated.joblib"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
