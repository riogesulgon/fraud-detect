from __future__ import annotations

import hashlib
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA = Path("data/raw/creditcard_fraud_synthetic.csv")
REPORT = Path("reports/calibration.json")
# Review-queue cost model: an undetected fraud costs 20x a wasted manual review.
COST_FN = 20.0
COST_FP = 1.0


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in itertools.pairwise(edges):
        mask = (p >= low) & (p < high if high < 1 else p <= high)
        if mask.any():
            error += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(error)


def choose_recall_threshold(y: np.ndarray, p: np.ndarray, target_recall: float = 0.8) -> float:
    precision, recall, thresholds = precision_recall_curve(y, p)
    eligible = np.flatnonzero(recall[:-1] >= target_recall)
    if len(eligible):
        return float(thresholds[eligible[int(np.argmax(precision[:-1][eligible]))]])
    return float(
        thresholds[
            int(
                np.argmax(
                    2
                    * precision[:-1]
                    * recall[:-1]
                    / np.maximum(precision[:-1] + recall[:-1], 1e-12)
                )
            )
        ]
    )


def expected_cost(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> dict[str, float]:
    pred = p >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    alerts = tp + fp
    return {
        "expected_cost": float(cost_fn * fn + cost_fp * fp),
        "alerts": float(alerts),
        "alerts_per_1000": float(1000 * alerts / max(len(y), 1)),
        "fraud_captured": float(tp / max(tp + fn, 1)),
    }


def choose_cost_threshold(
    y: np.ndarray, p: np.ndarray, cost_fn: float = COST_FN, cost_fp: float = COST_FP
) -> float:
    precision, recall, thresholds = precision_recall_curve(y, p)
    tp = recall[:-1] * y.sum()
    fp = tp * (1 - precision[:-1]) / np.maximum(precision[:-1], 1e-12)
    fn = y.sum() - tp
    costs = cost_fn * fn + cost_fp * fp
    return float(thresholds[int(np.argmin(costs))])


def operating_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": threshold,
    }


def main() -> None:
    frame = pd.read_csv(DATA).sort_values("timestamp_seconds").reset_index(drop=True)
    dataset_sha256 = hashlib.sha256(DATA.read_bytes()).hexdigest()
    frame = frame.iloc[:: max(1, len(frame) // 150_000)].copy()
    y = frame.pop("is_fraud").astype(int).to_numpy()
    x = frame.drop(columns=["transaction_id", "customer_id"], errors="ignore")
    x = (
        pd.get_dummies(x, columns=x.select_dtypes(include=["object"]).columns, dtype=float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
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
    calibrator = LogisticRegression(solver="lbfgs").fit(raw_cal.reshape(-1, 1), y_cal)
    raw_test = base.predict_proba(x_test)[:, 1]
    calibrated_cal = calibrator.predict_proba(raw_cal.reshape(-1, 1))[:, 1]
    calibrated_test = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    recall_threshold = choose_recall_threshold(y_cal, calibrated_cal)
    cost_threshold = choose_cost_threshold(y_cal, calibrated_cal)
    recall_cost = expected_cost(y_cal, calibrated_cal, recall_threshold)["expected_cost"]
    cost_cost = expected_cost(y_cal, calibrated_cal, cost_threshold)["expected_cost"]
    use_cost_based = cost_cost < recall_cost
    threshold = cost_threshold if use_cost_based else recall_threshold
    objective = (
        "minimize expected cost (FN=20x FP), chosen on validation"
        if use_cost_based
        else "maximize validation precision subject to recall >= 0.80 (cost model did not improve)"
    )
    result = {
        "dataset": "kaggle",
        "model_version": "kaggle-full-calibrated",
        "dataset_sha256": dataset_sha256,
        "training_timestamp": datetime.now(UTC).isoformat(),
        "split": "chronological 60/20/20",
        "rows": n,
        "calibration_method": "platt_sigmoid",
        "cost_model": {"false_negative": COST_FN, "false_positive": COST_FP},
        "operating_objective": objective,
        "operating_threshold": threshold,
        "threshold_candidates": {
            "recall_constrained": {
                "threshold": recall_threshold,
                "validation": {
                    **operating_metrics(y_cal, calibrated_cal, recall_threshold),
                    **expected_cost(y_cal, calibrated_cal, recall_threshold),
                },
                "test": {
                    **operating_metrics(y_test, calibrated_test, recall_threshold),
                    **expected_cost(y_test, calibrated_test, recall_threshold),
                },
            },
            "cost_based": {
                "threshold": cost_threshold,
                "validation": {
                    **operating_metrics(y_cal, calibrated_cal, cost_threshold),
                    **expected_cost(y_cal, calibrated_cal, cost_threshold),
                },
                "test": {
                    **operating_metrics(y_test, calibrated_test, cost_threshold),
                    **expected_cost(y_test, calibrated_test, cost_threshold),
                },
            },
        },
        "validation_operating_metrics": {
            **operating_metrics(y_cal, calibrated_cal, threshold),
            **expected_cost(y_cal, calibrated_cal, threshold),
        },
        "test_operating_metrics": {
            **operating_metrics(y_test, calibrated_test, threshold),
            **expected_cost(y_test, calibrated_test, threshold),
        },
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
        {
            "base_model": base,
            "calibrator": calibrator,
            "threshold": threshold,
            "feature_columns": list(x.columns),
        },
        "models/full-feature-calibrated.joblib",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
