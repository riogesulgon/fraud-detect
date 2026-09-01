from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA = Path("data/raw/creditcard_fraud_synthetic.csv")
REPORT = Path("reports/full_feature_benchmark.json")
TARGET = "is_fraud"
DROP = ["transaction_id", "customer_id", TARGET]


def operating_threshold(y: np.ndarray, p: np.ndarray, target_recall: float = 0.8) -> float:
    precision, recall, thresholds = precision_recall_curve(y, p)
    eligible = np.flatnonzero(recall[:-1] >= target_recall)
    if len(eligible):
        best = eligible[int(np.argmax(precision[:-1][eligible]))]
    else:
        best = int(
            np.argmax(
                2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
            )
        )
    return float(thresholds[best])


def scores(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "threshold": threshold,
    }


def main() -> None:
    frame = pd.read_csv(DATA).sort_values("timestamp_seconds").reset_index(drop=True)
    if len(frame) > 150_000:
        frame = frame.iloc[:: max(1, len(frame) // 150_000)].copy()
    y = frame.pop(TARGET).astype(int).to_numpy()
    x = frame.drop(columns=DROP, errors="ignore")
    x = pd.get_dummies(x, columns=x.select_dtypes(include=["object"]).columns, dtype=float)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    n = len(x)
    train_end, val_end = int(n * 0.7), int(n * 0.85)
    x_train, x_val, x_test = x.iloc[:train_end], x.iloc[train_end:val_end], x.iloc[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    weights = np.where(y_train == 1, (y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=42
    )
    model.fit(x_train, y_train, sample_weight=weights)
    threshold = operating_threshold(y_val, model.predict_proba(x_val)[:, 1])
    result = {
        "dataset": "kaggle",
        "rows": n,
        "features": int(x.shape[1]),
        "split": "chronological 70/15/15",
        "fraud_rate": float(y.mean()),
        "selected_model": scores(y_test, model.predict_proba(x_test)[:, 1], threshold),
        "selected_model_at_065": scores(y_test, model.predict_proba(x_test)[:, 1], 0.65),
        "target_recall": 0.8,
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
