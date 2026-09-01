from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.contracts import validate_features
from src.kaggle_adapter import load_kaggle
from src.model import generate_data, train


def metrics_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (probabilities >= threshold).astype(int)
    return {
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "threshold": float(threshold),
    }


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(round(float(thresholds[int(np.argmax(f1))]), 6))


def main() -> None:
    kaggle_path = Path("data/raw/creditcard_fraud_synthetic.csv")
    if kaggle_path.exists():
        rows, labels_series = load_kaggle(kaggle_path, sample_rows=100_000, seed=42)
        dataset = "kaggle"
    else:
        rows, labels = generate_data(1000, 42)
        labels_series = labels
        dataset = "synthetic"
    validate_features(rows)
    labels = np.asarray(labels_series, dtype=int)
    train_rows, holdout_rows, train_labels, holdout_labels = train_test_split(rows, labels, test_size=0.3, random_state=42, stratify=labels)
    validation_rows, test_rows, validation_labels, test_labels = train_test_split(holdout_rows, holdout_labels, test_size=2 / 3, random_state=42, stratify=holdout_labels)

    model_path = Path("models/risk-model.joblib")
    train(train_rows, train_labels, Path("models/benchmark.joblib"), dataset_name=dataset)
    gradient = joblib.load("models/benchmark.joblib")
    validation_prob = gradient.predict_proba(validation_rows)[:, 1]
    threshold = choose_threshold(validation_labels, validation_prob)
    final_info = train(train_rows + list(validation_rows), np.concatenate([train_labels, validation_labels]), model_path, dataset_name=dataset)
    final_model = joblib.load(model_path)
    test_prob = final_model.predict_proba(test_rows)[:, 1]

    vectorizer = DictVectorizer(sparse=False)
    logistic = Pipeline([("vectorizer", vectorizer), ("classifier", LogisticRegression(max_iter=300, class_weight="balanced", random_state=42))])
    logistic.fit(train_rows, train_labels)
    logistic_prob = logistic.predict_proba(test_rows)[:, 1]
    results = {
        "dataset": dataset,
        "model_version": final_info.model_version,
        "decision_threshold": threshold,
        "selected_model": metrics_at_threshold(test_labels, test_prob, threshold),
        "selected_model_at_065": metrics_at_threshold(test_labels, test_prob, 0.65),
        "logistic_regression": metrics_at_threshold(test_labels, logistic_prob, threshold),
        "benchmark_rows": len(rows),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/evaluation.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
