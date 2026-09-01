from __future__ import annotations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.model import generate_data, train
from src.kaggle_adapter import load_kaggle
from src.contracts import validate_features


def main() -> None:
    kaggle_path = Path("data/raw/creditcard_fraud_synthetic.csv")
    if kaggle_path.exists():
        rows, labels = load_kaggle(kaggle_path, sample_rows=100_000, seed=42)
    else:
        rows, labels = generate_data(1000, 42)
    validate_features(rows)
    train_rows, test_rows, train_labels, test_labels = train_test_split(rows, labels, test_size=0.2, random_state=42, stratify=labels)
    info = train(train_rows, train_labels, Path("models/risk-model.joblib"), dataset_name="kaggle" if kaggle_path.exists() else "synthetic")
    model = joblib.load("models/risk-model.joblib")
    predicted = model.predict(test_rows)
    probabilities = model.predict_proba(test_rows)[:, 1]
    metrics = {"precision": precision_score(test_labels, predicted, zero_division=0), "recall": recall_score(test_labels, predicted, zero_division=0), "f1": f1_score(test_labels, predicted, zero_division=0), "roc_auc": roc_auc_score(test_labels, probabilities)}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/evaluation.json").write_text(json.dumps({"dataset": "kaggle" if kaggle_path.exists() else "synthetic", "model_version": info.model_version, "metrics": metrics}, indent=2))
    try:
        import mlflow
        with mlflow.start_run(run_name="kaggle-risk-baseline" if kaggle_path.exists() else "synthetic-risk-baseline"):
            mlflow.log_params({"seed": 42, "rows": len(train_rows), "threshold": info.decision_threshold})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact("reports/evaluation.json")
    except ImportError:
        pass

if __name__ == "__main__":
    main()
