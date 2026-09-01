import joblib
from pathlib import Path
from sklearn.dummy import DummyClassifier
from generate_data import main as generate


def main() -> None:
    generate()
    model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/risk_model.joblib")


if __name__ == "__main__":
    main()
