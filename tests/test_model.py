from pathlib import Path

import numpy as np
import pytest

from src.contracts import validate_features
from src.model import generate_data, load, train


def test_generate_and_contract() -> None:
    rows, labels = generate_data(20, 42)
    assert len(rows) == len(labels) == 20
    assert validate_features(rows).shape == (20, 8)


def test_contract_rejects_missing_column() -> None:
    with pytest.raises(ValueError, match="missing feature"):
        validate_features([{"transaction_amount": 1}])


def test_contract_rejects_invalid_ranges() -> None:
    rows, _ = generate_data(1, 42)
    rows[0]["hour_utc"] = 24
    with pytest.raises(ValueError, match="hour_utc"):
        validate_features(rows)


def test_train_and_load(tmp_path: Path) -> None:
    rows, labels = generate_data(50, 42)
    info = train(rows, labels, tmp_path / "model.joblib")
    assert info.training_rows == 50
    model = load(tmp_path / "model.joblib")
    assert model.predict(rows[:2]).shape == (2,)


def test_load_builds_missing_artifact(tmp_path: Path) -> None:
    model = load(tmp_path / "missing.joblib")
    assert np.asarray(model.predict([generate_data(1, 42)[0][0]])).shape == (1,)
