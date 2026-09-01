"""Create a deterministic, dependency-light drift report for synthetic traffic."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import FEATURES, generate_data
from src.kaggle_adapter import load_kaggle

def main() -> None:
    kaggle_path = Path("data/raw/creditcard_fraud_synthetic.csv")
    if kaggle_path.exists():
        mapped, _ = load_kaggle(kaggle_path, sample_rows=100_000, seed=42)
        baseline, recent = mapped[:1_000], mapped[-200:]
        source = "kaggle"
    else:
        baseline, _ = generate_data(1000, 42)
        recent, _ = generate_data(200, 99)
        source = "synthetic"
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "source": source, "features": {}}
    for feature in FEATURES:
        if isinstance(baseline[0][feature], bool):
            a = sum(bool(row[feature]) for row in baseline) / len(baseline)
            b = sum(bool(row[feature]) for row in recent) / len(recent)
        elif isinstance(baseline[0][feature], (int, float)):
            a = sum(float(row[feature]) for row in baseline) / len(baseline)
            b = sum(float(row[feature]) for row in recent) / len(recent)
        else:
            a = b = None
        report["features"][feature] = {"baseline_mean": a, "recent_mean": b}
    out = Path("reports/drift.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
