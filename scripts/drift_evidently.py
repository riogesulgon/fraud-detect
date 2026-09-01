"""Generate an Evidently data-drift report from the Kaggle dataset.

Compares the earliest 60,000 transactions (baseline) with the latest 20,000
transactions (recent batch) after sorting chronologically. The target column is
excluded because drift monitoring observes features, not labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

DATA = Path("data/raw/creditcard_fraud_synthetic.csv")
OUT_DIR = Path("reports")
BASELINE_ROWS = 60_000
RECENT_ROWS = 20_000


def main() -> None:
    frame = pd.read_csv(DATA).sort_values("timestamp_seconds").reset_index(drop=True)
    baseline = frame.iloc[:BASELINE_ROWS].drop(columns=["is_fraud"])
    recent = frame.iloc[-RECENT_ROWS:].drop(columns=["is_fraud"])
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=baseline, current_data=recent)
    OUT_DIR.mkdir(exist_ok=True)
    snapshot.save_html(str(OUT_DIR / "drift_evidently.html"))
    snapshot.save_json(str(OUT_DIR / "drift_evidently.json"))
    print(f"saved {OUT_DIR / 'drift_evidently.html'} and {OUT_DIR / 'drift_evidently.json'}")


if __name__ == "__main__":
    main()
