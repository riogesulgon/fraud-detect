"""Create a deterministic, dependency-light drift report for synthetic traffic."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import FEATURES, generate_data

def main() -> None:
    baseline, _ = generate_data(1000, 42)
    recent, _ = generate_data(200, 99)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "features": {}}
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
