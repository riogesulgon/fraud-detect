import csv
from pathlib import Path
from src.model import FEATURES, generate_data
rows, labels = generate_data(500)
Path("data").mkdir(exist_ok=True)
with open("data/synthetic.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FEATURES + ["label_is_high_risk"])
    writer.writeheader()
    for row, label in zip(rows, labels):
        writer.writerow({**row, "label_is_high_risk": int(label)})
