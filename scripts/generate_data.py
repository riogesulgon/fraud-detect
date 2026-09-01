from pathlib import Path
import csv
import random
random.seed(42)
out=Path("data/sample.csv")
out.parent.mkdir(exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["transaction_amount", "merchant_category", "country_risk_score", "account_age_days", "transaction_count_24h", "failed_auth_attempts_24h", "is_new_device", "hour_utc", "label_is_high_risk"])
    for _ in range(100):
        amount = round(random.uniform(1, 1000), 2)
        w.writerow([amount, "retail", round(random.random(), 2), random.randint(1, 2000), random.randint(0, 20), random.randint(0, 4), random.choice([True, False]), random.randint(0, 23), int(amount > 700)])

def main() -> None:
    return None
