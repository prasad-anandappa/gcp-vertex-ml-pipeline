"""Generates the bundled sample dataset (data/customers.csv).

Synthetic customer-churn data. The generating process is a logistic model
with a clear signal — engagement (visits) and loyalty (tenure) protect
against churn, lower income slightly raises it, age is deliberately
irrelevant — plus noise, so the problem is learnable but not trivial.
Fixed seed: regenerating produces the identical file.

A RandomForest trained on this data lands around 0.82 held-out accuracy
(AUC ~0.90) against a ~0.59 majority-class baseline: comfortably clear of
the pipeline's 0.75 metric gate, while a stricter threshold (say 0.85)
still fails — the gate is a real decision, not a formality.

    python data/generate_sample.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["age", "income", "visits_per_month", "tenure_months"]
TARGET = "churned"


def main() -> None:
    rng = np.random.default_rng(42)
    n = 1200

    age = rng.integers(18, 71, n)
    income = np.clip(rng.normal(65_000, 25_000, n), 18_000, 180_000).round(0)
    visits = np.clip(np.round(rng.normal(9, 2.2, n)), 0, 20).astype(int)
    tenure = rng.integers(1, 121, n)

    z = (
        10.1
        - 0.85 * visits
        - 0.055 * tenure
        - 0.000006 * (income - 60_000)
        + rng.normal(0, 0.30, n)
    )
    p_churn = 1.0 / (1.0 + np.exp(-z))
    churned = (rng.random(n) < p_churn).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "visits_per_month": visits,
            "tenure_months": tenure,
            TARGET: churned,
        }
    )

    out = Path(__file__).parent / "customers.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows, churn rate {df[TARGET].mean():.2%}")


if __name__ == "__main__":
    main()
