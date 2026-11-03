"""Independent pandas oracle for the Ask suite. Written separately from src/ (reads the raw CSVs directly)
so a bug in src/features.py or src/ask.py cannot hide behind itself."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    c, l, r = raw["customers"].copy(), raw["loans"].copy(), raw["repayment"]
    c["gender"] = c["gender"].replace({"M": "Male", "F": "Female"})
    c["income_band"] = pd.cut(c["annual_income"], [0, 30000, 60000, 90000, 120000, np.inf], labels=["<30k", "30-60k", "60-90k", "90-120k", "120k+"], right=False).astype(str)
    c["age_band"] = pd.cut(c["age"], [0, 24, 34, 44, 54, 200], labels=["<25", "25-34", "35-44", "45-54", "55+"], right=True).astype(str)
    df = l.merge(c, on="customer_id", how="left")
    df["origination_year"] = pd.to_datetime(df["origination_date"]).dt.year
    miss = r[r["payment_status"] == "Missed"].groupby("account_id").size()
    df["n_missed"] = df["account_id"].map(miss).fillna(0)
    return df


def by(df, dim, value="target_default", agg="mean", min_n=30):
    g = df.groupby(dim)
    out = pd.DataFrame({"accounts": g.size(), "value": g[value].agg(agg) if agg != "size" else g.size()})
    return out[out["accounts"] >= min_n]
