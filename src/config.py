"""Central configuration: paths, seed, band definitions, anomaly parameters."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"
OUTPUTS = ROOT / "outputs"  # runtime state (review decisions); git-ignored

RAW_FILES = {
    "customers": "customers.csv",
    "loans": "loan_accounts.csv",
    "bureau": "bureau_credit.csv",
    "repayment": "repayment_history.csv",
    "collections": "collections_events.csv",
}

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Observation point: the end of repayment month 12 after origination.
# Every repayment feature uses payment_month 1..12 only (the data contains nothing else).
OBSERVATION_MONTH = 12

INCOME_BINS = [0, 30_000, 60_000, 90_000, 120_000, float("inf")]
INCOME_LABELS = ["<30k", "30-60k", "60-90k", "90-120k", "120k+"]
AGE_BINS = [0, 24, 34, 44, 54, 200]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55+"]

# Risk bands are cut at quantiles of the *training* score distribution (fixed once).
RISK_BAND_QUANTILES = [0.0, 0.50, 0.80, 0.95, 1.0]
RISK_BAND_LABELS = ["Low", "Medium", "High", "Very high"]

# Anomaly detection parameters (design parameters, documented in docs/data_quality_log.md
# and the solution overview; not tuned against the test-suite).
ANOMALY_PEER_PERCENTILE = 0.99      # a value above the peer group's 99th percentile is "extreme"
ANOMALY_MIN_SIGNALS = 2             # flag when at least this many independent signals fire
ANOMALY_OWN_DPD_JUMP_DAYS = 25      # last-3-month mean DPD minus first-9-month mean DPD
ANOMALY_OWN_DPD_LAST3_MIN = 30      # ... and the last-3-month mean DPD must be at least this
ANOMALY_OWN_PAID_DROP = 0.5         # drop in paid ratio (first 9 months -> last 3 months)
ANOMALY_STRONG_SINGLE_SIGNAL = True  # own-history deterioration alone is enough to flag

REVIEW_DB = Path(os.environ.get("REVIEW_DB", OUTPUTS / "review_decisions.db"))
