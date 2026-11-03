"""Ingestion: load the 5 raw CSVs, resolve data-quality issues, record every decision.

Every issue is detected on the RAW data (n_raw) and re-checked on the CLEANED data
(n_unresolved); both numbers are written to reports/dq_log.json and rendered into
docs/data_quality_log.md. Nothing is dropped silently: row counts are asserted unchanged.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config

BUREAU_CUTOFF = pd.Timestamp("2023-12-31")  # last loan origination in the data is 2023-12-30
UNEMPLOYED_INCOME_FLOOR = 12_000


@dataclass
class DQIssue:
    id: str
    table: str
    issue: str
    detector: str
    n_raw: int
    decision: str
    rationale: str
    handling: str          # fixed | flagged | retained
    n_unresolved: int = 0  # same detector re-run on the cleaned output (0 for "fixed")
    output_columns: str = ""


def load_raw(raw_dir=config.RAW) -> dict[str, pd.DataFrame]:
    """Read the five CSVs exactly as provided (dates parsed, nothing else changed)."""
    t = {k: pd.read_csv(raw_dir / f) for k, f in config.RAW_FILES.items()}
    t["customers"]["customer_since"] = pd.to_datetime(t["customers"]["customer_since"])
    t["loans"]["origination_date"] = pd.to_datetime(t["loans"]["origination_date"])
    t["bureau"]["date_opened"] = pd.to_datetime(t["bureau"]["date_opened"])
    t["bureau"]["date_closed"] = pd.to_datetime(t["bureau"]["date_closed"])
    t["collections"]["event_date"] = pd.to_datetime(t["collections"]["event_date"])
    return t


def clean_tables(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[DQIssue]]:
    c = raw["customers"].copy()
    l = raw["loans"].copy()
    b = raw["bureau"].copy()
    r = raw["repayment"].copy()
    k = raw["collections"].copy()
    issues: list[DQIssue] = []
    rows_before = {n: len(df) for n, df in raw.items()}

    # DQ01 gender labels -----------------------------------------------------------
    n = int(c["gender"].isin(["M", "F"]).sum())
    c["gender"] = c["gender"].replace({"M": "Male", "F": "Female"})
    issues.append(DQIssue(
        "DQ01", "customers", "Inconsistent gender labels ('M'/'F' alongside 'Male'/'Female')",
        "gender in {'M','F'}", n,
        "Map M->Male, F->Female.", "Same category, two spellings; leaving both would split every gender cut in two.",
        "fixed", int(c["gender"].isin(["M", "F"]).sum()), "gender"))

    # DQ02 missing years_employed ----------------------------------------------------
    miss = c["years_employed"].isna()
    n = int(miss.sum())
    c["years_employed_missing"] = miss.astype(int)
    med = c.groupby("employment_type")["years_employed"].transform("median")
    c["years_employed"] = c["years_employed"].fillna(med)
    issues.append(DQIssue(
        "DQ02", "customers", "Missing years_employed", "years_employed is null", n,
        "Impute the median within employment_type; add years_employed_missing flag.",
        "Tenure differs strongly by employment type, so a global median would be misleading. The target is never used.",
        "fixed", int(c["years_employed"].isna().sum()), "years_employed, years_employed_missing"))

    # DQ03 missing num_dependents ---------------------------------------------------
    miss = c["num_dependents"].isna()
    n = int(miss.sum())
    c["num_dependents_missing"] = miss.astype(int)
    c["num_dependents"] = c["num_dependents"].fillna(c["num_dependents"].median())
    issues.append(DQIssue(
        "DQ03", "customers", "Missing num_dependents", "num_dependents is null", n,
        "Impute the overall median; add num_dependents_missing flag.",
        "Small integer count with no strong group structure.", "fixed",
        int(c["num_dependents"].isna().sum()), "num_dependents, num_dependents_missing"))

    # DQ04 missing years_at_residence -----------------------------------------------
    miss = c["years_at_residence"].isna()
    n = int(miss.sum())
    c["years_at_residence_missing"] = miss.astype(int)
    med = c.groupby("housing_type")["years_at_residence"].transform("median")
    c["years_at_residence"] = c["years_at_residence"].fillna(med)
    issues.append(DQIssue(
        "DQ04", "customers", "Missing years_at_residence", "years_at_residence is null", n,
        "Impute the median within housing_type; add years_at_residence_missing flag.",
        "Residence tenure depends on housing type (owners stay longer).", "fixed",
        int(c["years_at_residence"].isna().sum()), "years_at_residence, years_at_residence_missing"))

    # DQ05 missing interest rate ------------------------------------------------------
    miss = l["interest_rate_pct"].isna()
    n = int(miss.sum())
    l["interest_rate_missing"] = miss.astype(int)
    med = l.groupby("loan_type")["interest_rate_pct"].transform("median")
    l["interest_rate_pct"] = l["interest_rate_pct"].fillna(med)
    issues.append(DQIssue(
        "DQ05", "loan_accounts", "Missing interest_rate_pct", "interest_rate_pct is null", n,
        "Impute the median within loan_type; add interest_rate_missing flag.",
        "Rates are tightly banded by product (e.g. cards 10-30 %, mortgages 4-9 %).", "fixed",
        int(l["interest_rate_pct"].isna().sum()), "interest_rate_pct, interest_rate_missing"))

    # DQ06 years_employed impossible vs age ------------------------------------------
    impossible = c["years_employed"] > (c["age"] - 16).clip(lower=0)
    n = int(impossible.sum())
    c["years_employed_capped"] = impossible.astype(int)
    c["years_employed"] = np.minimum(c["years_employed"], (c["age"] - 16).clip(lower=0))
    issues.append(DQIssue(
        "DQ06", "customers", "years_employed larger than age minus 16 (impossible)",
        "years_employed > max(age-16, 0)", n,
        "Cap at max(age-16, 0); add years_employed_capped flag.",
        "Nobody can have worked longer than their working-age life; capping keeps the row and the direction of the signal.",
        "fixed", int((c["years_employed"] > (c["age"] - 16).clip(lower=0)).sum()),
        "years_employed, years_employed_capped"))

    # DQ07 unemployed with income above the floor --------------------------------------
    conflict = (c["employment_type"] == "Unemployed") & (c["annual_income"] > UNEMPLOYED_INCOME_FLOOR)
    n = int(conflict.sum())
    c["income_employment_conflict"] = conflict.astype(int)
    issues.append(DQIssue(
        "DQ07", "customers", "Unemployed customers reporting income above the 12,000 floor used for all other unemployed",
        "employment_type == 'Unemployed' and annual_income > 12000", n,
        "Keep the values; add income_employment_conflict flag.",
        "Income may be rental, benefits or family support. Changing it would be a guess; the flag lets analysts exclude them.",
        "flagged", int(c["income_employment_conflict"].sum()), "income_employment_conflict"))

    # DQ08 bureau days_past_due missing ----------------------------------------------------
    miss = b["days_past_due"].isna()
    n = int(miss.sum())
    b["dpd_reported"] = (~miss).astype(int)
    issues.append(DQIssue(
        "DQ08", "bureau_credit", "Missing days_past_due on external credit lines",
        "days_past_due is null", n,
        "Do NOT impute 0. Keep NaN, add dpd_reported flag; aggregate with max() over reported lines and count unreported lines.",
        "Not reported is not the same as not late; imputing 0 would understate delinquency.",
        "flagged", int((b["dpd_reported"] == 0).sum()),
        "dpd_reported (bureau), bureau_dpd_unreported_lines (account)"))

    # DQ09 customers with no bureau file ------------------------------------------------------
    no_b = ~c["customer_id"].isin(b["customer_id"])
    n = int(no_b.sum())
    issues.append(DQIssue(
        "DQ09", "customers", "Customers with no bureau lines at all (thin file)",
        "customer_id not present in bureau_credit", n,
        "Keep via left join; counts become 0, ratios stay NaN; add no_bureau_file flag at account level.",
        "Dropping them would lose ~10 % of customers and bias the portfolio; 'no history' is itself a risk signal.",
        "flagged", n, "no_bureau_file"))

    # DQ10 loans originated before customer_since ------------------------------------------------
    tmp = l.merge(c[["customer_id", "customer_since"]], on="customer_id", how="left")
    before = (tmp["origination_date"] < tmp["customer_since"]).values
    n = int(before.sum())
    l["originated_before_customer_since"] = before.astype(int)
    issues.append(DQIssue(
        "DQ10", "loan_accounts / customers", "Loan originated before the customer relationship began",
        "origination_date < customer_since", n,
        "Keep dates; add originated_before_customer_since flag; do NOT use customer_since or 'relationship age' as a feature.",
        "About half of loans are affected, so the field is unreliable rather than a handful of typos. No way to know which date is right.",
        "flagged", int(l["originated_before_customer_since"].sum()), "originated_before_customer_since"))

    # DQ11 bureau lines opened after the last loan origination --------------------------------------
    late = (b["date_opened"] > BUREAU_CUTOFF).values
    n = int(late.sum())
    b["opened_after_data_cutoff"] = late.astype(int)
    issues.append(DQIssue(
        "DQ11", "bureau_credit", "Bureau lines opened after 2023-12-31 (later than every loan in the data); no snapshot date exists",
        "date_opened > 2023-12-31", n,
        "Keep; add opened_after_data_cutoff flag. Bureau fields are used as supplied; a point-in-time sensitivity run is reported in reports/metrics.json.",
        "The bureau extract has no as-of date, so exact point-in-time features are impossible. The sensitivity run shows how much the result depends on this.",
        "flagged", int(b["opened_after_data_cutoff"].sum()), "opened_after_data_cutoff"))

    # DQ12 collections events before origination ---------------------------------------------------------
    kk = k.merge(l[["account_id", "origination_date"]], on="account_id", how="left")
    bad = (kk["event_date"] < kk["origination_date"]).values
    n = int(bad.sum())
    k["event_before_origination"] = bad.astype(int)
    issues.append(DQIssue(
        "DQ12", "collections_events", "Collections action dated before the loan was originated",
        "event_date < origination_date", n,
        "Keep for analytics; add event_before_origination flag; EXCLUDE collections data from the default model entirely.",
        "Collections follow delinquency, so they are a consequence of the target; their dates cannot be ordered against the loan timeline.",
        "flagged", int(k["event_before_origination"].sum()), "event_before_origination; collections excluded from model"))

    # DQ13 label mismatch with the data dictionary ------------------------------------------------------------
    n = int((k["action_type"] == "SMS/Email Notice").sum())
    issues.append(DQIssue(
        "DQ13", "collections_events", "action_type 'SMS/Email Notice' is called 'SMS-Email' in DATA_DICTIONARY.md",
        "action_type == 'SMS/Email Notice'", n,
        "Keep the label in the data; documented here.", "The data is the source of truth; renaming would break joins to anything built on the raw file.",
        "retained", n, "action_type"))

    # DQ14 repayment history is always 12 months ------------------------------------------------------------------
    per = r.groupby("account_id").size()
    n = int((per != config.OBSERVATION_MONTH).sum())
    issues.append(DQIssue(
        "DQ14", "repayment_history", "Observation window: every account has exactly months 1..12, even 360-month mortgages",
        "count of months per account != 12", n,
        "Observation point = end of month 12; all repayment features use months 1..12 only.",
        "No default date or snapshot date exists. This definition is explicit and applied uniformly, so no feature can see beyond it.",
        "retained", n, "repayment features"))

    # Row-count guarantee: nothing may be dropped.
    cleaned = {"customers": c, "loans": l, "bureau": b, "repayment": r, "collections": k}
    for name, df in cleaned.items():
        assert len(df) == rows_before[name], f"{name}: rows changed {rows_before[name]} -> {len(df)}"
    # Derived bands used throughout the app.
    c["income_band"] = pd.cut(c["annual_income"], config.INCOME_BINS, labels=config.INCOME_LABELS, right=False).astype(str)
    c["age_band"] = pd.cut(c["age"], config.AGE_BINS, labels=config.AGE_LABELS, right=True).astype(str)
    return cleaned, issues


def join_audit(cleaned: dict[str, pd.DataFrame], analysis: pd.DataFrame) -> dict:
    """Row counts at each join plus key-integrity checks."""
    c, l, b, r, k = (cleaned[x] for x in ["customers", "loans", "bureau", "repayment", "collections"])
    return {
        "rows_raw": {n: int(len(df)) for n, df in cleaned.items()},
        "accounts_in": int(len(l)),
        "accounts_out": int(len(analysis)),
        "account_id_unique": bool(analysis["account_id"].is_unique),
        "orphans": {
            "loans_without_customer": int((~l["customer_id"].isin(c["customer_id"])).sum()),
            "bureau_without_customer": int((~b["customer_id"].isin(c["customer_id"])).sum()),
            "repayment_without_loan": int((~r["account_id"].isin(l["account_id"])).sum()),
            "collections_without_loan": int((~k["account_id"].isin(l["account_id"])).sum()),
        },
        "coverage": {
            "accounts_with_repayment": int(l["account_id"].isin(r["account_id"]).sum()),
            "accounts_with_collections": int(l["account_id"].isin(k["account_id"]).sum()),
            "customers_with_bureau": int(c["customer_id"].isin(b["customer_id"]).sum()),
        },
        "join_logic": [
            "loan_accounts (grain: account) LEFT JOIN customers ON customer_id  [many-to-one, validated]",
            "LEFT JOIN repayment aggregates (months 1..12 grouped by account_id)  [one row per account]",
            "LEFT JOIN bureau aggregates (grouped by customer_id)  [one row per customer, so one per account]",
            "LEFT JOIN collections aggregates (grouped by account_id; analytics only)  [one row per account]",
        ],
    }


def write_dq_log(issues: list[DQIssue], audit: dict) -> None:
    config.REPORTS.mkdir(exist_ok=True)
    config.DOCS.mkdir(exist_ok=True)
    (config.REPORTS / "dq_log.json").write_text(json.dumps([asdict(i) for i in issues], indent=2))
    (config.REPORTS / "join_audit.json").write_text(json.dumps(audit, indent=2))
    lines = [
        "# Data quality log and join logic",
        "",
        "Generated by `src/ingestion.py` (do not edit by hand; edit the code and rebuild).",
        "`n_raw` is measured on the raw CSVs, `n_unresolved` is the same detector re-run on the cleaned output.",
        "No rows are dropped anywhere: row counts are asserted equal before and after cleaning.",
        "",
        "## Join logic",
        "",
    ]
    lines += [f"{i+1}. {s}" for i, s in enumerate(audit["join_logic"])]
    lines += [
        "",
        f"Accounts in: **{audit['accounts_in']:,}**, accounts out: **{audit['accounts_out']:,}**, "
        f"`account_id` unique: **{audit['account_id_unique']}**, orphans: **{sum(audit['orphans'].values())}**.",
        "",
        "## Issues and decisions",
        "",
        "| ID | Table | Issue | n_raw | Decision | Why | Handling | n_unresolved |",
        "|---|---|---|---:|---|---|---|---:|",
    ]
    for i in issues:
        lines.append(f"| {i.id} | {i.table} | {i.issue} | {i.n_raw:,} | {i.decision} | {i.rationale} | {i.handling} | {i.n_unresolved:,} |")
    lines += [
        "",
        "## Handling types",
        "- **fixed**: value corrected or imputed; the detector finds 0 remaining.",
        "- **flagged**: value kept, an indicator column marks every affected row (analysts can filter); `n_unresolved` equals the number of flagged rows.",
        "- **retained**: documented behaviour of the data, no change needed.",
        "",
        "## Modelling-related decisions",
        "- Observation point is the end of repayment month 12; the target has no date, so it is assumed to be observed after that point.",
        "- Collections data is excluded from the model (it follows delinquency).",
        "- `gender` and `marital_status` are not used as model inputs (protected attributes); they remain available for portfolio profiling.",
        "- Imputation statistics use the whole table but never the target.",
    ]
    (config.DOCS / "data_quality_log.md").write_text("\n".join(lines) + "\n")
