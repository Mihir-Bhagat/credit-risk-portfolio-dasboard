"""Feature engineering and the one analysis-ready, account-grain dataset.

Cutoff logic (leakage control), also written to reports/feature_catalog.csv:
  * Observation point = end of repayment month 12. Repayment features use payment_month 1..12 only.
  * Collections events are NOT model features (they follow delinquency); analytics only.
  * Bureau fields are used as supplied (the extract has no as-of date); a point-in-time sensitivity run
    (lines opened on/before origination + 12 months) is reported in reports/metrics.json.
  * target_default, ids and dates are never model inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------------------
# Model feature lists (single source of truth for model, catalog and tests)
# --------------------------------------------------------------------------------------
MODEL_CATEGORICAL = [
    "loan_type", "loan_purpose", "disbursal_channel", "employment_type", "occupation_sector",
    "education_level", "region", "city_tier", "housing_type",
]

# name -> (source table, description, cutoff logic)
NUMERIC_CATALOG: dict[str, tuple[str, str, str]] = {
    # loan & customer
    "loan_amount": ("loan_accounts", "Original loan / credit amount", "known at origination"),
    "interest_rate_pct": ("loan_accounts", "Annual interest rate (imputed within loan type)", "known at origination"),
    "tenure_months": ("loan_accounts", "Loan tenure in months", "known at origination"),
    "collateral_flag": ("loan_accounts", "1 if secured", "known at origination"),
    "interest_rate_missing": ("loan_accounts", "Interest rate was missing", "known at origination"),
    "age": ("customers", "Age in years", "profile attribute"),
    "annual_income": ("customers", "Annual income in USD", "profile attribute"),
    "years_employed": ("customers", "Years in current employment (cleaned)", "profile attribute"),
    "num_dependents": ("customers", "Number of dependents (imputed)", "profile attribute"),
    "years_at_residence": ("customers", "Years at residence (imputed)", "profile attribute"),
    "years_employed_missing": ("customers", "years_employed was missing", "profile attribute"),
    "loan_to_income": ("derived", "loan_amount / annual_income", "origination + profile"),
    "emi_to_income": ("derived", "12 x monthly EMI / annual_income (payment burden)", "origination + profile"),
    "n_accounts_customer": ("loan_accounts", "Number of loan accounts held by the customer", "profile attribute"),
    # repayment (months 1..12 only)
    "rep_n_missed": ("repayment_history", "Months with status Missed (of 12)", "months 1-12"),
    "rep_n_late": ("repayment_history", "Months with status Late (of 12)", "months 1-12"),
    "rep_max_dpd": ("repayment_history", "Maximum days past due", "months 1-12"),
    "rep_mean_dpd": ("repayment_history", "Mean days past due", "months 1-12"),
    "rep_paid_ratio": ("repayment_history", "Total paid / total due", "months 1-12"),
    "rep_missed_first3": ("repayment_history", "Missed payments in months 1-3 (early warning)", "months 1-3"),
    "rep_missed_last3": ("repayment_history", "Missed payments in months 10-12", "months 10-12"),
    "rep_dpd_last3": ("repayment_history", "Mean DPD months 10-12", "months 10-12"),
    "rep_dpd_first9": ("repayment_history", "Mean DPD months 1-9", "months 1-9"),
    "rep_dpd_trend": ("repayment_history", "Mean DPD last 3 months minus first 9 months", "months 1-12"),
    "rep_paid_ratio_last3": ("repayment_history", "Paid / due in months 10-12", "months 10-12"),
    "rep_paid_ratio_first9": ("repayment_history", "Paid / due in months 1-9", "months 1-9"),
    "rep_longest_miss_streak": ("repayment_history", "Longest run of consecutive missed months", "months 1-12"),
    "rep_paydown_pct": ("repayment_history", "Balance reduction between month 1 and month 12", "months 1-12"),
    # bureau (customer level)
    "bur_n_lines": ("bureau_credit", "Number of external credit lines", "bureau extract as supplied"),
    "bur_n_active": ("bureau_credit", "Active external lines", "bureau extract as supplied"),
    "bur_n_written_off": ("bureau_credit", "Written-off external lines", "bureau extract as supplied"),
    "bur_total_sanctioned": ("bureau_credit", "Sum of sanctioned amounts", "bureau extract as supplied"),
    "bur_total_outstanding": ("bureau_credit", "Sum of outstanding amounts", "bureau extract as supplied"),
    "bur_utilization": ("bureau_credit", "Outstanding / sanctioned (NaN without a file)", "bureau extract as supplied"),
    "bur_max_dpd": ("bureau_credit", "Max reported days past due (NaN if none reported)", "bureau extract as supplied"),
    "bur_mean_dpd": ("bureau_credit", "Mean reported days past due", "bureau extract as supplied"),
    "bur_n_dpd30": ("bureau_credit", "Lines with DPD >= 30", "bureau extract as supplied"),
    "bur_dpd_unreported": ("bureau_credit", "Lines with DPD not reported", "bureau extract as supplied"),
    "bur_share_fintech_nbfc": ("bureau_credit", "Share of lines from NBFC / Fintech lenders", "bureau extract as supplied"),
    "no_bureau_file": ("bureau_credit", "Customer has no bureau lines", "bureau extract as supplied"),
}
MODEL_NUMERIC = list(NUMERIC_CATALOG)
MODEL_FEATURES = MODEL_CATEGORICAL + MODEL_NUMERIC

CATEGORICAL_CATALOG = {
    c: ("customers" if c in {"employment_type", "occupation_sector", "education_level", "region", "city_tier", "housing_type"} else "loan_accounts",
        f"{c} (categorical)", "known at origination / profile attribute")
    for c in MODEL_CATEGORICAL
}

EXCLUDED = {
    "target_default": "the target",
    "gender": "protected attribute (profiling only)",
    "marital_status": "protected attribute (profiling only)",
    "customer_since": "unreliable: 50 % of loans originate before it (DQ10)",
    "origination_date": "calendar date; not used as a predictor",
    "collections_events.*": "follows delinquency: consequence of the target, excluded (DQ12)",
    "account_id / customer_id": "identifiers",
}


def feature_catalog() -> pd.DataFrame:
    rows = [(n, s, "numeric", d, c) for n, (s, d, c) in NUMERIC_CATALOG.items()]
    rows += [(n, s, "categorical", d, c) for n, (s, d, c) in CATEGORICAL_CATALOG.items()]
    cat = pd.DataFrame(rows, columns=["feature", "source", "type", "description", "cutoff_logic"])
    return cat


# --------------------------------------------------------------------------------------
# Aggregations
# --------------------------------------------------------------------------------------
def _longest_streak(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def repayment_features(rep: pd.DataFrame) -> pd.DataFrame:
    """Per-account aggregates over payment_month 1..12 (the observation window)."""
    r = rep[rep["payment_month"].between(1, config.OBSERVATION_MONTH)].copy()
    r["missed"] = (r["payment_status"] == "Missed").astype(int)
    r["late"] = (r["payment_status"] == "Late").astype(int)
    r["is_last3"] = r["payment_month"] >= 10
    r["is_first3"] = r["payment_month"] <= 3
    r["is_first9"] = r["payment_month"] <= 9
    g = r.groupby("account_id")
    out = pd.DataFrame({
        "rep_n_missed": g["missed"].sum(),
        "rep_n_late": g["late"].sum(),
        "rep_max_dpd": g["days_past_due"].max(),
        "rep_mean_dpd": g["days_past_due"].mean(),
        "emi_due_monthly": g["emi_due"].median(),
        "paid_sum": g["emi_paid"].sum(),
        "due_sum": g["emi_due"].sum(),
    })
    out["rep_paid_ratio"] = out["paid_sum"] / out["due_sum"].replace(0, np.nan)
    f3 = r[r["is_first3"]].groupby("account_id")["missed"].sum()
    out["rep_missed_first3"] = f3.reindex(out.index).fillna(0)
    l3 = r[r["is_last3"]]
    out["rep_missed_last3"] = l3.groupby("account_id")["missed"].sum().reindex(out.index).fillna(0)
    out["rep_dpd_last3"] = l3.groupby("account_id")["days_past_due"].mean().reindex(out.index)
    f9 = r[r["is_first9"]]
    out["rep_dpd_first9"] = f9.groupby("account_id")["days_past_due"].mean().reindex(out.index)
    out["rep_dpd_trend"] = out["rep_dpd_last3"] - out["rep_dpd_first9"]
    pl3 = l3.groupby("account_id")[["emi_paid", "emi_due"]].sum()
    out["rep_paid_ratio_last3"] = (pl3["emi_paid"] / pl3["emi_due"].replace(0, np.nan)).reindex(out.index)
    pf9 = f9.groupby("account_id")[["emi_paid", "emi_due"]].sum()
    out["rep_paid_ratio_first9"] = (pf9["emi_paid"] / pf9["emi_due"].replace(0, np.nan)).reindex(out.index)
    rs = r.sort_values(["account_id", "payment_month"])
    streak = rs.groupby("account_id")["missed"].apply(lambda s: _longest_streak(s.to_numpy()))
    out["rep_longest_miss_streak"] = streak.reindex(out.index)
    bal = rs.groupby("account_id")["outstanding_balance"].agg(["first", "last"])
    out["rep_paydown_pct"] = ((bal["first"] - bal["last"]) / bal["first"].replace(0, np.nan)).reindex(out.index)
    return out.drop(columns=["paid_sum", "due_sum"]).reset_index()


def bureau_features(bureau: pd.DataFrame, key: str = "customer_id") -> pd.DataFrame:
    """Aggregates of external credit lines per `key` (customer_id by default)."""
    b = bureau.copy()
    b["is_active"] = (b["account_status"] == "Active").astype(int)
    b["is_wo"] = (b["account_status"] == "Written-Off").astype(int)
    b["dpd30"] = (b["days_past_due"] >= 30).astype(int)
    b["unrep"] = b["days_past_due"].isna().astype(int)
    b["fin_nbfc"] = b["lender_category"].isin(["NBFC", "Fintech"]).astype(int)
    g = b.groupby(key)
    out = pd.DataFrame({
        "bur_n_lines": g.size(),
        "bur_n_active": g["is_active"].sum(),
        "bur_n_written_off": g["is_wo"].sum(),
        "bur_total_sanctioned": g["sanctioned_amount"].sum(),
        "bur_total_outstanding": g["outstanding_amount"].sum(),
        "bur_max_dpd": g["days_past_due"].max(),
        "bur_mean_dpd": g["days_past_due"].mean(),
        "bur_n_dpd30": g["dpd30"].sum(),
        "bur_dpd_unreported": g["unrep"].sum(),
        "bur_share_fintech_nbfc": g["fin_nbfc"].mean(),
    })
    out["bur_utilization"] = out["bur_total_outstanding"] / out["bur_total_sanctioned"].replace(0, np.nan)
    return out.reset_index()


def bureau_features_point_in_time(bureau: pd.DataFrame, loans: pd.DataFrame, window_days: int = 365) -> pd.DataFrame:
    """Sensitivity variant: per ACCOUNT, only bureau lines opened on/before origination + window (observation point)."""
    m = loans[["account_id", "customer_id", "origination_date"]].merge(bureau, on="customer_id", how="inner")
    m = m[m["date_opened"] <= m["origination_date"] + pd.Timedelta(days=window_days)]
    return bureau_features(m, key="account_id")


def collections_features(coll: pd.DataFrame) -> pd.DataFrame:
    """Per-account collections aggregates. ANALYTICS ONLY, never model inputs."""
    g = coll.groupby("account_id")
    out = pd.DataFrame({
        "coll_n_events": g.size(),
        "coll_amount_recovered": g["amount_recovered"].sum(),
        "coll_had_legal": g["action_type"].apply(lambda s: int((s == "Legal Notice").any())),
        "coll_events_before_origination": g["event_before_origination"].sum(),
    })
    return out.reset_index()


def build_analysis_dataset(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One clean, analysis-ready table at account grain (one row per loan account)."""
    c, l, b, r, k = (t[x] for x in ["customers", "loans", "bureau", "repayment", "collections"])
    keep_c = [x for x in c.columns]
    df = l.merge(c[keep_c], on="customer_id", how="left", validate="many_to_one")
    df = df.merge(repayment_features(r), on="account_id", how="left", validate="one_to_one")
    df = df.merge(bureau_features(b), on="customer_id", how="left", validate="many_to_one")
    df = df.merge(collections_features(k), on="account_id", how="left", validate="one_to_one")

    # Missing-bureau handling (DQ09): counts become 0, ratios stay NaN.
    df["no_bureau_file"] = df["bur_n_lines"].isna().astype(int)
    for col in ["bur_n_lines", "bur_n_active", "bur_n_written_off", "bur_total_sanctioned",
                "bur_total_outstanding", "bur_n_dpd30", "bur_dpd_unreported"]:
        df[col] = df[col].fillna(0)
    # No collections activity = zero events (analytics only).
    for col in ["coll_n_events", "coll_amount_recovered", "coll_had_legal", "coll_events_before_origination"]:
        df[col] = df[col].fillna(0)

    cust_n = df.groupby("customer_id")["account_id"].transform("size")
    df["n_accounts_customer"] = cust_n
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
    df["emi_to_income"] = df["emi_due_monthly"] * 12 / df["annual_income"]
    df["origination_year"] = df["origination_date"].dt.year.astype(int)
    df["origination_quarter"] = df["origination_date"].dt.to_period("Q").astype(str)
    df["exposure"] = df["loan_amount"]  # original amount; no current balance exists at account level
    df["any_missed"] = (df["rep_n_missed"] > 0).astype(int)
    assert df["account_id"].is_unique
    return df
