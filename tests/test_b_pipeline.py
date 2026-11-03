"""B. Data pipeline (checklist items 20-28). Expectations are computed independently from the raw CSVs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BUR_CUTOFF = pd.Timestamp("2023-12-31")


@pytest.fixture(scope="module")
def analysis():
    return pd.read_parquet(ROOT / "data" / "processed" / "analysis_dataset.parquet")


@pytest.fixture(scope="module")
def dq():
    return json.loads((ROOT / "reports" / "dq_log.json").read_text())


@pytest.fixture(scope="module")
def audit():
    return json.loads((ROOT / "reports" / "join_audit.json").read_text())


@pytest.mark.item(20)
def test_all_five_tables_feed_the_final_dataset(analysis):
    expect = {"customers": "age", "loans": "loan_amount", "bureau": "bur_n_lines", "repayment": "rep_n_missed", "collections": "coll_n_events"}
    for t, col in expect.items():
        assert col in analysis.columns and analysis[col].notna().any(), f"no data from {t} ({col})"


@pytest.mark.item(21)
def test_one_clean_account_grain_dataset(analysis, raw):
    assert len(analysis) == len(raw["loans"]) == 14685
    assert analysis["account_id"].is_unique
    assert set(analysis["account_id"]) == set(raw["loans"]["account_id"])
    assert analysis["target_default"].isin([0, 1]).all()
    from src import features as F
    nan_cols = analysis[F.MODEL_FEATURES].isna().sum()
    allowed_nan = {"bur_utilization", "bur_max_dpd", "bur_mean_dpd", "bur_share_fintech_nbfc"}  # documented: no bureau file / DPD not reported (DQ08, DQ09)
    unexpected = set(nan_cols[nan_cols > 0].index) - allowed_nan
    assert not unexpected, f"unhandled NaN in model features: {unexpected}"
    # the allowed NaNs must be fully explained by the documented causes
    assert analysis.loc[analysis["bur_utilization"].isna(), "no_bureau_file"].eq(1).all()
    assert analysis.loc[analysis["no_bureau_file"] == 1, "bur_n_lines"].eq(0).all()


@pytest.mark.item(22)
def test_join_row_counts_no_explosion_no_orphans_independent_check(raw, audit, analysis):
    c, l, b, r, k = (raw[x] for x in ["customers", "loans", "bureau", "repayment", "collections"])
    n = len(l)
    assert c["customer_id"].is_unique and l["account_id"].is_unique
    j1 = l.merge(c, on="customer_id", how="left")
    assert len(j1) == n and j1["age"].notna().all(), "customer join changed row count or lost customers"
    rep_agg = r.groupby("account_id").size().rename("n")
    assert len(j1.merge(rep_agg, on="account_id", how="left")) == n
    bur_agg = b.groupby("customer_id").size().rename("n")
    assert len(j1.merge(bur_agg, on="customer_id", how="left")) == n
    col_agg = k.groupby("account_id").size().rename("n")
    assert len(j1.merge(col_agg, on="account_id", how="left")) == n
    assert not (~l["customer_id"].isin(c["customer_id"])).any() and not (~b["customer_id"].isin(c["customer_id"])).any()
    assert not (~r["account_id"].isin(l["account_id"])).any() and not (~k["account_id"].isin(l["account_id"])).any()
    assert audit["accounts_in"] == audit["accounts_out"] == n and audit["account_id_unique"] and sum(audit["orphans"].values()) == 0
    # aggregated values equal an independent recomputation
    chk = analysis.set_index("account_id")
    mism = r[r["payment_status"] == "Missed"].groupby("account_id").size().reindex(chk.index).fillna(0)
    assert (chk["rep_n_missed"] == mism).all()
    nlines = b.groupby("customer_id").size()
    assert (chk["bur_n_lines"] == chk["customer_id"].map(nlines).fillna(0)).all()


@pytest.mark.item(23)
def test_dq_log_documents_every_decision(dq, root):
    assert len(dq) >= 12
    for i in dq:
        for f in ("id", "table", "issue", "detector", "n_raw", "decision", "rationale", "handling"):
            assert i[f] not in (None, ""), (i["id"], f)
    md = (root / "docs" / "data_quality_log.md").read_text()
    for i in dq:
        assert i["id"] in md
    assert "Join logic" in md and "LEFT JOIN" in md


def _detectors(raw):
    """Independent detectors, written from the DATA_DICTIONARY and my profiling, never calling src/."""
    c, l, b, r, k = (raw[x] for x in ["customers", "loans", "bureau", "repayment", "collections"])
    tmp = l.merge(c[["customer_id", "customer_since"]], on="customer_id")
    kk = k.merge(l[["account_id", "origination_date"]], on="account_id")
    return {
        "DQ01": int(c["gender"].isin(["M", "F"]).sum()),
        "DQ02": int(c["years_employed"].isna().sum()),
        "DQ03": int(c["num_dependents"].isna().sum()),
        "DQ04": int(c["years_at_residence"].isna().sum()),
        "DQ05": int(l["interest_rate_pct"].isna().sum()),
        "DQ06": int((c["years_employed"] > (c["age"] - 16).clip(lower=0)).sum()),
        "DQ07": int(((c["employment_type"] == "Unemployed") & (c["annual_income"] > 12000)).sum()),
        "DQ08": int(b["days_past_due"].isna().sum()),
        "DQ09": int((~c["customer_id"].isin(b["customer_id"])).sum()),
        "DQ10": int((pd.to_datetime(tmp["origination_date"]) < pd.to_datetime(tmp["customer_since"])).sum()),
        "DQ11": int((pd.to_datetime(b["date_opened"]) > BUR_CUTOFF).sum()),
        "DQ12": int((pd.to_datetime(kk["event_date"]) < pd.to_datetime(kk["origination_date"])).sum()),
        "DQ13": int((k["action_type"] == "SMS/Email Notice").sum()),
        "DQ14": int((r.groupby("account_id").size() != 12).sum()),
    }


@pytest.mark.item(24, 25)
def test_every_logged_issue_is_actually_fixed_in_the_output(raw, dq):
    from src import ingestion
    counts = _detectors(raw)
    cleaned, _ = ingestion.clean_tables(ingestion.load_raw())
    c, l, b, k = cleaned["customers"], cleaned["loans"], cleaned["bureau"], cleaned["collections"]
    by_id = {i["id"]: i for i in dq}
    # log counts equal independently measured raw counts
    for k_, n in counts.items():
        assert by_id[k_]["n_raw"] == n, f"{k_}: log says {by_id[k_]['n_raw']}, raw data says {n}"
    # 'fixed' issues: the same detector finds nothing in the cleaned data
    assert not c["gender"].isin(["M", "F"]).any() and set(c["gender"]) <= {"Male", "Female"}
    assert c["years_employed"].notna().all() and c["num_dependents"].notna().all() and c["years_at_residence"].notna().all()
    assert l["interest_rate_pct"].notna().all()
    assert not (c["years_employed"] > (c["age"] - 16).clip(lower=0)).any()
    # 'flagged' issues: a flag column marks exactly the affected rows
    assert c["years_employed_missing"].sum() == counts["DQ02"] and c["num_dependents_missing"].sum() == counts["DQ03"]
    assert c["years_at_residence_missing"].sum() == counts["DQ04"] and l["interest_rate_missing"].sum() == counts["DQ05"]
    assert c["years_employed_capped"].sum() == counts["DQ06"]
    assert c["income_employment_conflict"].sum() == counts["DQ07"]
    assert (b["dpd_reported"] == 0).sum() == counts["DQ08"] and b["days_past_due"].isna().sum() == counts["DQ08"]  # NOT imputed with 0
    assert l["originated_before_customer_since"].sum() == counts["DQ10"]
    assert b["opened_after_data_cutoff"].sum() == counts["DQ11"]
    assert k["event_before_origination"].sum() == counts["DQ12"]
    # no rows lost anywhere
    for name, df in cleaned.items():
        assert len(df) == len(raw[name]), name
    # log's own re-check column agrees
    for i in dq:
        if i["handling"] == "fixed":
            assert i["n_unresolved"] == 0, i["id"]
        elif i["handling"] == "flagged":
            assert i["n_unresolved"] == i["n_raw"], i["id"]


@pytest.mark.item(26)
def test_pipeline_is_deterministic_from_raw():
    from src import features as F, ingestion
    outs = []
    for _ in range(2):
        cl, _ = ingestion.clean_tables(ingestion.load_raw())
        outs.append(F.build_analysis_dataset(cl))
    pd.testing.assert_frame_equal(outs[0], outs[1])


@pytest.mark.item(27)
def test_no_future_information_repayment_cutoff_is_enforced():
    """Append fake months 13-24 with extreme values: features must not change (observation point = end of month 12)."""
    from src import features as F, ingestion
    cl, _ = ingestion.clean_tables(ingestion.load_raw())
    base = F.repayment_features(cl["repayment"]).set_index("account_id")
    extra = cl["repayment"][cl["repayment"]["payment_month"] == 12].copy()
    frames = []
    for m in range(13, 25):
        e = extra.copy()
        e["payment_month"], e["days_past_due"], e["emi_paid"], e["payment_status"], e["outstanding_balance"] = m, 89, 0.0, "Missed", 1e9
        frames.append(e)
    polluted = pd.concat([cl["repayment"], *frames], ignore_index=True)
    after = F.repayment_features(polluted).set_index("account_id")
    pd.testing.assert_frame_equal(base.sort_index(), after.sort_index())


@pytest.mark.item(27)
def test_model_inputs_exclude_target_collections_protected_and_unreliable_fields(analysis):
    from src import features as F
    forbidden = {"target_default", "gender", "marital_status", "customer_since", "origination_date", "account_id", "customer_id"}
    assert not forbidden & set(F.MODEL_FEATURES)
    assert not [f for f in F.MODEL_FEATURES if f.startswith("coll_") or "collection" in f]
    cat = pd.read_csv(ROOT / "reports" / "feature_catalog.csv")
    assert set(cat["feature"]) == set(F.MODEL_FEATURES) and cat["cutoff_logic"].notna().all() and cat["source"].notna().all()
    assert {"repayment_history", "bureau_credit"} <= set(cat["source"])
    assert (ROOT / "reports" / "feature_exclusions.json").exists()


@pytest.mark.item(27)
def test_no_single_feature_has_auc_above_0_95(analysis, metrics):
    from src import features as F
    y = analysis["target_default"]
    worst = ("", 0.0)
    for f in F.MODEL_NUMERIC:
        v = analysis[f].fillna(analysis[f].median())
        if v.nunique() < 2:
            continue
        a = roc_auc_score(y, v)
        a = max(a, 1 - a)
        worst = max(worst, (f, a), key=lambda t: t[1])
    print("strongest single feature:", worst)
    assert worst[1] <= 0.95, f"possible leakage: {worst}"
    for c in F.MODEL_CATEGORICAL:  # categorical: AUC of the in-sample mean-target encoding is an upper bound
        enc = analysis.groupby(c)["target_default"].transform("mean")
        assert roc_auc_score(y, enc) <= 0.95, c
    assert abs(metrics["sensitivity"]["max_single_feature_auc"] - worst[1]) < 1e-6


@pytest.mark.item(28)
def test_issues_found_by_an_independent_profile_are_all_logged(raw, dq):
    logged_text = " ".join(f"{i['table']} {i['issue']} {i['output_columns'] if 'output_columns' in i else ''} {i['detector']}".lower() for i in dq)
    # 1. every column with nulls in any raw table must be mentioned (by name) in the log
    for tname, df in raw.items():
        for col in df.columns[df.isna().any()]:
            if tname == "bureau" and col == "date_closed":
                continue  # structurally null for active lines (verified below)
            assert col.lower() in logged_text or col.replace("_", " ").lower() in logged_text, f"null column {tname}.{col} is not in the data-quality log"
    b = raw["bureau"]
    assert b.loc[b["account_status"] == "Active", "date_closed"].isna().all() and b.loc[b["account_status"] != "Active", "date_closed"].notna().all()
    # 2. categorical label inconsistencies
    c = raw["customers"]
    for col in c.select_dtypes("object").columns:
        if col in ("customer_id", "customer_since"):
            continue
        vals = c[col].dropna().astype(str)
        low = vals.str.lower().str.strip()
        assert low.nunique() == vals.nunique() or col == "gender", f"case/space variants in {col}"
    assert {"M", "F"} <= set(c["gender"]) and any(i["id"] == "DQ01" for i in dq)
    # 3. logical/date inconsistencies the log must contain
    for needle in ["customer_since", "origination", "after 2023-12-31", "before the loan", "years_employed larger than age"]:
        assert needle.lower() in logged_text, f"missing documented issue: {needle}"
    # 4. duplicates and key integrity: none exist, and the log must not claim otherwise
    for tname, key in [("customers", "customer_id"), ("loans", "account_id"), ("bureau", "bureau_id"), ("collections", "collection_event_id")]:
        assert raw[tname][key].is_unique
    assert not raw["repayment"].duplicated(["account_id", "payment_month"]).any()
