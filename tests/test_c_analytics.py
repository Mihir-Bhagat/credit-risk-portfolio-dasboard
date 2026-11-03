"""C. Portfolio analytics (checklist items 29-32)."""
from __future__ import annotations

import pandas as pd
import pytest

import oracle


@pytest.mark.item(29)
def test_profile_covers_all_five_dimensions(root):
    prof = pd.read_csv(root / "reports" / "segment_profiles.csv")
    assert set(prof["dimension"]) == {"Income band", "Employment type", "Region", "Age band", "Loan type"}
    for d, g in prof.groupby("dimension"):
        assert g["accounts"].sum() == 14685, d
        assert abs(g["share_of_exposure"].sum() - 1) < 1e-9


@pytest.mark.item(30)
def test_four_to_five_insights_each_with_a_so_what(insights):
    ins = insights["insights"]
    assert 4 <= len(ins) <= 5
    for i in ins:
        assert len(i["title"]) > 15 and len(i["so_what"]) > 80, i["id"]
        assert i["evidence"], i["id"]
    assert len({i["id"] for i in ins}) == len(ins)


@pytest.mark.item(31)
def test_at_least_one_cross_dimensional_insight(insights):
    assert any(len(set(i["dims"]) - {"repayment"}) >= 2 for i in insights["insights"])


@pytest.mark.item(32)
def test_insight_numbers_match_an_independent_recomputation(raw, insights):
    df = oracle.load(raw)
    base = df["target_default"].mean()
    by = {i["id"]: i["evidence"] for i in insights["insights"]}
    # I1 loan type
    lt = oracle.by(df, "loan_type")
    top = lt["value"].idxmax()
    e = by["I1"]
    assert e["segment"] == top and abs(e["default_rate"] - lt.loc[top, "value"]) < 1e-9 and abs(e["portfolio_default_rate"] - base) < 1e-9
    assert abs(e["share_of_exposure"] - df.loc[df.loan_type == top, "loan_amount"].sum() / df["loan_amount"].sum()) < 1e-9
    # I2 income and age
    inc = oracle.by(df, "income_band")
    e = by["I2"]
    assert abs(e["low_rate"] - inc.loc["<30k", "value"]) < 1e-9 and abs(e["high_rate"] - inc.loc["120k+", "value"]) < 1e-9
    u60 = df[df.income_band.isin(["<30k", "30-60k"])]
    assert abs(e["under60k_share_accounts"] - len(u60) / len(df)) < 1e-9
    assert abs(e["under60k_share_defaults"] - u60.target_default.sum() / df.target_default.sum()) < 1e-9
    age = oracle.by(df, "age_band")
    assert abs(e["youngest_rate"] - age.loc["<25", "value"]) < 1e-9 and abs(e["oldest_rate"] - age.loc["55+", "value"]) < 1e-9
    # I3 employment
    emp = oracle.by(df, "employment_type")
    e = by["I3"]
    assert e["segment"] == emp["value"].idxmax() and abs(e["default_rate"] - emp["value"].max()) < 1e-9 and abs(e["salaried_rate"] - emp.loc["Salaried", "value"]) < 1e-9
    # I4 cross cut
    cell = df.groupby(["loan_type", "employment_type"]).agg(n=("target_default", "size"), r=("target_default", "mean"))
    cell = cell[cell.n >= 100].sort_values("r", ascending=False).iloc[0]
    e = by["I4"]
    assert (e["loan_type"], e["employment_type"]) == cell.name and abs(e["default_rate"] - cell.r) < 1e-9 and e["accounts"] == cell.n
    # I5 early warning
    r = raw["repayment"]
    early = r[(r.payment_month <= 3) & (r.payment_status == "Missed")].groupby("account_id").size()
    flag = df.account_id.isin(early.index)
    e = by["I5"]
    assert e["flagged_accounts"] == flag.sum() and abs(e["rate_with_early_miss"] - df.loc[flag, "target_default"].mean()) < 1e-9
    assert abs(e["rate_without"] - df.loc[~flag, "target_default"].mean()) < 1e-9
