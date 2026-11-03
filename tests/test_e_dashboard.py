"""E. Dashboard (checklist items 43-49), tested with Streamlit's AppTest."""
from __future__ import annotations

import pandas as pd
import pytest

import oracle
from apphelp import VIEWS, all_text, goto, metrics, n_charts, new_app


@pytest.mark.item(48, 4)
@pytest.mark.parametrize("view", VIEWS)
def test_every_view_loads_without_exception(view):
    at = new_app(view)
    assert not at.exception
    if view == "Ask the Portfolio":  # an input view: nothing to plot until a question is asked
        assert len(at.text_input) == 1 and len([b for b in at.button if b.key and b.key.startswith("preset_")]) >= 10
    else:
        assert n_charts(at) + len(at.dataframe) > 0, f"{view} rendered nothing"


@pytest.mark.item(43)
def test_portfolio_view_has_risk_segments_exposure_and_trends_across_dimensions():
    at = new_app("Overview")
    subs = [s.value for s in at.subheader]
    for s in ["Risk segments", "Trends across dimensions", "Profile by dimension", "Cross-dimensional cut"]:
        assert s in subs, subs
    assert n_charts(at) >= 7  # risk bands x2, trend x2, profile x2, heatmap
    labels = [m.label for m in at.metric]
    assert "Exposure (original amounts)" in labels and "Observed default rate" in labels
    # every profile dimension named in the brief is selectable
    opts = at.selectbox(key="profile_dim").options
    for d in ["Income band", "Employment type", "Region", "Age band", "Loan type"]:
        assert d in opts
    # the trend can be broken down by a dimension
    at.selectbox(key="trend_by").set_value("Loan type").run()
    assert not at.exception and n_charts(at) >= 7


@pytest.mark.item(44)
def test_account_view_shows_score_and_key_drivers_matching_model_output(portfolio):
    r = portfolio.sort_values("risk_score").iloc[-5]
    at = new_app("Account")
    at.selectbox(key="acct_id").set_value(r["account_id"]).run()
    assert not at.exception
    txt = all_text(at)
    assert f"{r['risk_score'] * 100:.1f}%" in txt, "risk score not shown"
    assert f"{r['risk_band']} risk" in txt
    assert n_charts(at) >= 3  # drivers, EMI, DPD
    import json
    drivers = json.loads(r["drivers_json"])
    figs = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]  # decode: plotly escapes "/" as \u002f
    drawn = [y for f in figs for t in f["data"] for y in (t.get("y") if isinstance(t.get("y"), list) else [])]
    for d in drivers:  # every driver of this account is drawn in the drivers chart, with its value
        assert any(y == f"{d['label']} = {d['value']}" for y in drawn), (d, drawn)
    assert len(at.dataframe) >= 1  # facts table


@pytest.mark.item(45, 47)
def test_filter_changes_displayed_numbers_and_kpis_match_independent_pandas(raw, portfolio):
    df = oracle.load(raw)
    at = new_app("Overview")
    m0 = metrics(at)
    assert m0[0] == f"{len(df):,}" and m0[2] == f"{df.target_default.mean() * 100:.1f}%" and m0[1] == f"${df.loan_amount.sum() / 1e6:,.1f}M"
    assert m0[3] == f"{portfolio.risk_score.mean() * 100:.1f}%"
    # filter: Mortgage
    at.sidebar.multiselect(key="f_loan_type").set_value(["Mortgage"]).run()
    m1 = metrics(at)
    mg = df[df.loan_type == "Mortgage"]
    assert m1 != m0
    assert m1[0] == f"{len(mg):,}" == "1,432"
    assert m1[1] == f"${mg.loan_amount.sum() / 1e6:,.1f}M"
    assert m1[2] == f"{mg.target_default.mean() * 100:.1f}%"
    pm = portfolio[portfolio.loan_type == "Mortgage"]
    assert m1[3] == f"{pm.risk_score.mean() * 100:.1f}%" and m1[4] == f"{int(pm.flagged.sum()):,}"
    # two filters combined
    at.sidebar.multiselect(key="f_region").set_value(["North"]).run()
    both = df[(df.loan_type == "Mortgage") & (df.region == "North")]
    m2 = metrics(at)
    assert m2[0] == f"{len(both):,}" and m2[2] == f"{both.target_default.mean() * 100:.1f}%"
    # year slider
    at.sidebar.slider(key="f_year").set_range(2021, 2023).run()
    y = both[both.origination_year >= 2021]
    assert metrics(at)[0] == f"{len(y):,}"
    # reset restores the portfolio
    at.sidebar.button(key="reset_filters").click().run()
    assert metrics(at)[0] == f"{len(df):,}"


@pytest.mark.item(45)
def test_drill_down_lists_accounts_of_the_selected_segment(portfolio):
    at = new_app("Segments & drill-down")
    at.selectbox(key="seg_dim").set_value("Employment type").run()
    at.selectbox(key="seg_value").set_value("Unemployed").run()
    assert not at.exception
    sub = portfolio[portfolio.employment_type == "Unemployed"]
    assert metrics(at)[0] == f"{len(sub):,}"
    table = at.dataframe[-1].value  # the account list
    assert len(table) == min(500, len(sub))
    assert (table["Employment"] == "Unemployed").all()
    assert table["Predicted PD"].is_monotonic_decreasing
    assert set(table["Account"]) <= set(sub.account_id)
    # a different dimension drills into a different set
    at.selectbox(key="seg_dim").set_value("Region").run()
    at.selectbox(key="seg_value").set_value("West").run()
    assert (at.dataframe[-1].value["Region"] == "West").all()
    # "Open in Account view" jumps to the chosen account
    pick = at.selectbox(key="seg_pick").value
    at.button(key="seg_open").click().run()
    assert at.sidebar.radio(key="view").value == "Account" and at.selectbox(key="acct_id").value == pick


@pytest.mark.item(46)
def test_both_ai_features_are_embedded_in_the_dashboard():
    at = new_app("Ask the Portfolio")
    assert at.text_input(key="ask_q") is not None and len(at.button) >= 10
    assert "How Ask the Portfolio works" in all_text(at)
    at = goto(at, "Anomaly review")
    assert len(at.dataframe) >= 1 and "How Anomaly detection works" in all_text(at)
    assert any(b.label == "Record decision" for b in at.button), [b.label for b in at.button]
