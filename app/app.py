"""Credit Risk Portfolio Monitor: Streamlit dashboard (entry point).

Run:  streamlit run app/app.py
Views: Overview | Segments & drill-down | Account | Ask the Portfolio | Anomaly review | Model | Data & methods
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from src import ai_notes, analytics as AN, ask as K, config, data, decisions as D, monitoring as MON

st.set_page_config(page_title="Credit Risk Portfolio Monitor", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------------------------------
# Secrets -> environment (the LLM key lives only server-side, never in the page)
# --------------------------------------------------------------------------------------------------
for _k in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "LLM_API_KEY"):
    if not os.environ.get(_k):
        try:
            if _k in st.secrets:
                os.environ[_k] = str(st.secrets[_k])
        except Exception:  # no secrets file / store configured: expected in offline (no-key) mode
            break

# --------------------------------------------------------------------------------------------------
# Design tokens (palette validated with the dataviz skill; categorical order is fixed, never cycled)
# --------------------------------------------------------------------------------------------------
INK, MUTED, GRID = "#14213d", "#5c6577", "#e3e7ee"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
BAND_COLORS = {"Low": "#9ec5f2", "Medium": "#4f93e0", "High": "#1f5fb0", "Very high": "#0f2f5e"}
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"

st.markdown(f"""
<style>
.block-container {{padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}}
h1, h2, h3 {{letter-spacing: -0.01em; color: {INK};}}
.hero {{padding: 4px 0 14px 0;}}
.hero h1 {{font-size: 1.75rem; margin: 0;}}
.hero p {{color: {MUTED}; margin: 2px 0 0 0; font-size: 0.95rem;}}
.pill {{display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600; color: #fff;}}
.card-ok {{border-left: 4px solid {BLUE}; background: #f3f8fe; padding: 14px 16px; border-radius: 6px; margin: 6px 0 10px 0;}}
.card-no {{border-left: 4px solid {ORANGE}; background: #fff6f1; padding: 14px 16px; border-radius: 6px; margin: 6px 0 10px 0;}}
.note {{color: {MUTED}; font-size: .86rem;}}
.reason {{border-left: 3px solid {ORANGE}; padding: 4px 10px; margin: 4px 0; background: #fff9f5; border-radius: 3px; font-size: .92rem;}}
[data-testid="stMetricValue"] {{font-size: 1.55rem;}}
footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


def style(fig: go.Figure, h: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=34, b=8), font=dict(family=FONT, size=12, color=INK),
                      paper_bgcolor="white", plot_bgcolor="white", showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0, font=dict(size=11)),
                      title=dict(font=dict(size=14, color=INK), x=0, xanchor="left"), hoverlabel=dict(font_size=12))
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def money(x: float) -> str:
    return f"${x / 1e6:,.1f}M" if abs(x) >= 1e6 else f"${x:,.0f}"


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Preparing data (the first run builds the model, about a minute)...")
def load() -> dict:
    data.ensure_built()
    rep = config.REPORTS
    return {
        "pf": data.portfolio().copy(),
        "metrics": data.read_json("metrics.json"),
        "insights": data.read_json("insights.json"),
        "dq": data.read_json("dq_log.json"),
        "audit": data.read_json("join_audit.json"),
        "imp": pd.read_csv(rep / "feature_importance.csv"),
        "tp": pd.read_csv(rep / "test_predictions.csv"),
        "psi": pd.read_csv(rep / "monitoring_psi.csv"),
        "catalog": pd.read_csv(rep / "feature_catalog.csv"),
        "anom": data.read_json("anomaly_summary.json"),
    }


@st.cache_data(show_spinner=False)
def load_repayment() -> pd.DataFrame:
    return data.repayment()


@st.cache_data(show_spinner=False)
def load_bureau() -> pd.DataFrame:
    return data.bureau()


@st.cache_resource(show_spinner=False)
def get_catalog_and_offline(_pf_len: int):
    pf = load()["pf"]
    cat = K.build_catalog(pf)
    return cat, K.OfflinePlanner(cat)


@st.cache_resource(show_spinner=False)
def get_llm(provider: str, model: str, base_url: str):
    cfg = K.provider_from_env()
    if cfg is None:
        return None
    cat, _ = get_catalog_and_offline(len(load()["pf"]))
    return K.LLMPlanner(cfg, cat)


def build_id() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=3).stdout.strip() or "n/a"
    except (OSError, subprocess.SubprocessError):
        return "n/a"


D_ = load()
PF: pd.DataFrame = D_["pf"]
N_ALL = len(PF)
BASE_DR = float(PF["target_default"].mean())
BASE_PD = float(PF["risk_score"].mean())

DIM_LABELS = {"loan_type": "Loan type", "employment_type": "Employment type", "region": "Region", "income_band": "Income band",
              "age_band": "Age band", "city_tier": "City tier", "housing_type": "Housing type", "education_level": "Education",
              "disbursal_channel": "Channel", "loan_purpose": "Loan purpose", "occupation_sector": "Occupation sector",
              "risk_band": "Model risk band", "origination_year": "Origination year"}
ORDERS = {"income_band": config.INCOME_LABELS, "age_band": config.AGE_LABELS, "risk_band": config.RISK_BAND_LABELS}


def color_map(dim: str) -> dict:
    vals = list(ORDERS.get(dim) or sorted(PF[dim].astype(str).unique()))
    return {v: CATEGORICAL[i % len(CATEGORICAL)] for i, v in enumerate(vals)}


# --------------------------------------------------------------------------------------------------
# Sidebar: navigation + filters
# --------------------------------------------------------------------------------------------------
VIEWS = ["Overview", "Segments & drill-down", "Account", "Ask the Portfolio", "Anomaly review", "Model", "Data & methods"]
FILTER_KEYS = ["f_loan_type", "f_employment_type", "f_region", "f_income_band", "f_age_band", "f_risk_band", "f_year"]


YEAR_MIN, YEAR_MAX = int(PF["origination_year"].min()), int(PF["origination_year"].max())


def reset_filters():
    for k in FILTER_KEYS:
        st.session_state[k] = (YEAR_MIN, YEAR_MAX) if k == "f_year" else []


if "acct_id" not in st.session_state:  # widget state is dropped when a view is left, so remember the last account separately
    st.session_state["acct_id"] = st.session_state.get("_acct_keep") or str(PF.sort_values(["flagged", "risk_score"], ascending=False)["account_id"].iloc[0])

st.sidebar.markdown("### Credit Risk Monitor")
st.sidebar.caption(f"Synthetic bank portfolio · {N_ALL:,} loan accounts · 5 source tables")
view = st.sidebar.radio("View", VIEWS, key="view", label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("**Filters**")
st.sidebar.caption("Apply to Overview, Segments and Anomaly review.")
fl = {
    "loan_type": st.sidebar.multiselect("Loan type", sorted(PF["loan_type"].unique()), key="f_loan_type"),
    "employment_type": st.sidebar.multiselect("Employment type", sorted(PF["employment_type"].unique()), key="f_employment_type"),
    "region": st.sidebar.multiselect("Region", sorted(PF["region"].unique()), key="f_region"),
    "income_band": st.sidebar.multiselect("Income band", config.INCOME_LABELS, key="f_income_band"),
    "age_band": st.sidebar.multiselect("Age band", config.AGE_LABELS, key="f_age_band"),
    "risk_band": st.sidebar.multiselect("Model risk band", config.RISK_BAND_LABELS, key="f_risk_band"),
}
years = st.sidebar.slider("Origination year", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX), key="f_year")
st.sidebar.button("Reset filters", on_click=reset_filters, key="reset_filters")

FILTERED = PF
for col, sel in fl.items():
    if sel:
        FILTERED = FILTERED[FILTERED[col].isin(sel)]
FILTERED = FILTERED[FILTERED["origination_year"].between(*years)]
FILTERS_ACTIVE = len(FILTERED) != N_ALL
st.sidebar.caption(f"**{len(FILTERED):,}** of {N_ALL:,} accounts selected")
st.sidebar.divider()
st.sidebar.caption(f"Build `{build_id()}` · synthetic data, for demonstration only")


def hero(title: str, sub: str):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def flash():
    msg = st.session_state.pop("_flash", None)
    if msg:
        st.success(msg)


def empty_guard(d: pd.DataFrame) -> bool:
    if len(d) == 0:
        st.warning("No accounts match the current filters. Use **Reset filters** in the sidebar.")
        return True
    return False


def notes_panel(notes: dict, key: str):
    """The four statements the brief requires for each AI feature."""
    with st.container(border=True):
        st.markdown(f"**How {notes['title']} works**")
        st.write(notes["how"])
        c = st.columns(4)
        for col, (label, k) in zip(c, [("Trigger", "trigger"), ("False positive looks like", "false_positive"),
                                       ("False negative looks like", "false_negative"), ("Human in the loop", "human")]):
            col.markdown(f"**{label}**")
            col.write(notes[k])
        if "data_sent" in notes:
            st.caption("**Data sent to the LLM:** " + notes["data_sent"])


# ==================================================================================================
# VIEW: Overview
# ==================================================================================================
def view_overview():
    hero("Portfolio overview", "Risk segments, exposure and how default behaves across dimensions. Use the sidebar filters to focus on a segment.")
    d = FILTERED
    if empty_guard(d):
        return
    n, expo, dr = len(d), float(d["exposure"].sum()), float(d["target_default"].mean())
    apd, nfl = float(d["risk_score"].mean()), int(d["flagged"].sum())
    k = st.columns(5)
    k[0].metric("Accounts", f"{n:,}", f"{n / N_ALL * 100:.1f}% of portfolio" if FILTERS_ACTIVE else None, delta_color="off", border=True)
    k[1].metric("Exposure (original amounts)", money(expo), f"{expo / PF['exposure'].sum() * 100:.1f}% of portfolio" if FILTERS_ACTIVE else None, delta_color="off", border=True)
    k[2].metric("Observed default rate", pct(dr), f"{(dr - BASE_DR) * 100:+.1f} pts vs portfolio" if FILTERS_ACTIVE else None, delta_color="inverse", border=True)
    k[3].metric("Avg predicted default probability", pct(apd), f"{(apd - BASE_PD) * 100:+.1f} pts vs portfolio" if FILTERS_ACTIVE else None, delta_color="inverse", border=True)
    k[4].metric("Flagged for review", f"{nfl:,}", f"{nfl / n * 100:.1f}% of accounts", delta_color="off", border=True)

    st.subheader("Risk segments")
    st.caption("Accounts are grouped by the model's predicted default probability. Bands are cut at the 50th, 80th and 95th percentile of training scores.")
    rb = d.groupby("risk_band").agg(accounts=("account_id", "size"), exposure=("exposure", "sum"), observed=("target_default", "mean"),
                                    predicted=("risk_score", "mean")).reindex(config.RISK_BAND_LABELS).dropna(subset=["accounts"]).reset_index()
    c1, c2 = st.columns(2)
    f1 = go.Figure(go.Bar(x=rb["risk_band"], y=rb["exposure"] / 1e6, marker_color=[BAND_COLORS[b] for b in rb["risk_band"]],
                          text=[f"{a:,} accts" for a in rb["accounts"]], textposition="outside", cliponaxis=False,
                          hovertemplate="%{x}<br>$%{y:,.1f}M exposure<extra></extra>"))
    f1.update_yaxes(title="Exposure ($M)")
    c1.plotly_chart(style(f1.update_layout(title="Exposure by model risk band"), legend=False), width="stretch")
    f2 = go.Figure()
    f2.add_bar(x=rb["risk_band"], y=rb["observed"] * 100, name="Observed default rate", marker_color=BLUE, text=[pct(v) for v in rb["observed"]], textposition="outside", cliponaxis=False)
    f2.add_bar(x=rb["risk_band"], y=rb["predicted"] * 100, name="Model-predicted", marker_color=ORANGE, text=[pct(v) for v in rb["predicted"]], textposition="outside", cliponaxis=False)
    f2.update_layout(barmode="group", title="Does the score separate risk? Observed vs predicted default rate")
    f2.update_yaxes(title="Default rate (%)")
    c2.plotly_chart(style(f2), width="stretch")

    st.subheader("Trends across dimensions")
    by_opts = {"All accounts": None, "Loan type": "loan_type", "Employment type": "employment_type", "Region": "region",
               "Income band": "income_band", "Age band": "age_band"}
    by_label = st.selectbox("Break down by", list(by_opts), key="trend_by")
    by = by_opts[by_label]
    t = AN.trend_table(d, by=by)
    cmap = color_map(by) if by else {}
    c1, c2 = st.columns(2)
    f3, f4 = go.Figure(), go.Figure()
    if by:
        for v in (ORDERS.get(by) or sorted(t[by].astype(str).unique())):
            s = t[t[by].astype(str) == v]
            if s.empty:
                continue
            f3.add_scatter(x=s["origination_year"], y=s["default_rate"] * 100, mode="lines+markers", name=v, line=dict(color=cmap[v], width=2), marker=dict(size=8))
            f4.add_bar(x=s["origination_year"], y=s["exposure"] / 1e6, name=v, marker_color=cmap[v])
        f4.update_layout(barmode="stack")
    else:
        f3.add_scatter(x=t["origination_year"], y=t["default_rate"] * 100, mode="lines+markers", name="All accounts", line=dict(color=BLUE, width=2), marker=dict(size=8))
        f4.add_bar(x=t["origination_year"], y=t["exposure"] / 1e6, name="Exposure", marker_color=BLUE)
    f3.update_yaxes(title="Default rate (%)", rangemode="tozero")
    f4.update_yaxes(title="Exposure ($M)")
    for f in (f3, f4):
        f.update_xaxes(dtick=1, title="Origination year")
    c1.plotly_chart(style(f3.update_layout(title="Default rate by origination year"), legend=bool(by)), width="stretch")
    c2.plotly_chart(style(f4.update_layout(title="Exposure by origination year"), legend=bool(by)), width="stretch")

    st.subheader("Profile by dimension")
    dim_opts = {"Income band": "income_band", "Employment type": "employment_type", "Region": "region", "Age band": "age_band",
                "Loan type": "loan_type", "City tier": "city_tier", "Housing type": "housing_type", "Education": "education_level"}
    dim_label = st.selectbox("Dimension", list(dim_opts), key="profile_dim")
    dim = dim_opts[dim_label]
    seg = AN.segment_table(d, dim)
    c1, c2 = st.columns(2)
    f5 = go.Figure(go.Bar(x=seg[dim], y=seg["default_rate"] * 100, marker_color=BLUE, text=[pct(v) for v in seg["default_rate"]], textposition="outside", cliponaxis=False,
                          customdata=np.stack([seg["accounts"], seg["defaults"]], axis=-1),
                          hovertemplate="%{x}<br>%{y:.1f}% default<br>%{customdata[1]:,} of %{customdata[0]:,} accounts<extra></extra>"))
    f5.add_hline(y=dr * 100, line_dash="dash", line_color=MUTED, annotation_text=f"selection {pct(dr)}", annotation_position="top left")
    f5.update_yaxes(title="Default rate (%)", rangemode="tozero")
    c1.plotly_chart(style(f5.update_layout(title=f"Default rate by {dim_label.lower()}"), legend=False), width="stretch")
    f6 = go.Figure()
    f6.add_bar(x=seg[dim], y=seg["share_of_exposure"] * 100, name="Share of exposure", marker_color=BLUE)
    f6.add_bar(x=seg[dim], y=seg["share_of_defaults"] * 100, name="Share of defaults", marker_color=ORANGE)
    f6.update_layout(barmode="group", title=f"Where exposure and defaults sit, by {dim_label.lower()}")
    f6.update_yaxes(title="% of total")
    c2.plotly_chart(style(f6), width="stretch")

    st.subheader("Cross-dimensional cut")
    cc = st.columns(2)
    rd = cc[0].selectbox("Rows", list(dim_opts), index=1, key="cross_row")
    cd = cc[1].selectbox("Columns", list(dim_opts), index=0, key="cross_col")
    if dim_opts[rd] == dim_opts[cd]:
        st.info("Pick two different dimensions.")
    else:
        ct = AN.cross_table(d, dim_opts[rd], dim_opts[cd])
        piv = ct.pivot(index=dim_opts[rd], columns=dim_opts[cd], values="default_rate")
        cnt = ct.pivot(index=dim_opts[rd], columns=dim_opts[cd], values="accounts")
        rows = [r for r in (ORDERS.get(dim_opts[rd]) or piv.index) if r in piv.index]
        cols = [c for c in (ORDERS.get(dim_opts[cd]) or piv.columns) if c in piv.columns]
        piv, cnt = piv.loc[rows, cols], cnt.loc[rows, cols]
        z = piv.where(cnt >= K.MIN_GROUP_SIZE) * 100
        text = [[("" if np.isnan(z.iloc[i, j]) else f"{z.iloc[i, j]:.0f}%<br><span style='font-size:10px'>n={int(cnt.iloc[i, j]):,}</span>") for j in range(z.shape[1])] for i in range(z.shape[0])]
        f7 = go.Figure(go.Heatmap(z=z.values, x=cols, y=rows, text=text, texttemplate="%{text}", colorscale=[[0, "#eaf2fc"], [0.5, "#4f93e0"], [1, "#0f2f5e"]],
                                  colorbar=dict(title="Default %"), hovertemplate="%{y} × %{x}<br>%{z:.1f}% default<extra></extra>"))
        st.plotly_chart(style(f7.update_layout(title=f"Default rate: {rd} × {cd} (cells under {K.MIN_GROUP_SIZE} accounts hidden)"), h=380, legend=False), width="stretch")

    st.subheader("Top insights (whole portfolio)")
    for ins in D_["insights"]["insights"]:
        with st.container(border=True):
            st.markdown(f"**{ins['title']}**")
            st.write(ins["so_what"])
            st.caption("Dimensions: " + ", ".join(ins["dims"]))


# ==================================================================================================
# VIEW: Segments & drill-down
# ==================================================================================================
def goto_account(acct: str):
    st.session_state["acct_id"] = acct
    st.session_state["view"] = "Account"


def view_segments():
    hero("Segments & drill-down", "Pick a dimension, then a segment, to list its accounts ranked by predicted risk.")
    d = FILTERED
    if empty_guard(d):
        return
    dims = {"Loan type": "loan_type", "Employment type": "employment_type", "Region": "region", "Income band": "income_band", "Age band": "age_band",
            "City tier": "city_tier", "Model risk band": "risk_band", "Origination year": "origination_year", "Loan purpose": "loan_purpose"}
    c = st.columns([1, 1])
    dim_label = c[0].selectbox("Segment by", list(dims), key="seg_dim")
    dim = dims[dim_label]
    seg = AN.segment_table(d, dim)
    seg_show = seg.rename(columns={dim: dim_label, "accounts": "Accounts", "defaults": "Defaults", "default_rate": "Default rate",
                                   "exposure": "Exposure", "avg_loan": "Avg loan", "avg_predicted_pd": "Avg predicted PD",
                                   "share_of_exposure": "Share of exposure"})[[dim_label, "Accounts", "Defaults", "Default rate", "Avg predicted PD", "Exposure", "Share of exposure"]]
    st.dataframe(seg_show, hide_index=True, width="stretch",
                 column_config={"Default rate": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=float(max(seg["default_rate"].max(), 0.01))),
                                "Avg predicted PD": st.column_config.NumberColumn(format="percent"),
                                "Exposure": st.column_config.NumberColumn(format="dollar"),
                                "Share of exposure": st.column_config.NumberColumn(format="percent"),
                                "Accounts": st.column_config.NumberColumn(format="localized"), "Defaults": st.column_config.NumberColumn(format="localized")})
    opts = [str(v) for v in seg[dim].tolist()]
    val = c[1].selectbox(f"Drill into {dim_label.lower()}", opts, key="seg_value")
    sub = d[d[dim].astype(str) == val].sort_values("risk_score", ascending=False)
    st.subheader(f"{dim_label}: {val}")
    m = st.columns(4)
    m[0].metric("Accounts", f"{len(sub):,}", border=True)
    m[1].metric("Observed default rate", pct(float(sub["target_default"].mean())), border=True)
    m[2].metric("Avg predicted PD", pct(float(sub["risk_score"].mean())), border=True)
    m[3].metric("Flagged for review", f"{int(sub['flagged'].sum()):,}", border=True)
    show = sub[["account_id", "loan_type", "employment_type", "region", "loan_amount", "risk_score", "risk_band", "flagged", "top_drivers", "target_default"]].rename(
        columns={"account_id": "Account", "loan_type": "Loan type", "employment_type": "Employment", "region": "Region", "loan_amount": "Loan amount",
                 "risk_score": "Predicted PD", "risk_band": "Risk band", "flagged": "Flagged", "top_drivers": "Key drivers", "target_default": "Observed default"})
    st.dataframe(show.head(500), hide_index=True, width="stretch", column_config={
        "Loan amount": st.column_config.NumberColumn(format="dollar"), "Predicted PD": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    st.caption(f"Showing the {min(len(sub), 500):,} highest-risk accounts of {len(sub):,}.")
    cc = st.columns([2, 1, 1])
    pick = cc[0].selectbox("Open an account", sub["account_id"].head(500).tolist(), key="seg_pick")
    cc[1].write("")
    cc[1].button("Open in Account view", on_click=goto_account, args=(pick,), key="seg_open")
    cc[2].write("")
    cc[2].download_button("Download segment CSV", show.to_csv(index=False).encode(), file_name=f"segment_{dim}_{val}.csv", mime="text/csv", key="seg_dl")


# ==================================================================================================
# VIEW: Account
# ==================================================================================================
def decision_form(account_id: str, key: str):
    with st.form(f"form_{key}", clear_on_submit=True):
        decision = st.radio("Decision", D.DECISIONS, horizontal=True, key=f"dec_{key}")
        note = st.text_area("Note (what did you check, why?)", key=f"note_{key}", height=80)
        reviewer = st.text_input("Reviewer", key=f"rev_{key}")
        submitted = st.form_submit_button("Record decision")
    if submitted:
        if len(note.strip()) < 3:
            st.error("Please add a short note so the decision can be audited.")
        else:
            rec = D.record_decision(account_id, decision, note, reviewer)
            st.session_state["_flash"] = f"Recorded **{rec['decision']}** for {rec['account_id']} at {rec['ts_utc']} (UTC)."
            st.rerun()


def view_account():
    hero("Account view", "Risk score, the factors behind it, repayment behaviour and any anomaly flags for a single loan account.")
    flash()
    ids = PF["account_id"].tolist()
    acct = st.selectbox("Account", ids, key="acct_id", help="Type to search, e.g. ACC2007")
    st.session_state["_acct_keep"] = acct
    r = PF[PF["account_id"] == acct].iloc[0]
    c1, c2 = st.columns([1, 2])
    with c1:
        with st.container(border=True):
            st.markdown("**Predicted default probability**")
            st.markdown(f"<div style='font-size:2.6rem;font-weight:700;color:{INK};line-height:1.1'>{pct(r['risk_score'])}</div>", unsafe_allow_html=True)
            st.markdown(f"<span class='pill' style='background:{BAND_COLORS[r['risk_band']]}'>{r['risk_band']} risk</span>", unsafe_allow_html=True)
            rank = float((PF["risk_score"] < r["risk_score"]).mean())
            st.caption(f"Higher than {rank * 100:.0f}% of accounts · portfolio average {pct(BASE_PD)}")
            src = "held-out test split" if r["split"] == "test" else "out-of-fold (cross-validated)"
            st.caption(f"Score source: {src}; model: {r['model_used']}")
            st.markdown(f"**Outcome in data:** {'Defaulted' if r['target_default'] == 1 else 'Performing'}  \n<span class='note'>for validation only, never used as an input</span>", unsafe_allow_html=True)
    with c2:
        items = json.loads(r["drivers_json"])[::-1]
        colors = [ORANGE if it["effect"] > 0 else BLUE for it in items]
        fig = go.Figure(go.Bar(y=[f"{it['label']} = {it['value']}" for it in items], x=[it["effect"] for it in items], orientation="h", marker_color=colors,
                               text=[f"{it['effect']:+.2f}" for it in items], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>effect on log-odds: %{x:+.3f}<extra></extra>"))
        fig.add_vline(x=0, line_color=MUTED)
        fig.update_xaxes(title="Effect on risk vs an average account (log-odds)  ←  lowers risk | raises risk  →")
        st.plotly_chart(style(fig.update_layout(title="Key drivers of this score"), h=300, legend=False), width="stretch")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("**Loan and customer**")
        facts = pd.DataFrame({"Field": ["Loan type", "Loan amount", "Interest rate", "Tenure (months)", "Employment", "Annual income", "Age", "Region", "City tier", "Housing"],
                              "Value": [r["loan_type"], f"${r['loan_amount']:,.0f}", f"{r['interest_rate_pct']:.2f}%", str(int(r["tenure_months"])), r["employment_type"],
                                        f"${r['annual_income']:,.0f}", str(int(r["age"])), r["region"], r["city_tier"], r["housing_type"]]})
        st.dataframe(facts, hide_index=True, width="stretch")
    with f2:
        peers = PF[(PF["loan_type"] == r["loan_type"]) & (PF["income_band"] == r["income_band"])]
        st.markdown("**Peer segment** (same loan type and income band)")
        pm = st.columns(2)
        pm[0].metric("Peer accounts", f"{len(peers):,}", border=True)
        pm[1].metric("Peer default rate", pct(float(peers["target_default"].mean())), border=True)
        st.caption(f"{r['loan_type']} / {r['income_band']}. This account's predicted PD is {pct(r['risk_score'])} vs a peer average of {pct(float(peers['risk_score'].mean()))}.")
    st.markdown("**Repayment behaviour (months 1-12)**")
    rep = load_repayment()
    rr = rep[rep["account_id"] == acct].sort_values("payment_month")
    status_color = {"On-Time": "#b7c4d6", "Late": ORANGE, "Missed": "#7a1f12"}
    g1, g2 = st.columns(2)
    fa = go.Figure()
    fa.add_bar(x=rr["payment_month"], y=rr["emi_due"], name="EMI due", marker_color="#b7c4d6")
    fa.add_bar(x=rr["payment_month"], y=rr["emi_paid"], name="EMI paid", marker_color=BLUE)
    fa.update_layout(barmode="group", title="EMI due vs paid")
    fa.update_xaxes(dtick=1, title="Month since origination")
    g1.plotly_chart(style(fa, h=280), width="stretch")
    fb = go.Figure()
    for stt, col in status_color.items():
        s = rr[rr["payment_status"] == stt]
        fb.add_bar(x=s["payment_month"], y=s["days_past_due"], name=stt, marker_color=col)
    fb.update_layout(title="Days past due by month", barmode="stack")
    fb.update_xaxes(dtick=1, title="Month since origination")
    g2.plotly_chart(style(fb, h=280), width="stretch")
    st.markdown("**External credit lines (bureau)**")
    bur = load_bureau()
    bb = bur[bur["customer_id"] == r["customer_id"]][["credit_type", "lender_category", "sanctioned_amount", "outstanding_amount", "account_status", "days_past_due", "date_opened"]]
    if bb.empty:
        st.info("No bureau file for this customer (thin file).")
    else:
        st.dataframe(bb.rename(columns=lambda c: c.replace("_", " ").capitalize()), hide_index=True, width="stretch")
    st.markdown("**Anomaly review**")
    if r["flagged"]:
        st.warning("This account is flagged for review. Reasons:")
        for s in json.loads(r["signals_json"]):
            st.markdown(f"<div class='reason'><b>{s['signal']}</b> ({s['basis']}): {s['text']}</div>", unsafe_allow_html=True)
        decision_form(acct, "acct")
    else:
        st.success("Not flagged: no strong anomaly against peers or its own history." + (f" Weak signals: {r['reasons']}" if r["n_signals"] else ""))
    hist = D.load_decisions()
    hist = hist[hist["account_id"] == acct]
    if not hist.empty:
        st.markdown("**Decision history for this account**")
        st.dataframe(hist, hide_index=True, width="stretch")


# ==================================================================================================
# VIEW: Ask the Portfolio
# ==================================================================================================
def set_question(q: str):
    st.session_state["ask_q"] = q
    st.session_state["ask_pending"] = q


def view_ask():
    hero("Ask the Portfolio", "Ask a question in plain English. Every figure is computed from the data and shown with where it came from.")
    catalog, offline = get_catalog_and_offline(N_ALL)
    cfg = K.provider_from_env()
    llm = get_llm(cfg.name, cfg.model, cfg.base_url) if cfg else None
    if cfg:
        st.success(f"AI planner on: {cfg.name} · {cfg.model}. If it is unavailable or rate limited the app falls back to the built-in interpreter and says so.")
    else:
        st.info("**No LLM API key configured, so the app is running in offline mode.** Questions are interpreted by a built-in rule-based "
                "interpreter, and the example questions below always work. Add a free key (see README) to switch the AI planner on.")
    with st.form("ask_form"):
        q = st.text_input("Your question", key="ask_q", placeholder="e.g. Which loan type has the highest default rate?")
        submitted = st.form_submit_button("Ask")
    st.caption("Examples (click to run):")
    cols = st.columns(2)
    for i, p in enumerate(K.PRESETS):
        cols[i % 2].button(p, key=f"preset_{i}", on_click=set_question, args=(p,), width="stretch")
    pending = st.session_state.pop("ask_pending", None)
    to_run = pending or (q if submitted else None)
    if to_run:
        ans = K.ask(to_run, PF, catalog, llm=llm, offline=offline)
        st.session_state["ask_result"] = (to_run, ans)
        st.session_state.setdefault("ask_history", []).insert(0, (to_run, ans.status, ans.text))
    res = st.session_state.get("ask_result")
    if res:
        question, ans = res
        st.markdown(f"**Question:** {question}")
        if ans.notice:
            st.info(ans.notice)
        if ans.status == "answered":
            st.markdown(f"<div class='card-ok'>{ans.text}</div>", unsafe_allow_html=True)
            kind = ans.metric_kind
            t = ans.table.copy()
            disp = t[ans.group_by + ["value", "accounts"] + (["defaults"] if "defaults" in t else [])].copy()
            disp["value"] = [K.fmt_value(kind, v) for v in t["value"]]
            disp = disp.rename(columns={"value": K.METRICS[ans.plan["metric"]]["label"], "accounts": "Accounts", "defaults": "Defaults"})
            if len(ans.group_by) == 1 and len(t) > 1:
                fig = go.Figure(go.Bar(x=t[ans.group_by[0]].astype(str), y=t["value"], marker_color=BLUE,
                                       text=[K.fmt_value(kind, v) for v in t["value"]], textposition="outside", cliponaxis=False))
                st.plotly_chart(style(fig.update_layout(title=K.METRICS[ans.plan["metric"]]["label"]), h=300, legend=False), width="stretch")
            st.dataframe(disp, hide_index=True, width="stretch")
            with st.expander("Where this came from"):
                st.write(ans.provenance)
                st.markdown("**How I read your question**")
                st.json(ans.plan)
                st.caption(f"Planner: {'LLM' if ans.mode == 'llm' else 'built-in offline interpreter'}. All numbers above were computed by pandas from the executed result.")
        else:
            st.markdown(f"<div class='card-no'><b>Declined.</b> {ans.text}</div>", unsafe_allow_html=True)
            st.caption(f"Reason category: {ans.category}")
    hist = st.session_state.get("ask_history", [])
    if len(hist) > 1:
        with st.expander("Recent questions"):
            for qh, status, text in hist[1:6]:
                st.markdown(f"- **{qh}** ({status}): {text}")
    notes_panel(ai_notes.ASK_NOTES, "ask")


# ==================================================================================================
# VIEW: Anomaly review
# ==================================================================================================
def view_anomaly():
    hero("Anomaly review", "Accounts whose repayment or bureau pattern is unusual against their peer segment or their own history. Review each one and record a decision.")
    flash()
    d = FILTERED
    if empty_guard(d):
        return
    status = D.latest_status()
    smap = dict(zip(status["account_id"], status["decision"]))
    fl_ = d[d["flagged"]].copy()
    fl_["Status"] = fl_["account_id"].map(smap).map({"Confirm": "Confirmed", "Dismiss": "Dismissed"}).fillna("Pending")
    k = st.columns(4)
    k[0].metric("Flagged accounts", f"{len(fl_):,}", f"{len(fl_) / len(d) * 100:.2f}% of selected accounts", delta_color="off", border=True)
    k[1].metric("Pending review", f"{int((fl_['Status'] == 'Pending').sum()):,}", border=True)
    k[2].metric("Confirmed", f"{int((fl_['Status'] == 'Confirmed').sum()):,}", border=True)
    k[3].metric("Dismissed", f"{int((fl_['Status'] == 'Dismissed').sum()):,}", border=True)
    f = st.columns(3)
    kinds = f[0].multiselect("Signal type", ["repayment", "bureau"], key="an_kinds")
    stat = f[1].multiselect("Status", ["Pending", "Confirmed", "Dismissed"], default=["Pending"], key="an_status")
    min_sig = f[2].slider("Minimum signals", 1, 4, 1, key="an_min")
    view_df = fl_[fl_["n_signals"] >= min_sig]
    if kinds:
        view_df = view_df[view_df["signal_kinds"].apply(lambda s: all(k_ in s.split(",") for k_ in kinds))]
    if stat:
        view_df = view_df[view_df["Status"].isin(stat)]
    view_df = view_df.sort_values("anomaly_score", ascending=False)
    st.markdown(f"**Review list** · {len(view_df):,} accounts")
    if view_df.empty:
        st.info("Nothing to review with these settings.")
    else:
        lst = view_df[["account_id", "anomaly_score", "n_signals", "signal_kinds", "Status", "risk_band", "reasons"]].rename(
            columns={"account_id": "Account", "anomaly_score": "Anomaly score", "n_signals": "Signals", "signal_kinds": "Type", "risk_band": "Risk band", "reasons": "Why flagged"})
        st.dataframe(lst, hide_index=True, width="stretch", height=300, column_config={"Anomaly score": st.column_config.NumberColumn(format="%.2f")})
        pick = st.selectbox("Open an account to review", view_df["account_id"].tolist(), key="an_pick")
        r = PF[PF["account_id"] == pick].iloc[0]
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**Why {pick} was flagged**")
            for s in json.loads(r["signals_json"]):
                st.markdown(f"<div class='reason'><b>{s['signal']}</b> ({s['basis']}, {s['kind']}): {s['text']}</div>", unsafe_allow_html=True)
            rr = load_repayment()
            rr = rr[rr["account_id"] == pick].sort_values("payment_month")
            fig = go.Figure()
            fig.add_bar(x=rr["payment_month"], y=rr["days_past_due"], marker_color=[{"On-Time": "#b7c4d6", "Late": ORANGE, "Missed": "#7a1f12"}[s] for s in rr["payment_status"]],
                        hovertemplate="Month %{x}: %{y} days past due<extra></extra>")
            fig.update_xaxes(dtick=1, title="Month since origination")
            fig.update_yaxes(title="Days past due")
            st.plotly_chart(style(fig.update_layout(title="Days past due by month (grey on-time, orange late, dark missed)"), h=250, legend=False), width="stretch")
            st.caption(f"{r['loan_type']} · {r['employment_type']} · income band {r['income_band']} · predicted PD {pct(r['risk_score'])} ({r['risk_band']}) · peer group {r['peer_group']}")
        with c2:
            st.markdown("**Your decision**")
            decision_form(pick, "an")
    st.markdown("**Decision log**")
    hist = D.load_decisions()
    if hist.empty:
        st.caption("No decisions recorded yet.")
    else:
        st.dataframe(hist, hide_index=True, width="stretch")
        st.download_button("Download decision log (CSV)", hist.to_csv(index=False).encode(), file_name="review_decisions.csv", mime="text/csv", key="an_dl")
    st.caption("Decisions are stored in a SQLite file on the app server. On free hosting that file is reset when the app restarts, so download the log to keep a copy.")
    notes_panel(ai_notes.ANOMALY_NOTES, "an")


# ==================================================================================================
# VIEW: Model
# ==================================================================================================
def view_model():
    hero("Default model", "Logistic-regression baseline versus a gradient-boosted model, evaluated once on a held-out test set split by customer.")
    m = D_["metrics"]
    sp = m["split"]
    st.caption(f"Split by customer: {sp['n_train']:,} training accounts, {sp['n_test']:,} test accounts, {sp['customer_overlap']} customers in both. "
               f"Tuning used {sp['tuning']}. The test set was scored once.")
    rows = []
    for key, name in [("logistic_regression", "Logistic regression (baseline)"), ("lightgbm", "LightGBM (challenger)")]:
        t = m["models"][key]["test"]
        rows.append({"Model": name, "AUC": t["auc"], "AUC 95% CI": f"{t['auc_ci95'][0]:.3f} - {t['auc_ci95'][1]:.3f}", "KS": t["ks"], "Gini": t["gini"],
                     "PR-AUC": t["pr_auc"], "Brier": t["brier"], "CV AUC (train)": m["models"][key]["cv_auc"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["AUC", "KS", "Gini", "PR-AUC", "Brier", "CV AUC (train)"]})
    champ = "LightGBM" if m["champion"] == "lightgbm" else "Logistic regression"
    st.info(f"**{m['comparison']['statement']}** Scores in this dashboard come from **{champ}**, chosen by cross-validated AUC on training data before the test set was touched.")
    tp = D_["tp"]
    c1, c2 = st.columns(2)
    fr = go.Figure()
    for col, name, colr in [("p_logistic_regression", "Logistic regression", ORANGE), ("p_lightgbm", "LightGBM", BLUE)]:
        fpr, tpr, _ = roc_curve(tp["y"], tp[col])
        fr.add_scatter(x=fpr, y=tpr, mode="lines", name=name, line=dict(color=colr, width=2))
    fr.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=MUTED, dash="dash", width=1))
    fr.update_xaxes(title="False positive rate")
    fr.update_yaxes(title="True positive rate")
    c1.plotly_chart(style(fr.update_layout(title="ROC curve (test set)"), h=340), width="stretch")
    pcol = "p_lightgbm" if m["champion"] == "lightgbm" else "p_logistic_regression"
    fk = go.Figure()
    for yv, name, colr in [(0, "Performing", BLUE), (1, "Defaulted", ORANGE)]:
        s = np.sort(tp.loc[tp["y"] == yv, pcol].to_numpy())
        fk.add_scatter(x=s, y=np.arange(1, len(s) + 1) / len(s), mode="lines", name=name, line=dict(color=colr, width=2))
    fk.update_xaxes(title="Predicted default probability")
    fk.update_yaxes(title="Cumulative share")
    c2.plotly_chart(style(fk.update_layout(title=f"KS: gap between the two curves = {m['models'][m['champion']]['test']['ks']:.3f}"), h=340), width="stretch")
    c1, c2 = st.columns(2)
    imp = D_["imp"].head(15).iloc[::-1]
    fi = go.Figure(go.Bar(y=imp["label"], x=imp["lgbm_mean_abs_shap"], orientation="h", marker_color=BLUE, hovertemplate="%{y}<br>mean |SHAP| %{x:.3f}<extra></extra>"))
    fi.update_xaxes(title="Mean |SHAP| (log-odds), test set")
    c1.plotly_chart(style(fi.update_layout(title="Feature importance (LightGBM SHAP)"), h=460, legend=False), width="stretch")
    te = PF[PF["split"] == "test"].copy()
    te["decile"] = pd.qcut(te["risk_score"].rank(method="first"), 10, labels=[str(i) for i in range(1, 11)])
    dc = te.groupby("decile", observed=True).agg(obs=("target_default", "mean"), pred=("risk_score", "mean"), n=("account_id", "size")).reset_index()
    fd = go.Figure()
    fd.add_bar(x=dc["decile"], y=dc["obs"] * 100, name="Observed", marker_color=BLUE)
    fd.add_scatter(x=dc["decile"], y=dc["pred"] * 100, name="Predicted", mode="lines+markers", line=dict(color=ORANGE, width=2), marker=dict(size=8))
    fd.update_xaxes(title="Score decile (1 = lowest risk)")
    fd.update_yaxes(title="Default rate (%)")
    c2.plotly_chart(style(fd.update_layout(title="Calibration by decile (test set)"), h=460), width="stretch")
    st.subheader("How much does each data source add? (robustness, reported not used for selection)")
    s = m["sensitivity"]
    ab = pd.DataFrame({"Feature set": ["Profile + loan only", "+ repayment (months 1-12)", "+ bureau (all features)", "Bureau limited to lines opened by month 12"],
                       "LightGBM test AUC": [s["lgbm_test_auc_profile_only"], s["lgbm_test_auc_profile_plus_repayment"], s["lgbm_test_auc_all_features"], s.get("lgbm_test_auc_bureau_point_in_time", np.nan)]})
    st.dataframe(ab, hide_index=True, width="stretch", column_config={"LightGBM test AUC": st.column_config.NumberColumn(format="%.3f")})
    st.caption(f"Leakage check: the strongest single feature on its own reaches AUC {s['max_single_feature_auc']:.3f} ({s['max_single_feature_name']}); a value above 0.95 would indicate leakage.")
    st.subheader("Monitoring after deployment")
    ps = D_["psi"].copy()
    ps["PSI"] = ps["psi"].round(4)
    st.dataframe(ps[["item", "PSI", "status"]].rename(columns={"item": "Score / feature", "status": "Status"}), hide_index=True, width="stretch")
    st.caption("Demonstration: PSI of the test sample against the training sample. In production the same calculation runs on each new month of accounts.")
    for title, text in MON.MONITORING_PLAN:
        st.markdown(f"- **{title}.** {text}")


# ==================================================================================================
# VIEW: Data & methods
# ==================================================================================================
def view_data():
    hero("Data & methods", "How the five tables were joined, every data-quality decision, and which fields feed the model.")
    a = D_["audit"]
    k = st.columns(4)
    k[0].metric("Accounts in", f"{a['accounts_in']:,}", border=True)
    k[1].metric("Accounts out", f"{a['accounts_out']:,}", border=True)
    k[2].metric("account_id unique", str(a["account_id_unique"]), border=True)
    k[3].metric("Orphan keys", f"{sum(a['orphans'].values())}", border=True)
    st.markdown("**Join logic**")
    for i, s_ in enumerate(a["join_logic"], 1):
        st.markdown(f"{i}. {s_}")
    st.markdown("**Data-quality log**")
    dq = pd.DataFrame(D_["dq"])[["id", "table", "issue", "n_raw", "decision", "handling", "n_unresolved"]]
    st.dataframe(dq.rename(columns={"n_raw": "Found (raw)", "n_unresolved": "Remaining (cleaned)"}), hide_index=True, width="stretch")
    st.markdown("**Model features and their cutoff logic**")
    st.caption("Observation point = end of repayment month 12. Collections data and protected attributes are excluded; see docs/data_quality_log.md.")
    st.dataframe(D_["catalog"], hide_index=True, width="stretch", height=320)
    st.markdown("**Key assumptions**")
    for s_ in ["The target has no date; it is assumed to be observed after month 12.",
               "The bureau extract has no as-of date; it is used as supplied, with a point-in-time sensitivity run on the Model view.",
               "Exposure is the original loan amount (no current balance exists at account level).",
               "All data is synthetic; results are illustrative, not evidence about any real bank."]:
        st.markdown(f"- {s_}")


{"Overview": view_overview, "Segments & drill-down": view_segments, "Account": view_account, "Ask the Portfolio": view_ask,
 "Anomaly review": view_anomaly, "Model": view_model, "Data & methods": view_data}[view]()
