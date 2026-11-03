"""Portfolio analytics: segment profiles, cross-cuts, trends, and data-driven insights.

Every number in an insight is computed here from the analysis dataset; the so-what text only
interpolates those numbers. reports/insights.json is read by the dashboard and the PDF generator.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

TARGET = "target_default"
PROFILE_DIMS = {
    "Income band": "income_band",
    "Employment type": "employment_type",
    "Region": "region",
    "Age band": "age_band",
    "Loan type": "loan_type",
}
ORDER = {
    "income_band": config.INCOME_LABELS,
    "age_band": config.AGE_LABELS,
    "risk_band": config.RISK_BAND_LABELS,
}


def segment_table(df: pd.DataFrame, dim: str, score_col: str | None = "risk_score") -> pd.DataFrame:
    g = df.groupby(dim, observed=True)
    out = pd.DataFrame({
        "accounts": g.size(),
        "defaults": g[TARGET].sum(),
        "default_rate": g[TARGET].mean(),
        "exposure": g["exposure"].sum(),
        "avg_loan": g["loan_amount"].mean(),
    })
    if score_col and score_col in df:
        out["avg_predicted_pd"] = g[score_col].mean()
    out["share_of_accounts"] = out["accounts"] / out["accounts"].sum()
    out["share_of_exposure"] = out["exposure"] / out["exposure"].sum()
    out["share_of_defaults"] = out["defaults"] / out["defaults"].sum()
    out = out.reset_index()
    if dim in ORDER:
        out[dim] = pd.Categorical(out[dim], ORDER[dim], ordered=True)
        out = out.sort_values(dim)
        out[dim] = out[dim].astype(str)
    return out.reset_index(drop=True)


def cross_table(df: pd.DataFrame, row: str, col: str) -> pd.DataFrame:
    """Long table: default rate and size for each (row, col) cell."""
    g = df.groupby([row, col], observed=True)
    out = pd.DataFrame({"accounts": g.size(), "default_rate": g[TARGET].mean(), "exposure": g["exposure"].sum()})
    return out.reset_index()


def trend_table(df: pd.DataFrame, by: str | None = None, period: str = "origination_year") -> pd.DataFrame:
    keys = [period] + ([by] if by else [])
    g = df.groupby(keys, observed=True)
    out = pd.DataFrame({"accounts": g.size(), "default_rate": g[TARGET].mean(), "exposure": g["exposure"].sum()})
    return out.reset_index()


def profile_all(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, dim in PROFILE_DIMS.items():
        t = segment_table(df, dim, score_col=None)
        t.insert(0, "dimension", label)
        t = t.rename(columns={dim: "segment"})
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def compute_insights(df: pd.DataFrame, min_cell: int = 100) -> list[dict]:
    base = float(df[TARGET].mean())
    n_total = len(df)
    exp_total = float(df["exposure"].sum())
    ins: list[dict] = []

    # 1. Loan type: highest default rate vs exposure concentration --------------------------------
    lt = segment_table(df, "loan_type", None).sort_values("default_rate", ascending=False)
    top = lt.iloc[0]
    ins.append({
        "id": "I1", "dims": ["loan_type"],
        "title": f"{top['loan_type']}s are the riskiest product and also {_pct(top['share_of_exposure'], 0)} of exposure",
        "evidence": {"segment": top["loan_type"], "default_rate": float(top["default_rate"]), "portfolio_default_rate": base,
                     "share_of_exposure": float(top["share_of_exposure"]), "accounts": int(top["accounts"]),
                     "exposure": float(top["exposure"]),
                     "lowest_segment": lt.iloc[-1]["loan_type"], "lowest_default_rate": float(lt.iloc[-1]["default_rate"])},
        "so_what": (f"{top['loan_type']} default rate is {_pct(top['default_rate'])} vs {_pct(base)} for the portfolio and "
                    f"{_pct(lt.iloc[-1]['default_rate'])} for {lt.iloc[-1]['loan_type']}. Because it carries {_pct(top['share_of_exposure'], 0)} of exposure, "
                    f"a few points of extra default here matter more than any other product. Secured does not mean safe: review underwriting and "
                    f"collateral valuation on this book first."),
    })

    # 2. Income gradient ---------------------------------------------------------------------------------------
    inc = segment_table(df, "income_band", None)
    low, high = inc.iloc[0], inc.iloc[-1]
    low2 = inc[inc["income_band"].isin(["<30k", "30-60k"])]
    share_acc = low2["accounts"].sum() / n_total
    share_def = low2["defaults"].sum() / df[TARGET].sum()
    age = segment_table(df, "age_band", None)
    ins.append({
        "id": "I2", "dims": ["income_band", "age_band"],
        "title": f"Default risk falls steadily with income: {_pct(low['default_rate'], 0)} below 30k vs {_pct(high['default_rate'], 0)} above 120k",
        "evidence": {"low_band": low["income_band"], "low_rate": float(low["default_rate"]), "high_band": high["income_band"],
                     "high_rate": float(high["default_rate"]), "under60k_share_accounts": float(share_acc),
                     "under60k_share_defaults": float(share_def),
                     "youngest_rate": float(age.iloc[0]["default_rate"]), "oldest_rate": float(age.iloc[-1]["default_rate"])},
        "so_what": (f"Customers earning under 60k are {_pct(share_acc, 0)} of accounts but {_pct(share_def, 0)} of defaults. "
                    f"Age follows the same pattern ({_pct(age.iloc[0]['default_rate'])} under 25 vs {_pct(age.iloc[-1]['default_rate'])} at 55+). "
                    f"Affordability checks and lower limits for low-income, young applicants are the cheapest lever."),
    })

    # 3. Employment --------------------------------------------------------------------------------------------------------------
    emp = segment_table(df, "employment_type", None).sort_values("default_rate", ascending=False)
    worst = emp.iloc[0]
    sal = emp[emp["employment_type"] == "Salaried"].iloc[0]
    ins.append({
        "id": "I3", "dims": ["employment_type"],
        "title": f"{worst['employment_type']} customers default at {_pct(worst['default_rate'], 0)}, about {worst['default_rate'] / sal['default_rate']:.1f}x salaried",
        "evidence": {"segment": worst["employment_type"], "default_rate": float(worst["default_rate"]), "accounts": int(worst["accounts"]),
                     "share_of_exposure": float(worst["share_of_exposure"]), "salaried_rate": float(sal["default_rate"]),
                     "second_segment": emp.iloc[1]["employment_type"], "second_rate": float(emp.iloc[1]["default_rate"])},
        "so_what": (f"The segment is only {_pct(worst['share_of_exposure'])} of exposure but is the sharpest single risk marker "
                    f"({int(worst['accounts']):,} accounts). Other employment types sit within a narrow band "
                    f"({_pct(emp.iloc[-1]['default_rate'])} to {_pct(emp.iloc[1]['default_rate'])}), so a decline or manual-review rule for this segment is "
                    f"easy to target without penalising everyone else."),
    })

    # 4. Cross-dimensional cut: loan type x employment -----------------------------------------------------------------------------
    ct = cross_table(df, "loan_type", "employment_type")
    cells = ct[ct["accounts"] >= min_cell].sort_values("default_rate", ascending=False)
    cell = cells.iloc[0]
    mort = ct[(ct["loan_type"] == top["loan_type"]) & (ct["employment_type"] != worst["employment_type"])]
    mort = mort[mort["accounts"] >= min_cell]
    ins.append({
        "id": "I4", "dims": ["loan_type", "employment_type"],
        "title": f"Cross-cut: {cell['employment_type']} {cell['loan_type']} holders default at {_pct(cell['default_rate'], 0)} (n={int(cell['accounts']):,})",
        "evidence": {"loan_type": cell["loan_type"], "employment_type": cell["employment_type"], "default_rate": float(cell["default_rate"]),
                     "accounts": int(cell["accounts"]), "exposure": float(cell["exposure"]), "min_cell_size": min_cell,
                     "top_loan_type_employed_min": float(mort["default_rate"].min()), "top_loan_type_employed_max": float(mort["default_rate"].max())},
        "so_what": (f"The worst cell with at least {min_cell} accounts is {cell['employment_type']} x {cell['loan_type']} at {_pct(cell['default_rate'])} "
                    f"({cell['default_rate'] / base:.1f}x the portfolio rate). "
                    f"Separately, {top['loan_type']} default stays between {_pct(mort['default_rate'].min())} and {_pct(mort['default_rate'].max())} "
                    f"even for employed groups, so its risk is a product effect, not just an unemployment effect. Use the pair, not either dimension alone, for limit setting."),
    })

    # 5. Early-warning from repayment behaviour ------------------------------------------------------------------------------------------
    e3 = df["rep_missed_first3"] >= 1
    r_e3, r_ok = float(df.loc[e3, TARGET].mean()), float(df.loc[~e3, TARGET].mean())
    any_m = df["any_missed"] == 1
    ins.append({
        "id": "I5", "dims": ["repayment"],
        "title": f"A missed payment in the first 3 months lifts default risk from {_pct(r_ok, 0)} to {_pct(r_e3, 0)}",
        "evidence": {"flagged_accounts": int(e3.sum()), "flagged_share": float(e3.mean()), "rate_with_early_miss": r_e3,
                     "rate_without": r_ok, "rate_any_missed_12m": float(df.loc[any_m, TARGET].mean()),
                     "rate_never_missed_12m": float(df.loc[~any_m, TARGET].mean()), "share_any_missed_12m": float(any_m.mean())},
        "so_what": (f"{int(e3.sum()):,} accounts ({_pct(e3.mean())}) missed at least one of their first three payments and default at {_pct(r_e3)} "
                    f"vs {_pct(r_ok)}. Accounts that never miss in 12 months default at {_pct(df.loc[~any_m, TARGET].mean())}. "
                    f"An early-collections trigger at month 3 reaches the riskiest accounts while there is still time to restructure."),
    })
    return ins


def region_spread(df: pd.DataFrame) -> dict:
    r = segment_table(df, "region", None)
    return {"min_region": r.loc[r["default_rate"].idxmin(), "region"], "min_rate": float(r["default_rate"].min()),
            "max_region": r.loc[r["default_rate"].idxmax(), "region"], "max_rate": float(r["default_rate"].max())}


def write_reports(df: pd.DataFrame) -> list[dict]:
    config.REPORTS.mkdir(exist_ok=True)
    profile_all(df).to_csv(config.REPORTS / "segment_profiles.csv", index=False)
    ins = compute_insights(df)
    payload = {"portfolio": {"accounts": int(len(df)), "default_rate": float(df[TARGET].mean()),
                             "exposure": float(df["exposure"].sum()), "region_spread": region_spread(df)},
               "insights": ins}
    (config.REPORTS / "insights.json").write_text(json.dumps(payload, indent=2))
    return ins
