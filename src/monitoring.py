"""Post-deployment monitoring helpers: Population Stability Index and the written monitoring plan."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index of `actual` against `expected` (bins from expected quantiles)."""
    e = pd.Series(expected).dropna().to_numpy(dtype=float)
    a = pd.Series(actual).dropna().to_numpy(dtype=float)
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_status(v: float) -> str:
    if np.isnan(v):
        return "n/a"
    return "stable" if v < 0.10 else ("watch" if v < 0.25 else "act")


MONITORING_PLAN = [
    ("Population drift", "Weekly PSI on the score and on the top-10 features vs the training sample. PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 investigate."),
    ("Discrimination", "Monthly AUC / KS / Gini on matured cohorts (accounts with a known outcome); alert if AUC falls more than 0.03 below the validation value or the paired-bootstrap interval excludes it."),
    ("Calibration", "Compare mean predicted PD with observed default rate by risk band each month; alert if any band is off by more than 20 % relative."),
    ("Data quality", "Daily checks on null rates, new category values, bureau extract freshness and the share of imputed fields (the DQ flags in this repo)."),
    ("Fairness and stability", "Quarterly review of score and approval distribution across age and region; protected attributes stay out of the model."),
    ("Action", "Retrain or recalibrate when two consecutive monthly checks breach; challenger model runs in shadow before replacement; every change is reviewed by model risk."),
]
