"""D. Default model (checklist items 33-42)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]


def auc_rank(y, p):
    """Mann-Whitney AUC with tie handling, written independently of sklearn."""
    y, p = np.asarray(y), np.asarray(p)
    ranks = pd.Series(p).rank(method="average").to_numpy()
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


@pytest.fixture(scope="module")
def tp():
    return pd.read_csv(ROOT / "reports" / "test_predictions.csv")


@pytest.mark.item(33)
def test_features_engineered_from_bureau_and_repayment():
    cat = pd.read_csv(ROOT / "reports" / "feature_catalog.csv")
    assert (cat["source"] == "repayment_history").sum() >= 8 and (cat["source"] == "bureau_credit").sum() >= 8
    rep = cat[cat.source == "repayment_history"]["feature"]
    assert rep.str.startswith("rep_").all()  # engineered aggregates, not raw columns


@pytest.mark.item(34, 35)
def test_both_models_trained_and_reported(metrics):
    for k in ("logistic_regression", "lightgbm"):
        t = metrics["models"][k]["test"]
        assert all(0 < t[m] < 1 for m in ("auc", "ks", "gini")), k
        assert metrics["models"][k]["best_params"]
    assert (ROOT / "models" / "logistic_regression.joblib").exists() and (ROOT / "models" / "lightgbm.joblib").exists()


@pytest.mark.item(36)
def test_test_set_is_customer_disjoint_and_never_used_for_tuning(raw, portfolio, metrics, monkeypatch):
    sp = metrics["split"]
    assert sp["customer_overlap"] == 0
    tr, te = portfolio[portfolio.split == "train"], portfolio[portfolio.split == "test"]
    assert set(tr.customer_id).isdisjoint(set(te.customer_id))
    assert len(tr) == sp["n_train"] and len(te) == sp["n_test"] and abs(len(te) / len(portfolio) - 0.2) < 0.02
    # Spy on the real tuning function: every row it sees must be a training row.
    from src import model
    seen = []
    real = model.tune

    def spy(builder, grid, X, y, groups, folds):
        seen.append(set(np.unique(groups)))
        return real(builder, grid, X, y, groups, folds)

    monkeypatch.setattr(model, "tune", spy)
    df = pd.read_parquet(ROOT / "data" / "processed" / "analysis_dataset.parquet")
    model.train_and_score(df, n_boot=20, sensitivity=False)
    assert len(seen) == 2
    test_customers = set(te.customer_id)
    for s in seen:
        assert s.isdisjoint(test_customers), "tuning saw test-set customers"
    # the champion is picked by CV on train, never by test AUC
    assert metrics["champion_rule"].startswith("higher mean GroupKFold AUC on training")
    m = metrics["models"]
    assert metrics["champion"] == ("lightgbm" if m["lightgbm"]["cv_auc"] >= m["logistic_regression"]["cv_auc"] else "logistic_regression")


@pytest.mark.item(36, 37)
def test_reported_metrics_recomputed_independently_and_consistent(metrics, tp):
    for key, col in (("logistic_regression", "p_logistic_regression"), ("lightgbm", "p_lightgbm")):
        t = metrics["models"][key]["test"]
        auc = auc_rank(tp["y"], tp[col])
        assert abs(auc - t["auc"]) < 1e-9, f"{key}: AUC mismatch"
        assert abs(t["gini"] - (2 * auc - 1)) < 1e-9, "Gini != 2*AUC-1"
        ks = ks_2samp(tp.loc[tp.y == 1, col], tp.loc[tp.y == 0, col]).statistic
        assert abs(t["ks"] - ks) < 1e-9, f"{key}: KS mismatch"
        lo, hi = t["auc_ci95"]
        assert lo < t["auc"] < hi
    assert len(tp) == metrics["split"]["n_test"] and abs(tp.y.mean() - metrics["split"]["test_default_rate"]) < 1e-12


@pytest.mark.item(38)
def test_training_twice_with_same_seed_gives_identical_metrics():
    from src import model
    df = pd.read_parquet(ROOT / "data" / "processed" / "analysis_dataset.parquet")
    runs = [model.train_and_score(df, seed=42, n_boot=40, sensitivity=False) for _ in range(2)]
    a, b = runs[0].metrics, runs[1].metrics
    for k in ("logistic_regression", "lightgbm"):
        for m in ("auc", "ks", "gini", "pr_auc", "brier"):
            assert round(a["models"][k]["test"][m], 9) == round(b["models"][k]["test"][m], 9), (k, m)
        assert a["models"][k]["cv_auc"] == b["models"][k]["cv_auc"]
    pd.testing.assert_frame_equal(runs[0].scores, runs[1].scores)
    # and a different seed really changes the split (so the equality above is meaningful)
    other = model.train_and_score(df, seed=7, n_boot=20, sensitivity=False)
    assert other.metrics["models"]["lightgbm"]["test"]["auc"] != a["models"]["lightgbm"]["test"]["auc"]


@pytest.mark.item(39)
def test_model_comparison_is_fair_and_honestly_worded(metrics):
    c = metrics["comparison"]
    lr, gb = (metrics["models"][k]["test"]["auc"] for k in ("logistic_regression", "lightgbm"))
    assert abs(c["auc_diff_lightgbm_minus_lr"] - (gb - lr)) < 0.01  # bootstrap mean ~ point difference
    lo, hi = c["ci95"]
    assert c["significant"] == (not (lo <= 0 <= hi))
    s = c["statement"].lower()
    if not c["significant"]:
        assert "no reliable difference" in s
    elif gb > lr:
        assert "lightgbm is better" in s
    else:
        assert "logistic baseline is better" in s
    # same data, same features for both models
    assert metrics["n_features"] == len(pd.read_csv(ROOT / "reports" / "feature_catalog.csv"))
    assert metrics["split"]["tuning"].startswith("GroupKFold")


@pytest.mark.item(40)
def test_feature_importance_outputs_exist_and_are_sensible():
    imp = pd.read_csv(ROOT / "reports" / "feature_importance.csv")
    assert len(imp) == 49 and imp["lgbm_mean_abs_shap"].notna().all() and (imp["lgbm_mean_abs_shap"] >= 0).all()
    assert imp["lgbm_mean_abs_shap"].iloc[0] >= imp["lgbm_mean_abs_shap"].iloc[-1]
    top = set(imp["feature"].head(8))
    assert top & {"annual_income", "years_employed", "age", "rep_mean_dpd", "bur_mean_dpd", "rep_n_late"}, top


@pytest.mark.item(41)
def test_every_account_has_a_score_and_top_drivers(portfolio, raw):
    assert len(portfolio) == 14685 and set(portfolio.account_id) == set(raw["loans"].account_id)
    assert portfolio["risk_score"].notna().all() and portfolio["risk_score"].between(0, 1).all()
    assert portfolio["risk_band"].isin(["Low", "Medium", "High", "Very high"]).all()
    assert (portfolio["top_drivers"].str.len() > 3).all()
    for s in portfolio["drivers_json"].sample(500, random_state=1):
        d = json.loads(s)
        assert 1 <= len(d) <= 5 and all({"feature", "label", "value", "effect"} <= set(x) for x in d)
    # scores are informative: observed default rises across risk bands
    order = ["Low", "Medium", "High", "Very high"]
    rates = portfolio.groupby("risk_band")["target_default"].mean().reindex(order)
    assert rates.is_monotonic_increasing, rates.to_dict()
    # train scores are out-of-fold: their AUC should be close to (not far above) held-out AUC
    from sklearn.metrics import roc_auc_score
    a_tr = roc_auc_score(portfolio[portfolio.split == "train"].target_default, portfolio[portfolio.split == "train"].risk_score)
    a_te = roc_auc_score(portfolio[portfolio.split == "test"].target_default, portfolio[portfolio.split == "test"].risk_score)
    assert abs(a_tr - a_te) < 0.06, (a_tr, a_te)


@pytest.mark.item(42)
def test_monitoring_psi_works_and_plan_is_documented(root):
    from src.monitoring import MONITORING_PLAN, psi
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(1.0, 1, 5000)) > 0.25
    titles = " ".join(t for t, _ in MONITORING_PLAN).lower()
    for k in ("drift", "discrimination", "calibration", "data quality", "action"):
        assert k in titles
    psi_tab = pd.read_csv(root / "reports" / "monitoring_psi.csv")
    assert (psi_tab["item"] == "risk_score").any() and psi_tab["psi"].notna().all()
