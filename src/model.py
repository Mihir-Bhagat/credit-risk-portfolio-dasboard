"""Default model: logistic-regression baseline and a LightGBM challenger.

Protocol (fixed in code, asserted in tests):
  1. Split by customer_id (a customer can hold up to 3 accounts) 80/20 with a fixed seed.
  2. All tuning uses GroupKFold cross-validation on the TRAINING rows only.
  3. The test set is scored exactly once, after both models are fixed.
  4. The champion used for dashboard scores is chosen by mean CV AUC on training data (never by test).
  5. Robustness analyses (ablations, point-in-time bureau) are reported but never used for selection.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp  # noqa: F401  (used by tests; kept for parity checks)
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config, features as F
from .monitoring import psi, psi_status

TARGET = "target_default"
LR_LOG_COLS = ["loan_amount", "annual_income", "bur_total_sanctioned", "bur_total_outstanding"]


# ------------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------------
def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metric_block(y, p) -> dict:
    auc = float(roc_auc_score(y, p))
    return {"auc": auc, "ks": ks_stat(y, p), "gini": 2 * auc - 1,
            "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p))}


def bootstrap_auc(y, p, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_auc_diff(y, p_a, p_b, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Paired bootstrap of AUC(a) - AUC(b) on the same resamples."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ------------------------------------------------------------------------------------------------
# Data prep and estimators
# ------------------------------------------------------------------------------------------------
def prepare_xy(df: pd.DataFrame, feature_cols: list[str] | None = None):
    cols = feature_cols or F.MODEL_FEATURES
    X = df[cols].copy()
    for c in [c for c in cols if c in F.MODEL_CATEGORICAL]:
        X[c] = pd.Categorical(X[c].astype(str), categories=sorted(X[c].astype(str).unique()))
    for c in [c for c in cols if c not in F.MODEL_CATEGORICAL]:
        X[c] = X[c].astype(float)
    return X, df[TARGET].astype(int).to_numpy(), df["customer_id"].to_numpy()


def make_lr(cols: list[str], C: float = 1.0) -> Pipeline:
    cat = [c for c in cols if c in F.MODEL_CATEGORICAL]
    num = [c for c in cols if c not in F.MODEL_CATEGORICAL]
    skew = [c for c in num if c in LR_LOG_COLS]
    rest = [c for c in num if c not in LR_LOG_COLS]
    transformers = []
    if rest:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                               ("sc", StandardScaler())]), rest))
    if skew:
        transformers.append(("skew", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                                ("sc", StandardScaler())]), skew))
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), cat))
    pre = ColumnTransformer(transformers, verbose_feature_names_out=True)
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, max_iter=3000, solver="lbfgs"))])


def make_lgbm(num_leaves: int = 8, n_estimators: int = 200, seed: int = config.SEED) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=0.03,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
        random_state=seed, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1)


LR_GRID = [{"C": c} for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)]
LGBM_GRID = [{"num_leaves": nl, "n_estimators": ne} for nl in (3, 4, 8) for ne in (80, 150, 300)]


def tune(builder, grid, X, y, groups, folds: int):
    """Grid search by mean GroupKFold AUC. Only ever called with training rows."""
    cv = GroupKFold(n_splits=folds)
    best, best_score, table = None, -1.0, []
    for params in grid:
        s = cross_val_score(builder(**params), X, y, groups=groups, cv=cv, scoring="roc_auc", n_jobs=1)
        table.append({**params, "cv_auc": float(s.mean()), "cv_auc_std": float(s.std())})
        if s.mean() > best_score:
            best, best_score = params, float(s.mean())
    return best, best_score, table


# ------------------------------------------------------------------------------------------------
# Drivers (per-account explanations)
# ------------------------------------------------------------------------------------------------
def _base_name(transformed_name: str) -> str:
    """Map a ColumnTransformer output name back to the original model feature."""
    n = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
    if n.startswith("missingindicator_"):
        return n[len("missingindicator_"):]
    for c in F.MODEL_CATEGORICAL:
        if n == c or n.startswith(c + "_"):
            return c
    return n


def lr_contributions(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Z = pre.transform(X)
    names = list(pre.get_feature_names_out())
    contrib = Z * clf.coef_[0]
    df = pd.DataFrame(contrib, columns=[_base_name(n) for n in names], index=X.index)
    return df.T.groupby(level=0).sum().T


def lgbm_contributions(model: lgb.LGBMClassifier, X: pd.DataFrame) -> pd.DataFrame:
    sv = model.predict(X, pred_contrib=True)  # TreeSHAP, last column = base value
    return pd.DataFrame(sv[:, :-1], columns=X.columns, index=X.index)


def short_label(feature: str) -> str:
    if feature in F.NUMERIC_CATALOG:
        return F.NUMERIC_CATALOG[feature][1].split("(")[0].strip()
    return feature.replace("_", " ").capitalize()


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.0f}" if abs(v) >= 1000 else (f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}")
    return str(v)


def top_drivers(contrib: pd.DataFrame, raw_values: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Per account: the k features with the largest absolute effect (log-odds), signed."""
    arr = contrib.to_numpy()
    cols = np.array(contrib.columns)
    order = np.argsort(-np.abs(arr), axis=1)[:, :k]
    rows_json, rows_txt = [], []
    for i in range(len(arr)):
        items = []
        for j in order[i]:
            f = cols[j]
            items.append({"feature": f, "label": short_label(f), "value": _fmt(raw_values.iloc[i][f]),
                          "effect": round(float(arr[i, j]), 4)})
        rows_json.append(json.dumps(items))
        up = [it for it in items if it["effect"] > 0][:3] or items[:1]
        rows_txt.append("; ".join(f"{it['label']} = {it['value']}" for it in up))
    return pd.DataFrame({"drivers_json": rows_json, "top_drivers": rows_txt}, index=contrib.index)


# ------------------------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------------------------
@dataclass
class ModelResult:
    metrics: dict
    scores: pd.DataFrame
    test_predictions: pd.DataFrame
    importance: pd.DataFrame
    psi_table: pd.DataFrame
    tuning: dict
    models: dict = field(default_factory=dict)


def _fit_eval_lgbm(df, cols, params, train_idx, test_idx, seed):
    X, y, _ = prepare_xy(df, cols)
    m = make_lgbm(seed=seed, **params).fit(X.iloc[train_idx], y[train_idx])
    return float(roc_auc_score(y[test_idx], m.predict_proba(X.iloc[test_idx])[:, 1]))


def train_and_score(df: pd.DataFrame, seed: int = config.SEED, folds: int = config.CV_FOLDS,
                    n_boot: int = 500, sensitivity: bool = True, bureau_pit: pd.DataFrame | None = None) -> ModelResult:
    df = df.reset_index(drop=True)
    X, y, groups = prepare_xy(df)
    gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_tr, y_tr, g_tr = X.iloc[train_idx], y[train_idx], groups[train_idx]
    X_te, y_te = X.iloc[test_idx], y[test_idx]
    overlap = len(set(groups[train_idx]) & set(groups[test_idx]))
    assert overlap == 0, "customer leakage across the split"

    # ---- tuning on TRAIN ONLY ---------------------------------------------------------------------------
    lr_params, lr_cv, lr_table = tune(lambda C: make_lr(F.MODEL_FEATURES, C), LR_GRID, X_tr, y_tr, g_tr, folds)
    gb_params, gb_cv, gb_table = tune(lambda **p: make_lgbm(seed=seed, **p), LGBM_GRID, X_tr, y_tr, g_tr, folds)

    # ---- final fit on TRAIN, then ONE evaluation on TEST --------------------------------------------------------
    lr = make_lr(F.MODEL_FEATURES, **lr_params).fit(X_tr, y_tr)
    gb = make_lgbm(seed=seed, **gb_params).fit(X_tr, y_tr)
    p_lr = lr.predict_proba(X_te)[:, 1]
    p_gb = gb.predict_proba(X_te)[:, 1]
    m_lr, m_gb = metric_block(y_te, p_lr), metric_block(y_te, p_gb)
    m_lr["auc_ci95"] = bootstrap_auc(y_te, p_lr, n_boot, seed)
    m_gb["auc_ci95"] = bootstrap_auc(y_te, p_gb, n_boot, seed)
    diff, lo, hi = bootstrap_auc_diff(y_te, p_gb, p_lr, n_boot, seed)
    significant = not (lo <= 0 <= hi)
    champion = "lightgbm" if gb_cv >= lr_cv else "logistic_regression"
    if significant and diff > 0:
        stmt = f"LightGBM is better than the logistic baseline on the test set by {diff:.3f} AUC (95% paired-bootstrap CI {lo:.3f} to {hi:.3f})."
    elif significant and diff < 0:
        stmt = f"The logistic baseline is better than LightGBM on the test set by {-diff:.3f} AUC (95% paired-bootstrap CI {lo:.3f} to {hi:.3f})."
    else:
        stmt = (f"No reliable difference between the models: LightGBM minus logistic AUC = {diff:+.3f} "
                f"(95% paired-bootstrap CI {lo:.3f} to {hi:.3f} includes 0). The simpler, more explainable model is a defensible choice.")

    # ---- scores for every account ----------------------------------------------------------------------------------
    champ_builder = (lambda: make_lgbm(seed=seed, **gb_params)) if champion == "lightgbm" else (lambda: make_lr(F.MODEL_FEATURES, **lr_params))
    final = gb if champion == "lightgbm" else lr
    oof = cross_val_predict(clone(champ_builder()), X_tr, y_tr, groups=g_tr, cv=GroupKFold(n_splits=folds), method="predict_proba")[:, 1]
    score = np.empty(len(df))
    score[train_idx] = oof                                   # out-of-fold for training accounts
    score[test_idx] = p_gb if champion == "lightgbm" else p_lr  # held-out for test accounts
    split = np.array(["train"] * len(df), dtype=object)
    split[test_idx] = "test"
    cuts = np.quantile(oof, config.RISK_BAND_QUANTILES[1:-1])
    band = pd.cut(score, [-np.inf, *cuts, np.inf], labels=config.RISK_BAND_LABELS).astype(str)

    contrib = lgbm_contributions(gb, X) if champion == "lightgbm" else lr_contributions(lr, X)
    drivers = top_drivers(contrib, df[F.MODEL_FEATURES])
    scores = pd.DataFrame({"account_id": df["account_id"], "risk_score": score, "risk_band": band, "split": split,
                           "model_used": champion}).join(drivers.reset_index(drop=True))

    # ---- feature importance ------------------------------------------------------------------------------------------------
    shap = lgbm_contributions(gb, X_te)
    lrc = lr_contributions(lr, X_te)
    imp = pd.DataFrame({"feature": F.MODEL_FEATURES})
    imp["label"] = imp["feature"].map(short_label)
    imp["lgbm_mean_abs_shap"] = imp["feature"].map(shap.abs().mean())
    imp["lgbm_gain"] = imp["feature"].map(pd.Series(gb.booster_.feature_importance("gain"), index=gb.booster_.feature_name()))
    imp["lr_mean_abs_contribution"] = imp["feature"].map(lrc.abs().mean())
    imp = imp.sort_values("lgbm_mean_abs_shap", ascending=False).reset_index(drop=True)

    # ---- monitoring demo: PSI between train (out-of-fold) and test -------------------------------------------------------------------
    psi_rows = [{"item": "risk_score", "psi": psi(oof, score[test_idx])}]
    for f in imp["feature"].head(10):
        if f in F.MODEL_CATEGORICAL:
            continue
        psi_rows.append({"item": f, "psi": psi(X_tr[f], X_te[f])})
    psi_df = pd.DataFrame(psi_rows)
    psi_df["status"] = psi_df["psi"].map(psi_status)

    # ---- robustness analyses (reported only; never used to choose anything) ----------------------------------------------------------------
    sens = {}
    if sensitivity:
        base = [c for c in F.MODEL_FEATURES if not c.startswith(("rep_", "bur_")) and c not in ("no_bureau_file",)]
        plus_rep = base + [c for c in F.MODEL_FEATURES if c.startswith("rep_")]
        sens["lgbm_test_auc_profile_only"] = _fit_eval_lgbm(df, base, gb_params, train_idx, test_idx, seed)
        sens["lgbm_test_auc_profile_plus_repayment"] = _fit_eval_lgbm(df, plus_rep, gb_params, train_idx, test_idx, seed)
        sens["lgbm_test_auc_all_features"] = m_gb["auc"]
        if bureau_pit is not None:
            d2 = df.drop(columns=[c for c in bureau_pit.columns if c != "account_id" and c in df.columns]).merge(
                bureau_pit, on="account_id", how="left")
            for col in ["bur_n_lines", "bur_n_active", "bur_n_written_off", "bur_total_sanctioned", "bur_total_outstanding",
                        "bur_n_dpd30", "bur_dpd_unreported"]:
                d2[col] = d2[col].fillna(0)
            d2["no_bureau_file"] = (d2["bur_n_lines"] == 0).astype(int)
            sens["lgbm_test_auc_bureau_point_in_time"] = _fit_eval_lgbm(d2, F.MODEL_FEATURES, gb_params, train_idx, test_idx, seed)
        single = {}
        for f in F.MODEL_NUMERIC:
            v = X[f].fillna(X[f].median())
            a = roc_auc_score(y, v)
            single[f] = float(max(a, 1 - a))
        sens["max_single_feature_auc"] = max(single.values())
        sens["max_single_feature_name"] = max(single, key=single.get)

    metrics = {
        "seed": seed,
        "split": {"method": "GroupShuffleSplit by customer_id", "test_size": config.TEST_SIZE,
                  "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                  "train_default_rate": float(y_tr.mean()), "test_default_rate": float(y_te.mean()),
                  "customer_overlap": int(overlap), "tuning_rows": int(len(train_idx)),
                  "tuning": f"GroupKFold({folds}) on training rows only"},
        "models": {
            "logistic_regression": {"best_params": lr_params, "cv_auc": lr_cv, "test": m_lr},
            "lightgbm": {"best_params": gb_params, "cv_auc": gb_cv, "test": m_gb},
        },
        "comparison": {"auc_diff_lightgbm_minus_lr": diff, "ci95": [lo, hi], "significant": bool(significant), "statement": stmt},
        "champion": champion,
        "champion_rule": "higher mean GroupKFold AUC on training rows",
        "risk_band_thresholds": {"quantiles": config.RISK_BAND_QUANTILES[1:-1], "cutoffs": [float(c) for c in cuts]},
        "n_features": len(F.MODEL_FEATURES),
        "psi_score_train_vs_test": float(psi_df.loc[psi_df["item"] == "risk_score", "psi"].iloc[0]),
        "sensitivity": sens,
    }
    test_pred = pd.DataFrame({"account_id": df["account_id"].iloc[test_idx].to_numpy(), "y": y_te,
                              "p_logistic_regression": p_lr, "p_lightgbm": p_gb})
    return ModelResult(metrics, scores, test_pred, imp, psi_df,
                       {"logistic_regression": lr_table, "lightgbm": gb_table}, {"logistic_regression": lr, "lightgbm": gb})


def save_result(res: ModelResult, reports=config.REPORTS, processed=config.PROCESSED, models=config.MODELS) -> None:
    for d in (reports, processed, models):
        d.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(res.metrics, indent=2))
    res.importance.to_csv(reports / "feature_importance.csv", index=False)
    res.test_predictions.to_csv(reports / "test_predictions.csv", index=False)
    res.psi_table.to_csv(reports / "monitoring_psi.csv", index=False)
    (reports / "tuning_grid.json").write_text(json.dumps(res.tuning, indent=2))
    res.scores.to_parquet(processed / "scores.parquet", index=False)
    for name, m in res.models.items():
        joblib.dump(m, models / f"{name}.joblib")
