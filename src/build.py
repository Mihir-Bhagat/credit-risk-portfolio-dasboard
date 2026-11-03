"""One command builds everything from the raw CSVs:  python -m src.build

raw CSVs -> cleaning (DQ log) -> account-grain dataset -> analytics/insights -> model + scores -> anomaly flags
         -> data/processed/*.parquet and reports/*  -> docs/*.pdf
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np
import pandas as pd

from . import analytics, anomaly, config, features as F, ingestion, model


def build(skip_docs: bool = False, quiet: bool = False) -> dict:
    t0 = time.time()
    log = (lambda *a: None) if quiet else print
    for d in (config.PROCESSED, config.REPORTS, config.MODELS, config.DOCS):
        d.mkdir(parents=True, exist_ok=True)

    log("[1/6] ingestion: load, clean, document")
    raw = ingestion.load_raw()
    cleaned, issues = ingestion.clean_tables(raw)
    analysis = F.build_analysis_dataset(cleaned)
    audit = ingestion.join_audit(cleaned, analysis)
    assert audit["accounts_in"] == audit["accounts_out"] and audit["account_id_unique"]
    ingestion.write_dq_log(issues, audit)
    analysis.to_parquet(config.PROCESSED / "analysis_dataset.parquet", index=False)
    cleaned["repayment"].to_parquet(config.PROCESSED / "repayment_clean.parquet", index=False)
    cleaned["bureau"].to_parquet(config.PROCESSED / "bureau_clean.parquet", index=False)
    cleaned["collections"].to_parquet(config.PROCESSED / "collections_clean.parquet", index=False)
    F.feature_catalog().to_csv(config.REPORTS / "feature_catalog.csv", index=False)
    (config.REPORTS / "feature_exclusions.json").write_text(json.dumps(F.EXCLUDED, indent=2))

    log("[2/6] analytics and insights")
    analytics.write_reports(analysis)

    log("[3/6] model: logistic baseline + LightGBM (this takes ~30-60 s)")
    pit = F.bureau_features_point_in_time(cleaned["bureau"], cleaned["loans"])
    res = model.train_and_score(analysis, bureau_pit=pit)
    model.save_result(res)

    log("[4/6] anomaly detection")
    anom = anomaly.detect_anomalies(analysis)
    anom.to_parquet(config.PROCESSED / "anomalies.parquet", index=False)

    log("[5/6] portfolio table for the dashboard")
    pf = analysis.merge(res.scores, on="account_id", how="left", validate="one_to_one")
    pf = pf.merge(anom, on="account_id", how="left", validate="one_to_one")
    pf["anomaly_status"] = np.where(pf["flagged"], "Flagged", "Not flagged")
    assert len(pf) == len(analysis) and pf["account_id"].is_unique
    pf.to_parquet(config.PROCESSED / "portfolio.parquet", index=False)

    summary = {
        "accounts": int(len(pf)), "flagged": int(pf["flagged"].sum()), "flag_rate": float(pf["flagged"].mean()),
        "default_rate_flagged": float(pf.loc[pf["flagged"], "target_default"].mean()),
        "default_rate_not_flagged": float(pf.loc[~pf["flagged"], "target_default"].mean()),
        "flagged_by_kind": {k: int(v) for k, v in pf.loc[pf["flagged"], "signal_kinds"].value_counts().items()},
        "params": {k: getattr(config, k) for k in dir(config) if k.startswith("ANOMALY_")},
    }
    (config.REPORTS / "anomaly_summary.json").write_text(json.dumps(summary, indent=2))
    (config.REPORTS / "run_info.json").write_text(json.dumps({
        "python": platform.python_version(), "pandas": pd.__version__, "seconds": round(time.time() - t0, 1)}, indent=2))

    if not skip_docs:
        log("[6/6] documents (PDF)")
        from . import docs
        docs.make_all()
    log(f"done in {time.time() - t0:.0f}s -> data/processed, reports, docs")
    return res.metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-docs", action="store_true")
    ns = ap.parse_args(argv)
    build(skip_docs=ns.skip_docs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
