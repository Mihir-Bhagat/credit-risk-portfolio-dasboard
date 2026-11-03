"""Data access for the dashboard: cached loaders over the artifacts produced by `python -m src.build`."""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from . import config

REQUIRED = ["portfolio.parquet", "repayment_clean.parquet", "bureau_clean.parquet"]


def artifacts_ready() -> bool:
    return all((config.PROCESSED / f).exists() for f in REQUIRED) and (config.REPORTS / "metrics.json").exists()


def ensure_built() -> None:
    if not artifacts_ready():
        from . import build
        build.build(skip_docs=True, quiet=True)


def read_json(name: str):
    return json.loads((config.REPORTS / name).read_text())


@lru_cache(maxsize=1)
def portfolio() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED / "portfolio.parquet")


@lru_cache(maxsize=1)
def repayment() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED / "repayment_clean.parquet")


@lru_cache(maxsize=1)
def bureau() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED / "bureau_clean.parquet")
