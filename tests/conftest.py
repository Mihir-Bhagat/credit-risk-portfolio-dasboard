"""Shared fixtures and the plugin that maps every test to the checklist item(s) it verifies."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
# Decisions go to a throw-away database during tests (never the real outputs/ folder).
os.environ["REVIEW_DB"] = str(Path(tempfile.mkdtemp(prefix="verify_db_")) / "decisions.db")
for _k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "LLM_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL"):
    os.environ.pop(_k, None)  # tests control the key explicitly; default = no key

RESULTS_FILE = Path(os.environ.get("VERIFY_RESULTS", ROOT / "tests" / ".verify_results.json"))
_ITEMS: dict[str, list[int]] = {}
_OUT: dict[str, dict] = {}


def pytest_configure(config):
    config.addinivalue_line("markers", "item(*ids): checklist item numbers this test verifies")


def pytest_collection_modifyitems(items):
    for it in items:
        m = it.get_closest_marker("item")
        _ITEMS[it.nodeid] = list(m.args) if m else []


def pytest_runtest_logreport(report):
    if report.when == "call":
        _OUT[report.nodeid] = {"items": _ITEMS.get(report.nodeid, []), "outcome": report.outcome, "wasxfail": hasattr(report, "wasxfail")}
    elif report.outcome != "passed":  # setup/teardown error or skip
        _OUT[report.nodeid] = {"items": _ITEMS.get(report.nodeid, []), "outcome": "failed" if report.outcome == "failed" else "skipped", "wasxfail": False}


def pytest_sessionfinish(session, exitstatus):
    RESULTS_FILE.write_text(json.dumps(_OUT, indent=1))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def raw() -> dict[str, pd.DataFrame]:
    """Raw CSVs read independently of src/ingestion.py."""
    r = ROOT / "data" / "raw"
    return {"customers": pd.read_csv(r / "customers.csv"), "loans": pd.read_csv(r / "loan_accounts.csv"),
            "bureau": pd.read_csv(r / "bureau_credit.csv"), "repayment": pd.read_csv(r / "repayment_history.csv"),
            "collections": pd.read_csv(r / "collections_events.csv")}


@pytest.fixture(scope="session")
def portfolio() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "processed" / "portfolio.parquet")


@pytest.fixture(scope="session")
def metrics() -> dict:
    return json.loads((ROOT / "reports" / "metrics.json").read_text())


@pytest.fixture(scope="session")
def insights() -> dict:
    return json.loads((ROOT / "reports" / "insights.json").read_text())
