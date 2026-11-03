"""Human-in-the-loop decision store for the anomaly review list (SQLite, stdlib only).

Each decision is an immutable row: timestamp (UTC), account id, decision, note, reviewer.
The latest row per account is that account's current status; earlier rows remain as an audit trail.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

DECISIONS = ("Confirm", "Dismiss")
_SCHEMA = """CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    account_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('Confirm','Dismiss')),
    note TEXT NOT NULL DEFAULT '',
    reviewer TEXT NOT NULL DEFAULT ''
)"""


def _connect(path: Path | str | None) -> sqlite3.Connection:
    p = Path(path) if path else config.REVIEW_DB
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute(_SCHEMA)
    return con


def record_decision(account_id: str, decision: str, note: str = "", reviewer: str = "", path=None) -> dict:
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    if not account_id:
        raise ValueError("account_id is required")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con = _connect(path)
    try:
        with con:
            con.execute("INSERT INTO decisions (ts_utc, account_id, decision, note, reviewer) VALUES (?,?,?,?,?)",
                        (ts, account_id, decision, note.strip(), reviewer.strip()))
    finally:
        con.close()
    return {"ts_utc": ts, "account_id": account_id, "decision": decision, "note": note.strip(), "reviewer": reviewer.strip()}


def load_decisions(path=None) -> pd.DataFrame:
    con = _connect(path)
    try:
        return pd.read_sql_query("SELECT ts_utc, account_id, decision, note, reviewer FROM decisions ORDER BY id DESC", con)
    finally:
        con.close()


def latest_status(path=None) -> pd.DataFrame:
    d = load_decisions(path)
    if d.empty:
        return d.assign(status=[])[["account_id", "decision", "ts_utc", "note", "reviewer"]]
    return d.drop_duplicates("account_id", keep="first")[["account_id", "decision", "ts_utc", "note", "reviewer"]]
