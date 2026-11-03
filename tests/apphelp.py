"""Helpers for Streamlit AppTest based tests."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "app.py")
VIEWS = ["Overview", "Segments & drill-down", "Account", "Ask the Portfolio", "Anomaly review", "Model", "Data & methods"]


def new_app(view: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    if view:
        goto(at, view)
    return at


def goto(at: AppTest, view: str) -> AppTest:
    at.sidebar.radio(key="view").set_value(view).run()
    assert not at.exception, [e.value[:400] for e in at.exception]
    return at


def all_text(at: AppTest) -> str:
    parts = []
    for coll in (at.markdown, at.info, at.success, at.warning, at.error, at.caption, at.text, at.subheader, at.header, at.title, at.json, at.code):
        parts += [str(getattr(e, "value", "")) for e in coll]
    parts += [str(t.value) for t in at.text_input]
    return "\n".join(parts)


def metrics(at: AppTest) -> list[str]:
    return [m.value for m in at.metric]


def n_charts(at: AppTest) -> int:
    return len(at.get("plotly_chart"))
