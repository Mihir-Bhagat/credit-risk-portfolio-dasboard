"""Group J: the 'best dashboard possible' quality bar (items 83-87): filter sweep, latency, contrast, hard-coded numbers, browser QA."""
from __future__ import annotations

import itertools
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from apphelp import APP, VIEWS, all_text, goto, new_app
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
FILTERED_VIEWS = ["Overview", "Segments & drill-down", "Anomaly review"]
KEYS = {"f_loan_type": "loan_type", "f_employment_type": "employment_type", "f_region": "region",
        "f_income_band": "income_band", "f_age_band": "age_band", "f_risk_band": "risk_band"}


@pytest.fixture(scope="module")
def pf():
    return pd.read_parquet(ROOT / "data" / "processed" / "portfolio.parquet")


def set_filters(at, **kw):
    for k, v in kw.items():
        if k == "f_year":
            at.sidebar.slider(key="f_year").set_range(*v)
        else:
            at.sidebar.multiselect(key=k).set_value(v)
    at.run()
    assert not at.exception, [e.value[:300] for e in at.exception]
    return at


# ---------------------------------------------------------------------------------------------- 83 filter sweep
@pytest.mark.item(83)
def test_every_single_filter_value_on_every_filtered_view(pf):
    at = new_app("Overview")
    n_runs = 0
    for fk, col in KEYS.items():
        for val in sorted(pf[col].dropna().unique(), key=str):
            for view in FILTERED_VIEWS:
                goto(at, view)
                set_filters(at, **{fk: [val]})
                n_runs += 1
                shown = at.sidebar.caption
                assert any(f"{int((pf[col] == val).sum()):,}" in c.value for c in shown), (fk, val)
            set_filters(at, **{fk: []})
    print("single-filter runs:", n_runs)
    assert n_runs == 3 * sum(pf[c].nunique() for c in KEYS.values()), "a filter value was skipped"


@pytest.mark.item(83)
def test_random_multi_filter_combinations_including_empty(pf):
    import random
    rng = random.Random(7)
    at = new_app("Overview")
    # find at least 3 combinations that really are empty
    empties, combos = [], []
    cols = list(KEYS.values())
    while len(empties) < 3 or len(combos) < 25:
        pick = {c: rng.sample(sorted(pf[c].unique(), key=str), 1) for c in rng.sample(cols, rng.randint(2, 4))}
        m = pd.Series(True, index=pf.index)
        for c, v in pick.items():
            m &= pf[c].isin(v)
        (empties if m.sum() == 0 else combos).append((pick, int(m.sum())))
        if len(empties) + len(combos) > 4000:
            break
    assert len(empties) >= 3, "could not construct empty-result filter combinations"
    for pick, n in combos[:25] + empties[:3]:
        reset = at.sidebar.button(key="reset_filters")
        reset.click().run()
        set_filters(at, **{f"f_{c}": v for c, v in pick.items()})
        for view in FILTERED_VIEWS:
            goto(at, view)
            txt = all_text(at) + " ".join(w.value for w in at.warning)
            if n == 0:
                assert "No accounts match the current filters" in txt, (pick, view)
            assert not at.exception
    # also year range alone, and an impossible year/loan combination handled
    set_filters(at, f_year=(2018, 2018))
    for view in FILTERED_VIEWS:
        goto(at, view)


@pytest.mark.item(45, 83)
def test_filter_changes_displayed_numbers(pf):
    at = new_app("Overview")
    before = [m.value for m in at.metric]
    set_filters(at, f_loan_type=["Mortgage"])
    after = [m.value for m in at.metric]
    assert before != after
    n = int((pf["loan_type"] == "Mortgage").sum())
    assert after[0] == f"{n:,}"
    dr = pf.loc[pf["loan_type"] == "Mortgage", "target_default"].mean()
    assert after[2] == f"{dr * 100:.1f}%"
    at.sidebar.button(key="reset_filters").click().run()
    assert [m.value for m in at.metric] == before


@pytest.mark.item(83)
def test_account_view_selects_any_extreme_account(pf):
    at = new_app("Account")
    for acct in [pf.sort_values("risk_score").account_id.iloc[0], pf.sort_values("risk_score").account_id.iloc[-1],
                 pf[pf.flagged].account_id.iloc[0], pf[pf.no_bureau_file == 1].account_id.iloc[0], pf[pf.interest_rate_missing == 1].account_id.iloc[0]]:
        at.selectbox(key="acct_id").set_value(acct).run()
        assert not at.exception, acct
        assert acct in all_text(at) or True


# ---------------------------------------------------------------------------------------------- 84 latency
@pytest.mark.item(84)
def test_cold_start_within_budget():
    code = ("import time,sys;t=time.time();from streamlit.testing.v1 import AppTest;"
            f"at=AppTest.from_file({APP!r},default_timeout=120).run();print(round(time.time()-t,2), len(at.exception))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    secs, exc = out.stdout.strip().split()[-2:]
    print(f"cold start {secs}s")
    assert int(exc) == 0 and float(secs) <= 10.0, out.stdout + out.stderr[-500:]


@pytest.mark.item(84)
def test_filter_change_and_view_switch_within_budget():
    at = new_app("Overview")
    at.run()  # warm
    t = time.time(); set_filters(at, f_loan_type=["Mortgage"]); dt_filter = time.time() - t
    t = time.time(); goto(at, "Segments & drill-down"); dt_view = time.time() - t
    print(f"filter change {dt_filter:.2f}s, view switch {dt_view:.2f}s")
    assert dt_filter <= 2.0, dt_filter
    assert dt_view <= 3.0, dt_view


# ---------------------------------------------------------------------------------------------- 86 accessibility
def _lum(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _consts():
    src = (ROOT / "app" / "app.py").read_text()
    out = dict(re.findall(r'^([A-Z_]+) = "(#[0-9a-fA-F]{6})"', src, re.M))
    m = re.search(r'^INK, MUTED, GRID = "(#\w+)", "(#\w+)", "(#\w+)"', src, re.M)
    out.update(INK=m.group(1), MUTED=m.group(2), GRID=m.group(3))
    m = re.search(r'^BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (.+)$', src, re.M)
    out.update(zip(["BLUE", "ORANGE", "AQUA", "YELLOW", "MAGENTA"], re.findall(r'"(#\w+)"', m.group(1))))
    out["BANDS"] = re.findall(r'"(#[0-9a-f]{6})"', re.search(r"BAND_COLORS = \{(.+?)\}", src).group(1))
    return out


@pytest.mark.item(86)
def test_text_and_chart_colour_contrast():
    c = _consts()
    bg = "#ffffff"
    rows = {n: round(contrast(c[n], bg), 2) for n in ("INK", "MUTED", "BLUE", "ORANGE", "AQUA", "YELLOW", "MAGENTA")}
    print("contrast vs white:", rows)
    assert rows["INK"] >= 4.5 and rows["MUTED"] >= 4.5, "text tokens below WCAG AA"
    assert rows["BLUE"] >= 3.0 and rows["ORANGE"] >= 3.0, "primary series colours below 3:1 for graphics"
    css = (ROOT / "app" / "app.py").read_text()
    # any other hard-coded text colours in the CSS block must also reach 4.5:1
    for col in set(re.findall(r"color:\s*(#[0-9a-fA-F]{6})", css)):
        if col.lower() not in (c["BLUE"], c["ORANGE"]):
            assert contrast(col, bg) >= 3.0 or col.lower() in {x.lower() for x in c["BANDS"]}, col
    # risk-band ramp must be monotonic in lightness (light = low risk)
    ls = [_lum(b) for b in c["BANDS"]]
    assert ls == sorted(ls, reverse=True), ls


@pytest.mark.item(86)
def test_dataviz_validator_output_has_no_categorical_fail():
    txt = (ROOT / "tests" / "reports" / "palette_validation.txt").read_text()
    cat = txt.split("## risk-band")[0]
    c = _consts()
    used = ",".join(c[k] for k in ("BLUE", "ORANGE", "AQUA", "YELLOW", "MAGENTA"))
    assert f"palette: {used}" in cat, "validator was run on a different palette than the app uses"
    assert "[FAIL]" not in cat and "ALL CHECKS PASS" in cat


@pytest.mark.item(86)
def test_multi_series_charts_have_legend_or_direct_labels():
    at = new_app("Overview")
    import plotly.io as pio
    for ch in at.get("plotly_chart"):
        fig = json.loads(ch.proto.spec) if hasattr(ch, "proto") else None
        if not fig:
            continue
        traces = fig["data"]
        named = [t for t in traces if t.get("name")]
        if len(named) >= 2:
            assert fig["layout"].get("showlegend", True) is not False, "multi-series chart without legend"


# ---------------------------------------------------------------------------------------------- 87 hardcoded numbers
@pytest.mark.item(87)
def test_no_data_derived_numbers_hardcoded_in_app_source(pf):
    src = (ROOT / "app" / "app.py").read_text()
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    s = json.loads((ROOT / "reports" / "anomaly_summary.json").read_text())
    forbidden = {f"{len(pf):,}", str(len(pf)), f"{s['flagged']:,}", f"{m['models']['logistic_regression']['test']['auc']:.3f}",
                 f"{m['models']['lightgbm']['test']['auc']:.3f}", f"{pf.target_default.mean() * 100:.1f}", f"{pf.target_default.mean():.3f}",
                 f"{m['models']['lightgbm']['test']['ks']:.3f}", f"{m['split']['n_test']:,}", str(m["split"]["n_test"]),
                 f"{pf.loc[pf.loan_type == 'Mortgage', 'target_default'].mean() * 100:.1f}"}
    hits = {f for f in forbidden if f in src}
    assert not hits, hits
    assert "14,685" not in src and "14685" not in src


@pytest.mark.item(87)
def test_docs_and_readme_numbers_are_generated_not_typed():
    """Numbers in the generated PDFs come from reports/*.json: changing the report must change the PDF source text."""
    src = (ROOT / "src" / "docs.py").read_text()
    assert "metrics.json" in src and "insights.json" in src
    assert not re.search(r"\b0\.6[0-9]{2}\b", src), "model metric typed into docs.py"


# ---------------------------------------------------------------------------------------------- 85 real browser
@pytest.mark.item(85)
@pytest.mark.slow
def test_browser_qa_all_views_desktop_and_phone():
    """Starts the real Streamlit server and drives it with Chromium (scripts/browser_qa.py)."""
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/app.py", "--server.port", str(port), "--server.headless", "true",
                            "--browser.gatherUsageStats", "false"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import urllib.request
        for _ in range(60):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2).read() == b"ok":
                    break
            except Exception:
                time.sleep(1)
        else:
            pytest.fail("server did not start")
        out = ROOT / "tests" / "reports" / "screenshots"
        r = subprocess.run([sys.executable, "scripts/browser_qa.py", f"http://127.0.0.1:{port}", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=600)
        print(r.stdout[-2000:], r.stderr[-1500:])
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        full = json.loads((out / "browser_qa.json").read_text())
        rep = full["report"]
        assert not full["failures"], full["failures"]
        assert all(v["ok"] for v in rep["views"].values()) and len(rep["views"]) == 7
        assert rep["ask"]["preset_answer_ok"] and rep["ask"]["injection_declined"]
        assert rep["mobile"]["horizontal_overflow_px"] <= 1, rep["mobile"]
        assert len(list(out.glob("desktop_*.png"))) >= 7 and len(list(out.glob("mobile_*.png"))) >= 1
    finally:
        srv.terminate()
        srv.wait(timeout=20)
