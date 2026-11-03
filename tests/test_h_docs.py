"""Group H: the two one-page PDFs (items 7, 8, 10, 72-79)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
ARCH, OVER = ROOT / "docs" / "architecture.pdf", ROOT / "docs" / "solution_overview.pdf"


def text(p: Path) -> str:
    return re.sub(r"\s+", " ", PdfReader(str(p)).pages[0].extract_text())


@pytest.mark.item(7, 8)
def test_pdfs_exist_and_are_real_pdfs():
    for p in (ARCH, OVER):
        assert p.is_file() and p.read_bytes()[:5] == b"%PDF-" and p.stat().st_size > 5_000


@pytest.mark.item(10)
def test_exactly_two_submission_documents():
    pdfs = sorted(x.name for x in (ROOT / "docs").glob("*.pdf"))
    assert pdfs == ["architecture.pdf", "solution_overview.pdf"]
    assert not list(ROOT.rglob("*.pptx")) and not list(ROOT.rglob("*.docx"))


@pytest.mark.item(72, 75)
def test_each_pdf_is_exactly_one_page():
    for p in (ARCH, OVER):
        assert len(PdfReader(str(p)).pages) == 1, p
        if shutil.which("pdfinfo"):
            out = subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True).stdout
            assert re.search(r"^Pages:\s+1$", out, re.M), out


@pytest.mark.item(73)
def test_architecture_shows_every_stage_and_both_ai_features():
    t = text(ARCH).lower()
    for stage in ("raw data", "ingestion", "analytics", "model", "dashboard"):
        assert stage in t, stage
    assert "ask the portfolio" in t and "anomaly detection" in t
    # order of the main flow boxes follows the pipeline
    pos = [t.index(x) for x in ("1 raw data", "2 ingestion", "3 analytics", "4 model", "5 dashboard")]
    assert pos == sorted(pos)


@pytest.mark.item(74)
def test_architecture_names_tools_llm_call_site_and_payload():
    t = text(ARCH)
    for tool in ("pandas", "scikit-learn", "LightGBM", "Streamlit", "Plotly", "SQLite", "pyarrow"):
        assert tool in t, tool
    assert re.search(r"Call site: src/ask\.py", t)
    assert "SENT:" in t and "question text and the table catalog only" in t
    assert "NEVER SENT" in t and "row-level data" in t and "API key" in t
    assert "fall" in t.lower() and "offline" in t.lower()
    # everything the PDF says exists in the repo
    for path in set(re.findall(r"\b((?:src|app)/[\w/]+\.py)", t)):
        assert (ROOT / path).is_file(), path


@pytest.mark.item(76)
def test_overview_has_the_five_required_sections():
    t = text(OVER)
    heads = ["1. What the dashboard shows", "2. The two AI features", "3. Model results and approach", "4. Top 3 portfolio insights", "5. Key assumptions"]
    pos = [t.find(h) for h in heads]
    assert all(p >= 0 for p in pos), pos
    assert pos == sorted(pos)
    low = t.lower()
    for need in ("how.", "trigger.", "false positive.", "false negative.", "human"):
        assert low.count(need) >= (2 if need != "human" else 1), need
    assert "bank" in low and ("it and data" in low or "it teams" in low or "from it" in low)
    for v in ("Overview", "Account", "Ask the Portfolio", "Anomaly review"):
        assert v in t


@pytest.mark.item(77)
def test_metrics_in_overview_equal_metrics_json():
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    t = text(OVER)
    for key, name in (("logistic_regression", "Logistic regression (baseline)"), ("lightgbm", "LightGBM (challenger)")):
        te = m["models"][key]["test"]
        lo, hi = te["auc_ci95"]
        exp = f"{name} {te['auc']:.3f} ({lo:.3f}-{hi:.3f}) {te['ks']:.3f} {te['gini']:.3f}"
        assert exp in t, exp
    assert f"{m['split']['n_test']:,} accounts" in t
    st = re.sub(r"\s+", " ", m["comparison"]["statement"])
    assert st in t, "the PDF must carry the model-comparison statement from metrics.json verbatim"
    assert f"{m['models'][m['champion']]['test']['auc']:.3f}" in t
    # Gini identity holds for what the PDF prints
    for key in ("logistic_regression", "lightgbm"):
        te = m["models"][key]["test"]
        assert abs(te["gini"] - (2 * te["auc"] - 1)) < 1e-9


@pytest.mark.item(77)
def test_architecture_numbers_match_reports(portfolio):
    s = json.loads((ROOT / "reports" / "anomaly_summary.json").read_text())
    t = text(ARCH)
    assert f"{s['flagged']} of {s['accounts']:,} accounts flagged ({s['flag_rate'] * 100:.1f}%)" in t
    assert f"{len(portfolio):,} loan accounts" in t


@pytest.mark.item(78)
def test_top3_insights_match_analytics_output():
    ins = json.loads((ROOT / "reports" / "insights.json").read_text())["insights"]
    t = text(OVER)
    sec = t[t.index("4. Top 3 portfolio insights"):t.index("5. Key assumptions")]
    shown = [i for i in ins if re.sub(r"\s+", " ", i["title"]) in sec]
    assert len(shown) == 3, [i["id"] for i in shown]
    for i in shown:
        first_sentence = re.sub(r"\s+", " ", i["so_what"]).split(". ")[0]
        assert first_sentence[:60] in sec, i["id"]
    cross = [i for i in ins if len(i["dims"]) >= 2]
    assert cross, "analytics must hold a cross-dimensional insight"


@pytest.mark.item(79)
def test_no_text_outside_page_or_overlapping_margin():
    for p in (ARCH, OVER):
        page = PdfReader(str(p)).pages[0]
        w, h = float(page.mediabox.width), float(page.mediabox.height)
        pts = []

        def visit(t, cm, tm, fd, fs):
            if t.strip():
                x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
                y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
                pts.append((x, y, t))

        page.extract_text(visitor_text=visit)
        assert pts
        out = [(x, y, t[:30]) for x, y, t in pts if x < 12 or x > w - 12 or y < 12 or y > h - 12]
        assert not out, (p.name, out[:5])


@pytest.mark.item(73, 79)
def test_rasterised_pages_have_ink_but_leave_margins_blank(tmp_path):
    if not shutil.which("pdftoppm"):
        pytest.skip("pdftoppm missing")  # would count as FAIL in verify_all
    from PIL import Image
    import numpy as np
    for p in (ARCH, OVER):
        out = tmp_path / p.stem
        subprocess.run(["pdftoppm", "-r", "60", "-png", str(p), str(out)], check=True)
        img = np.asarray(Image.open(next(tmp_path.glob(p.stem + "*.png"))).convert("L"))
        h, w = img.shape
        assert (img < 200).mean() > 0.02, "page looks empty"
        m = 6  # px at 60 dpi = ~7 pt
        for edge in (img[:m, :], img[-m:, :], img[:, :m], img[:, -m:]):
            assert (edge < 128).mean() < 0.01, f"{p.name}: ink touching the page edge"
        keep = ROOT / "tests" / "reports" / "pdf_previews"
        keep.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftoppm", "-r", "110", "-png", str(p), str(keep / p.stem)], check=True)
