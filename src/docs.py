"""Generate docs/architecture.pdf and docs/solution_overview.pdf (one page each).

All numbers are read from reports/*.json at generation time (never retyped), and every text block is
checked to fit its box / frame; generation fails loudly instead of silently clipping.
"""
from __future__ import annotations

import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, KeepInFrame, Paragraph, Table, TableStyle

from . import ai_notes, config
from .monitoring import MONITORING_PLAN

INK, MUTED = colors.HexColor("#14213d"), colors.HexColor("#5c6577")
BLUE, ORANGE, AQUA = colors.HexColor("#2a78d6"), colors.HexColor("#eb6834"), colors.HexColor("#1baf7a")
LIGHT_BLUE, LIGHT_ORANGE, LIGHT_GRAY = colors.HexColor("#eaf2fc"), colors.HexColor("#fff1e9"), colors.HexColor("#f3f5f8")


def _reports():
    r = config.REPORTS
    return (json.loads((r / "metrics.json").read_text()), json.loads((r / "insights.json").read_text()),
            json.loads((r / "anomaly_summary.json").read_text()), json.loads((r / "join_audit.json").read_text()))


# ------------------------------------------------------------------------------------------------
# Architecture (landscape, drawn with the canvas)
# ------------------------------------------------------------------------------------------------
def _box(c: canvas.Canvas, x, y, w, h, title, body, fill, stroke, title_color=INK, size=7.4, bold_title=True):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(1.1)
    c.roundRect(x, y, w, h, 5, fill=1, stroke=1)
    c.setFillColor(title_color)
    c.setFont("Helvetica-Bold" if bold_title else "Helvetica", size + 1.4)
    c.drawString(x + 7, y + h - 13, title)
    c.setFillColor(INK)
    c.setFont("Helvetica", size)
    lead = size + 1.8
    cy = y + h - 25
    for para in body:
        lines = simpleSplit(para, "Helvetica", size, w - 14)
        for ln in lines:
            if cy < y + 4:
                raise ValueError(f"text overflows box '{title}': {ln!r}")
            c.drawString(x + 7, cy, ln)
            cy -= lead
        cy -= 1.5


def _arrow(c, x1, y1, x2, y2, color=MUTED, width=1.4):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    L = 6
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - L * math.cos(a - 0.45), y2 - L * math.sin(a - 0.45))
    p.lineTo(x2 - L * math.cos(a + 0.45), y2 - L * math.sin(a + 0.45))
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def make_architecture(path=None) -> None:
    path = path or config.DOCS / "architecture.pdf"
    _, _, anom, audit = _reports()
    W, H = landscape(A4)
    c = canvas.Canvas(str(path), pagesize=(W, H))
    c.setTitle("Architecture: Credit Risk Portfolio Monitor")
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(28, H - 34, "Architecture: raw data to dashboard, with both AI features")
    c.setFont("Helvetica", 8.5)
    c.setFillColor(MUTED)
    c.drawString(28, H - 48, "Credit Risk Portfolio Monitor · Python · Streamlit · scikit-learn · LightGBM · optional free-tier LLM planner · all figures computed locally with pandas")

    # Pipeline row
    bw, bh, gap, top = 140, 128, 20, H - 195
    x0 = 28
    stages = [
        ("1  Raw data", ["data/raw: 5 CSVs as provided.", "customers · loan_accounts · bureau_credit · repayment_history · collections_events",
                         f"{audit['accounts_in']:,} loan accounts.", "Tool: pandas.read_csv"], LIGHT_GRAY),
        ("2  Ingestion", ["src/ingestion.py, src/features.py", "Cleans 14 documented data-quality issues; no rows dropped.",
                          "LEFT joins on customer_id / account_id; row counts asserted.", "Repayment (months 1-12) and bureau aggregated to account grain.", "Tool: pandas, pyarrow"], LIGHT_BLUE),
        ("3  Analytics", ["src/analytics.py", "Profiles by income, employment, region, age, loan type; cross-cuts; trends; 5 insights with a 'so what'.", "Tool: pandas"], LIGHT_BLUE),
        ("4  Model", ["src/model.py", "Logistic regression baseline + LightGBM. Customer-grouped split; GroupKFold tuning on train only; one test evaluation.",
                      "AUC, KS, Gini; TreeSHAP drivers per account; PSI monitoring.", "Tool: scikit-learn, LightGBM"], LIGHT_BLUE),
        ("5  Dashboard", ["app/app.py: 7 views", "Overview · Segments · Account · Ask the Portfolio · Anomaly review · Model · Data & methods.", "Hosted on Streamlit Community Cloud; LLM key in its secrets store.", "Tool: Streamlit, Plotly"], LIGHT_GRAY),
    ]
    xs = []
    for i, (t, body, fill) in enumerate(stages):
        x = x0 + i * (bw + gap)
        xs.append(x)
        _box(c, x, top, bw, bh, t, body, fill, BLUE if fill == LIGHT_BLUE else MUTED)
        if i:
            _arrow(c, x - gap + 1, top + bh / 2, x - 1, top + bh / 2)

    # AI feature 1: Ask the Portfolio
    ay, ah = 62, 232
    aw1 = 520
    ax = 28
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(ax, ay + ah + 8, "AI feature 1: Ask the Portfolio (LLM question answering)")
    steps = [("Question", ["Analyst types a question in the dashboard."]),
             ("Guardrails", ["src/ask.py. Declines injection, raw-row, forecast, causal and off-topic questions before any planner runs."]),
             ("Planner", ["LLM (if a key is set) or offline rules. Returns a JSON plan: metric, group-by, filters."]),
             ("Validate + run", ["Plan checked against a closed list; pandas executes it locally."]),
             ("Answer", ["Text rendered from the result; every number checked against it. Source and plan shown."])]
    sw, sg = 88, 20
    for i, (t, body) in enumerate(steps):
        x = ax + i * (sw + sg)
        hl = t == "Planner"
        _box(c, x, ay + 108, sw, 118, t, body, LIGHT_ORANGE if hl else LIGHT_BLUE, ORANGE if hl else BLUE, size=6.6)
        if i:
            _arrow(c, x - sg + 1, ay + 108 + 59, x - 1, ay + 108 + 59)
    _box(c, ax, ay, aw1, 98, "Where the LLM is called and what is sent",
         ["Call site: src/ask.py, LLMPlanner.plan() · one HTTPS request to an OpenAI-compatible chat endpoint (Groq, Gemini or OpenRouter free tiers), key read server-side from the secrets store.",
          "SENT: the question text and the table catalog only (metric and column descriptions, category names, limits).",
          "RETURNED: one JSON query plan.",
          "NEVER SENT: row-level data, identifiers, computed results or the API key. No key, rate limit or error: the app falls back to the offline interpreter with a visible notice."],
         LIGHT_ORANGE, ORANGE, size=7.0)
    _arrow(c, ax + 4 * (sw + sg) + sw / 2, ay + 108 + 118, xs[4] + bw / 2 - 40, top, ORANGE)

    # AI feature 2: anomaly detection
    bx = 570
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(bx, ay + ah + 8, "AI feature 2: Anomaly detection")
    _box(c, bx, ay + 128, 244, 98, "Detect (src/anomaly.py, pandas)",
         [f"Peer segment (loan type x income band): value above the group's 99th percentile. Own history: last 3 months vs first 9. Flag when 2+ signals, or a sharp own-history break.",
          f"Result: {anom['flagged']:,} of {anom['accounts']:,} accounts flagged ({anom['flag_rate'] * 100:.1f}%), each with reasons."], LIGHT_BLUE, BLUE, size=6.8)
    _box(c, bx, ay + 64, 244, 56, "Review list (dashboard)", ["Analyst reads the reasons with their numbers, then records Confirm or Dismiss with a note."], LIGHT_BLUE, BLUE, size=6.8)
    _box(c, bx, ay, 244, 56, "Decision store (src/decisions.py, SQLite)", ["Timestamp, account id, decision, note, reviewer. Download as CSV. Never changes the model."], LIGHT_GRAY, MUTED, size=6.8)
    _arrow(c, bx + 122, ay + 128, bx + 122, ay + 120)
    _arrow(c, bx + 122, ay + 64, bx + 122, ay + 56)
    _arrow(c, bx + 244, ay + 176, xs[4] + bw / 2 + 40, top, BLUE)
    c.setFont("Helvetica", 7)
    c.setFillColor(MUTED)
    c.drawString(28, 26, "Data flow: raw CSVs -> ingestion -> account-grain dataset -> analytics + model + anomaly flags -> portfolio table -> dashboard. Reproducible with: pip install -r requirements.txt; python -m src.build; streamlit run app/app.py")
    c.drawString(28, 15, "Synthetic data. Protected attributes (gender, marital status) and collections events are kept out of the model.")
    c.showPage()
    c.save()


# ------------------------------------------------------------------------------------------------
# Solution overview (portrait, platypus; fit is asserted)
# ------------------------------------------------------------------------------------------------
def make_solution_overview(path=None) -> None:
    path = path or config.DOCS / "solution_overview.pdf"
    m, ins, anom, audit = _reports()
    W, H = A4
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle("Solution overview: Credit Risk Portfolio Monitor")
    base = ParagraphStyle("b", fontName="Helvetica", fontSize=7.7, leading=9.4, textColor=INK)
    h = ParagraphStyle("h", parent=base, fontName="Helvetica-Bold", fontSize=9.4, leading=11.2, spaceBefore=5, spaceAfter=2, textColor=BLUE)
    small = ParagraphStyle("s", parent=base, fontSize=7.1, leading=8.7)
    bul = ParagraphStyle("bul", parent=base, leftIndent=7, bulletIndent=0)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(28, H - 32, "Solution overview: Credit Risk Portfolio Monitor")
    c.setFont("Helvetica", 7.6)
    c.setFillColor(MUTED)
    c.drawString(28, H - 44, f"{audit['accounts_in']:,} loan accounts from 5 synthetic bank tables · Streamlit dashboard with two embedded AI features · Python, pandas, scikit-learn, LightGBM")

    lr, gb = m["models"]["logistic_regression"]["test"], m["models"]["lightgbm"]["test"]
    sens = m["sensitivity"]
    champ = "LightGBM" if m["champion"] == "lightgbm" else "Logistic regression"
    metric_rows = [["Model (test set)", "AUC (95% CI)", "KS", "Gini"],
                   ["Logistic regression (baseline)", f"{lr['auc']:.3f} ({lr['auc_ci95'][0]:.3f}-{lr['auc_ci95'][1]:.3f})", f"{lr['ks']:.3f}", f"{lr['gini']:.3f}"],
                   ["LightGBM (challenger)", f"{gb['auc']:.3f} ({gb['auc_ci95'][0]:.3f}-{gb['auc_ci95'][1]:.3f})", f"{gb['ks']:.3f}", f"{gb['gini']:.3f}"]]
    tbl = Table(metric_rows, colWidths=[118, 100, 40, 40], rowHeights=11)
    tbl.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 6.8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 6.8),
                             ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9d1de")),
                             ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5), ("LEFTPADDING", (0, 0), (-1, -1), 3)]))

    left: list = [Paragraph("1. What the dashboard shows and how an analyst uses it", h)]
    for t in [
        "<b>Overview.</b> KPIs, exposure and observed vs predicted default by model risk band, trends and profiles by income, employment, region, age and loan type, a cross-dimension heatmap and the portfolio insights. Start here to see where risk and money sit.",
        "<b>Segments &amp; drill-down.</b> Choose a dimension and a segment to list its accounts ranked by predicted risk; open any account or download the list.",
        "<b>Account.</b> Predicted default probability, risk band, the top drivers of the score (TreeSHAP), 12-month repayment chart, bureau lines, peer comparison and any anomaly reasons.",
        "<b>Ask the Portfolio.</b> Plain-English questions answered with exact figures and their source (AI feature 1).",
        "<b>Anomaly review.</b> The review list with reasons and a Confirm/Dismiss decision log (AI feature 2).",
        "<b>Model</b> (ROC, KS, SHAP, calibration, PSI monitoring) and <b>Data &amp; methods</b> (join logic, data-quality log, feature cutoffs).",
        "Sidebar filters (loan type, employment, region, income, age, risk band, origination year) re-cut every portfolio view.",
    ]:
        left.append(Paragraph(t, bul, bulletText="•"))
    left.append(Paragraph("2. The two AI features", h))
    an, ak = ai_notes.ANOMALY_NOTES, ai_notes.ASK_NOTES
    ai_rows = [[Paragraph("<b>Ask the Portfolio</b> (LLM)", small), Paragraph("<b>Anomaly detection</b>", small)],
               [Paragraph("<b>How.</b> " + ak["how"], small), Paragraph("<b>How.</b> " + an["how"], small)],
               [Paragraph("<b>Trigger.</b> " + ak["trigger"], small), Paragraph("<b>Trigger.</b> " + an["trigger"], small)],
               [Paragraph("<b>False positive.</b> " + ak["false_positive"], small), Paragraph("<b>False positive.</b> " + an["false_positive"], small)],
               [Paragraph("<b>False negative.</b> " + ak["false_negative"], small), Paragraph("<b>False negative.</b> " + an["false_negative"], small)],
               [Paragraph("<b>Human review.</b> " + ak["human"] + " <b>Sent to LLM:</b> " + ak["data_sent"], small),
                Paragraph("<b>Human review.</b> " + an["human"] + f" Result: {anom['flagged']:,} flagged ({anom['flag_rate'] * 100:.1f}%); flagged accounts default at {anom['default_rate_flagged'] * 100:.0f}% vs {anom['default_rate_not_flagged'] * 100:.0f}%.", small)]]
    at = Table(ai_rows, colWidths=[255, 255])
    at.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
                            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9d1de")), ("TOPPADDING", (0, 0), (-1, -1), 1.6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    left.append(at)
    left.append(Paragraph("3. Model results and approach", h))
    left.append(tbl)
    left.append(Paragraph(
        f"Test set: {m['split']['n_test']:,} accounts ({m['split']['test_default_rate'] * 100:.1f}% default), split by customer so no customer appears in both sets. "
        f"{m['n_features']} features from loan, customer, repayment months 1-12 and bureau data; collections data and protected attributes are excluded. "
        f"Tuning used GroupKFold on training rows only; the test set was scored once. <b>{m['comparison']['statement']}</b> Dashboard scores use {champ}, chosen by cross-validated AUC. "
        f"Robustness: profile-only AUC {sens['lgbm_test_auc_profile_only']:.3f}, plus repayment {sens['lgbm_test_auc_profile_plus_repayment']:.3f}, plus bureau {sens['lgbm_test_auc_all_features']:.3f}; "
        f"bureau limited to lines opened by month 12: {sens['lgbm_test_auc_bureau_point_in_time']:.3f}. Strongest single feature AUC {sens['max_single_feature_auc']:.3f} (leakage ceiling 0.95). "
        f"<b>Monitoring:</b> " + " ".join(f"{t}: {s}" for t, s in MONITORING_PLAN[:4]), base))
    left.append(Paragraph("4. Top 3 portfolio insights", h))
    by_id = {i["id"]: i for i in ins["insights"]}
    for k in ("I1", "I2", "I5"):
        left.append(Paragraph(f"<b>{by_id[k]['title']}.</b> {by_id[k]['so_what']}", bul, bulletText="•"))
    left.append(Paragraph("5. Key assumptions and what production needs from the bank", h))
    left.append(Paragraph(
        "<b>Assumptions.</b> The observation point is the end of repayment month 12 and the target is assumed to be observed afterwards (no default date exists). "
        "The bureau extract has no as-of date, so it is used as supplied (sensitivity above). Exposure is the original loan amount. Medians used for imputation never use the target. "
        "Data is synthetic: results illustrate the method, not any real bank.", base))
    left.append(Paragraph(
        "<b>From IT and data teams.</b> (1) Scheduled extracts from core banking, the repayment ledger with true balances, bureau pulls with an as-of date, and the collections system, under data contracts with lineage. "
        "(2) A dated default definition (for example 90+ days past due) and clear observation and outcome windows. (3) A secured environment: SSO and role-based access, PII masking, audit logging of every analyst decision, "
        "and an approved LLM gateway or an on-premise model so no customer data leaves the bank. (4) Case-management integration for anomaly decisions. (5) Model-risk validation, fairness review and the monitoring thresholds above.", base))
    f = Frame(24, 22, W - 48, H - 76, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, showBoundary=0)
    story = left
    f.addFromList(story, c)
    if story:
        raise ValueError(f"solution overview does not fit on one page; {len(story)} flowables left over")
    c.showPage()
    c.save()


def make_all() -> None:
    config.DOCS.mkdir(exist_ok=True)
    make_architecture()
    make_solution_overview()
