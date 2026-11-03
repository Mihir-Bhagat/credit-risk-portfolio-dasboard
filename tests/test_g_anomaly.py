"""Group G: anomaly detection (items 64-70). Reasons are recomputed from the raw CSVs; extreme accounts are injected
into a copy of the raw tables and run through the real pipeline."""
from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apphelp import all_text, goto, new_app
from src import anomaly, config, decisions as D, features as F, ingestion


@pytest.fixture(scope="module")
def pf():
    return pd.read_parquet(config.PROCESSED / "portfolio.parquet")


# ---------------------------------------------------------------------------------------------- method coverage
@pytest.mark.item(64)
def test_both_data_sources_and_both_bases_present(pf):
    fl = pf[pf["flagged"]]
    sigs = [s for j in fl["signals_json"] for s in json.loads(j)]
    kinds = {s["kind"] for s in sigs}
    bases = {s["basis"] for s in sigs}
    assert kinds == {"repayment", "bureau"}, kinds
    assert bases == {"peer segment", "own history"}, bases
    names = {s["signal"] for s in sigs}
    assert len(names) >= 6, names
    # each basis is covered by independent code paths with their own thresholds
    assert {"loan_type", "income_band"} == set(anomaly.PEER_COLS)


@pytest.mark.item(66)
def test_flag_rate_inside_preregistered_band(pf):
    rate = float(pf["flagged"].mean())
    print(f"flag rate = {rate:.4%} ({int(pf['flagged'].sum())} of {len(pf)})")
    assert 0.005 <= rate <= 0.10


@pytest.mark.item(66)
def test_flagged_accounts_default_more_often(pf):
    a, b = pf.loc[pf["flagged"], "target_default"].mean(), pf.loc[~pf["flagged"], "target_default"].mean()
    print(f"default rate flagged {a:.3f} vs not flagged {b:.3f}")
    assert a > b, "flags carry no information about outcomes"


@pytest.mark.item(64, 66)
def test_flag_rule_is_exactly_as_documented(pf):
    own = pf["signals_json"].map(lambda j: any(s["basis"] == "own history" for s in json.loads(j)))
    expected = (pf["n_signals"] >= config.ANOMALY_MIN_SIGNALS) | (config.ANOMALY_STRONG_SINGLE_SIGNAL & own)
    assert (expected.to_numpy() == pf["flagged"].to_numpy()).all()
    assert (pf.loc[pf["n_signals"] == 0, "flagged"] == False).all()  # noqa: E712
    assert pf["n_signals"].eq(pf["signals_json"].map(lambda j: len(json.loads(j)))).all()


@pytest.mark.item(64)
def test_detector_is_deterministic_and_matches_stored(pf):
    cols = [c for c in pf.columns if c not in ("flagged", "n_signals", "anomaly_score", "peer_group", "signal_kinds", "signals_json", "reasons")]
    a = anomaly.detect_anomalies(pf[cols])
    b = anomaly.detect_anomalies(pf[cols])
    pd.testing.assert_frame_equal(a, b)
    assert (a["flagged"].to_numpy() == pf["flagged"].to_numpy()).all()


# ---------------------------------------------------------------------------------------------- reasons are true
def _raw_stats(raw):
    r = raw["repayment"]
    r = r[r["payment_month"].between(1, 12)]
    g = r.groupby("account_id")
    rep = pd.DataFrame({"missed": g["payment_status"].apply(lambda s: int((s == "Missed").sum())), "maxdpd": g["days_past_due"].max(),
                        "paid": g["emi_paid"].sum(), "due": g["emi_due"].sum()})
    b = raw["bureau"]
    bg = b.groupby("customer_id")
    bur = pd.DataFrame({"lines": bg.size(), "wo": bg["account_status"].apply(lambda s: int((s == "Written-Off").sum())), "maxdpd": bg["days_past_due"].max(),
                        "out": bg["outstanding_amount"].sum(), "san": bg["sanctioned_amount"].sum()})
    return rep, bur


@pytest.mark.item(67)
def test_every_flag_has_reason_and_numbers_match_raw_data(pf, raw):
    fl = pf[pf["flagged"]]
    assert (fl["reasons"].str.len() > 20).all(), "flag without readable reason"
    rep, bur = _raw_stats(raw)
    cust = raw["loans"].set_index("account_id")["customer_id"]
    checked = {"missed": 0, "dpd": 0, "paid": 0, "wo": 0, "bdpd": 0, "lines": 0, "util": 0, "jump": 0}
    for _, row in fl.iterrows():
        a, c = row["account_id"], cust[row["account_id"]]
        for s in json.loads(row["signals_json"]):
            t = s["text"]
            if s["signal"] == "Missed payments":
                assert int(re.match(r"Missed (\d+) of 12", t).group(1)) == rep.loc[a, "missed"]; checked["missed"] += 1
            elif s["signal"] == "Peak days past due":
                assert int(re.match(r"Peak (\d+) days", t).group(1)) == rep.loc[a, "maxdpd"]; checked["dpd"] += 1
            elif s["signal"] == "Payment shortfall":
                got = int(re.match(r"Paid (\d+)%", t).group(1))
                assert got == round(rep.loc[a, "paid"] / rep.loc[a, "due"] * 100), (a, t); checked["paid"] += 1
            elif s["signal"] == "Written-off bureau lines":
                assert int(re.match(r"(\d+) written-off", t).group(1)) == bur.loc[c, "wo"]; checked["wo"] += 1
            elif s["signal"] == "Bureau days past due":
                assert int(re.match(r"Worst external DPD (\d+)", t).group(1)) == bur.loc[c, "maxdpd"]; checked["bdpd"] += 1
            elif s["signal"] == "Many credit lines":
                assert int(re.match(r"(\d+) external credit lines", t).group(1)) == bur.loc[c, "lines"]; checked["lines"] += 1
            elif s["signal"] == "Bureau utilization":
                got = int(re.match(r"External credit utilization (\d+)%", t).group(1))
                assert got == round(bur.loc[c, "out"] / bur.loc[c, "san"] * 100); checked["util"] += 1
            elif s["signal"] == "Sudden deterioration vs own history":
                r = raw["repayment"]
                r = r[(r["account_id"] == a) & r["payment_month"].between(1, 12)]
                first9, last3 = r[r.payment_month <= 9].days_past_due.mean(), r[r.payment_month >= 10].days_past_due.mean()
                assert abs((last3 - first9) - s["value"]) < 1e-6; checked["jump"] += 1
            # every signal's own numbers must satisfy its stated threshold
            assert s["value"] >= s["threshold"] or s["signal"] in ("Payment shortfall",), (a, s)
    print("reason checks:", checked)
    assert sum(checked.values()) >= 200 and sum(v > 0 for v in checked.values()) >= 5, checked


# ---------------------------------------------------------------------------------------------- true positives
def _inject(raw, n=12):
    """Append n synthetic accounts with extreme behaviour to a COPY of the raw tables."""
    t = {k: v.copy() for k, v in raw.items()}
    base_c = t["customers"].iloc[0].copy()
    base_l = t["loans"].iloc[0].copy()
    new_c, new_l, new_r, new_b = [], [], [], []
    kinds = ["collapse", "all_missed", "bureau", "mixed"]
    for i in range(n):
        cid, aid = f"CUSTX{i:04d}", f"ACCX{i:04d}"
        c = base_c.copy(); c["customer_id"] = cid; new_c.append(c)
        l = base_l.copy(); l["account_id"], l["customer_id"], l["target_default"] = aid, cid, 1
        l["loan_type"], l["loan_amount"] = ["Personal Loan", "Auto Loan", "Mortgage", "Credit Card"][i % 4], 20000.0
        new_l.append(l)
        kind = kinds[i % 4]
        for m in range(1, 13):
            due = 1000.0
            if kind == "collapse":                       # fine for 9 months, then everything stops
                bad = m >= 10
            elif kind == "all_missed":
                bad = True
            elif kind == "bureau":
                bad = m % 6 == 0                          # mild repayment problem only
            else:
                bad = m >= 7
            status = "Missed" if bad else "On-Time"
            new_r.append({"account_id": aid, "payment_month": m, "emi_due": due, "emi_paid": 0.0 if bad else due,
                          "days_past_due": (120 if bad else 0), "payment_status": status, "outstanding_balance": 19000.0})
        if kind in ("bureau", "mixed"):
            for j in range(8):
                new_b.append({"bureau_id": f"BURX{i:03d}{j}", "customer_id": cid, "credit_type": "Personal Loan", "lender_category": "Fintech",
                              "sanctioned_amount": 10000.0, "outstanding_amount": 10000.0, "account_status": "Written-Off" if j < 4 else "Active",
                              "days_past_due": 180.0, "date_opened": pd.Timestamp("2021-01-01"), "date_closed": pd.NaT})
    t["customers"] = pd.concat([t["customers"], pd.DataFrame(new_c)], ignore_index=True)
    t["loans"] = pd.concat([t["loans"], pd.DataFrame(new_l)], ignore_index=True)
    t["repayment"] = pd.concat([t["repayment"], pd.DataFrame(new_r)], ignore_index=True)
    t["bureau"] = pd.concat([t["bureau"], pd.DataFrame(new_b)], ignore_index=True)
    return t, [f"ACCX{i:04d}" for i in range(n)]


@pytest.mark.item(69)
def test_injected_extreme_accounts_are_all_flagged(pf):
    raw = ingestion.load_raw()
    inj, ids = _inject(raw, 12)
    cleaned, _ = ingestion.clean_tables(inj)
    ds = F.build_analysis_dataset(cleaned)
    out = anomaly.detect_anomalies(ds)
    hit = out[out["account_id"].isin(ids)]
    assert len(hit) == 12
    missed = hit[~hit["flagged"]]["account_id"].tolist()
    assert not missed, f"extreme accounts not flagged: {missed}"
    assert hit["reasons"].str.len().min() > 20
    base = out[~out["account_id"].isin(ids)]
    # injecting 12 rows must not change the normal population's flag rate materially (no collateral damage)
    assert abs(base["flagged"].mean() - pf["flagged"].mean()) < 0.003
    print(f"injected 12/12 flagged; normal-population flag rate {base['flagged'].mean():.4%} vs {pf['flagged'].mean():.4%}")


@pytest.mark.item(69)
def test_normal_looking_accounts_are_not_flagged_by_construction(pf):
    """False-positive probe: clone 12 accounts with perfect repayment and no bureau issues; none may be flagged."""
    inj, ids = _inject(ingestion.load_raw(), 12)
    r = inj["repayment"]
    sel = r["account_id"].isin(ids)
    r.loc[sel, ["emi_paid", "days_past_due", "payment_status"]] = [1000.0, 0, "On-Time"]
    inj["bureau"] = inj["bureau"][~inj["bureau"]["customer_id"].str.startswith("CUSTX")]
    cleaned, _ = ingestion.clean_tables(inj)
    out = anomaly.detect_anomalies(F.build_analysis_dataset(cleaned))
    assert not out[out["account_id"].isin(ids)]["flagged"].any()


# ---------------------------------------------------------------------------------------------- human decision
@pytest.mark.item(68)
def test_decision_store_roundtrip_and_audit_trail(tmp_path):
    db = tmp_path / "d.db"
    D.record_decision("ACC200001", "Confirm", "looks bad", "alice", path=db)
    D.record_decision("ACC200001", "Dismiss", "second look: fine", "bob", path=db)
    D.record_decision("ACC200002", "Confirm", "x" * 3, "", path=db)
    h = D.load_decisions(db)
    assert len(h) == 3 and set(h.columns) == {"ts_utc", "account_id", "decision", "note", "reviewer"}
    assert pd.Timestamp(h["ts_utc"].iloc[0]).tzinfo is not None
    st_ = D.latest_status(db).set_index("account_id")
    assert st_.loc["ACC200001", "decision"] == "Dismiss" and st_.loc["ACC200002", "decision"] == "Confirm"
    with pytest.raises(ValueError):
        D.record_decision("ACC1", "Maybe", path=db)
    with pytest.raises(ValueError):
        D.record_decision("", "Confirm", path=db)
    # data really is on disk (a new connection sees it)
    con = sqlite3.connect(db)
    assert con.execute("select count(*) from decisions").fetchone()[0] == 3
    con.close()


@pytest.mark.item(65, 68)
def test_review_list_and_decision_survive_app_restart(pf):
    at = new_app("Anomaly review")
    t = all_text(at)
    assert "Review list" in t
    assert [m.label for m in at.metric] == ["Flagged accounts", "Pending review", "Confirmed", "Dismissed"]
    assert len(at.dataframe) >= 1 and len(at.dataframe[0].value) > 0, "review list empty"
    assert at.metric[0].value == f"{int(pf['flagged'].sum()):,}"
    pick = at.selectbox(key="an_pick")
    acct = pick.value
    flagged_reasons = pf.loc[pf["account_id"] == acct, "signals_json"].iloc[0]
    assert flagged_reasons and "Why {} was flagged".format(acct) in all_text(at)
    # empty note is refused
    [b for b in at.button if b.label == "Record decision"][0].click().run()
    assert any("short note" in e.value for e in at.error)
    at.text_area(key="note_an").set_value("Checked repayment chart: genuine break")
    at.text_input(key="rev_an").set_value("tester")
    at.radio(key="dec_an").set_value("Confirm")
    [b for b in at.button if b.label == "Record decision"][0].click().run()
    assert not at.exception
    assert f"Recorded **Confirm** for {acct}" in all_text(at) or acct in all_text(at)
    # a brand-new app instance (a restart / reload) still shows it
    at2 = new_app("Anomaly review")
    hist = D.load_decisions()
    assert acct in set(hist["account_id"])
    row = hist[hist["account_id"] == acct].iloc[0]
    assert row["decision"] == "Confirm" and row["note"].startswith("Checked repayment") and row["reviewer"] == "tester"
    assert int(at2.metric[2].value.replace(",", "")) >= 1, "confirmed counter did not update"
    assert any(acct in str(d.value.to_dict()) for d in at2.dataframe)


@pytest.mark.item(63, 70)
def test_anomaly_panel_states_trigger_fp_fn_human():
    t = all_text(new_app("Anomaly review")).lower()
    for needle in ("trigger", "false positive", "false negative", "human"):
        assert needle in t, needle
    from src import ai_notes
    for notes in (ai_notes.ANOMALY_NOTES, ai_notes.ASK_NOTES):
        for k in ("how", "trigger", "false_positive", "false_negative", "human"):
            assert len(notes[k]) > 30, k
