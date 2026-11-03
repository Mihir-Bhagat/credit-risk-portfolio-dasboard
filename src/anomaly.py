"""Anomaly detection: unusual repayment or bureau patterns versus the account's own history or its peer segment.

Peer segment = loan_type x income_band. Peer signals fire when a value is beyond the peer group's 99th percentile
AND above a materiality floor (so a 1-day difference in a tight group never fires). Own-history signals compare the
last 3 months with the first 9 months of the SAME account. An account is flagged when
  * at least ANOMALY_MIN_SIGNALS independent signals fire, or
  * an own-history deterioration signal fires (a sharp break from the account's own behaviour).
Every signal carries the numbers behind it; the reason text is built from the same numbers.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

PEER_COLS = ["loan_type", "income_band"]


def _fmt(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}".rstrip("0").rstrip(".") if isinstance(x, (float, np.floating)) else str(x)


def detect_anomalies(df: pd.DataFrame, peer_pct: float = config.ANOMALY_PEER_PERCENTILE) -> pd.DataFrame:
    """Return one row per account with signals, a flag and readable reasons. Pure function of `df`."""
    d = df.reset_index(drop=True)
    peer = d[PEER_COLS[0]].astype(str) + " / " + d[PEER_COLS[1]].astype(str)
    g = d.groupby(peer)
    sig: list[tuple] = []  # (name, kind, basis, mask, value, threshold, median, text_fn, severity)

    def hi_peer(col, floor, name, kind, text):
        q = g[col].transform(lambda s: s.quantile(peer_pct))
        med = g[col].transform("median")
        mask = (d[col] > q) & (d[col] >= floor)
        sig.append((name, kind, "peer segment", mask.fillna(False), d[col], q, med, text, (d[col] / max(floor, 1e-9)).clip(0, 3) / 3))

    def lo_peer(col, ceil, name, kind, text):
        q = g[col].transform(lambda s: s.quantile(1 - peer_pct))
        med = g[col].transform("median")
        mask = (d[col] < q) & (d[col] <= ceil)
        sig.append((name, kind, "peer segment", mask.fillna(False), d[col], q, med, text, (1 - d[col]).clip(0, 1)))

    hi_peer("rep_n_missed", 3, "Missed payments", "repayment",
            lambda v, t, m, peer_name: f"Missed {int(v)} of 12 EMIs; peers ({peer_name}) median {_fmt(m)}, 99th percentile {_fmt(t)}")
    hi_peer("rep_max_dpd", 60, "Peak days past due", "repayment",
            lambda v, t, m, peer_name: f"Peak {int(v)} days past due; peers ({peer_name}) median {_fmt(m)}, 99th percentile {_fmt(t)}")
    lo_peer("rep_paid_ratio", 0.6, "Payment shortfall", "repayment",
            lambda v, t, m, peer_name: f"Paid {v * 100:.0f}% of amounts due; peers ({peer_name}) median {m * 100:.0f}%, 1st percentile {t * 100:.0f}%")
    hi_peer("bur_utilization", 0.90, "Bureau utilization", "bureau",
            lambda v, t, m, peer_name: f"External credit utilization {v * 100:.0f}%; peers ({peer_name}) median {m * 100:.0f}%, 99th percentile {t * 100:.0f}%")
    hi_peer("bur_n_written_off", 2, "Written-off bureau lines", "bureau",
            lambda v, t, m, peer_name: f"{int(v)} written-off external lines; peers ({peer_name}) median {_fmt(m)}, 99th percentile {_fmt(t)}")
    hi_peer("bur_max_dpd", 60, "Bureau days past due", "bureau",
            lambda v, t, m, peer_name: f"Worst external DPD {int(v)} days; peers ({peer_name}) median {_fmt(m)}, 99th percentile {_fmt(t)}")
    hi_peer("bur_n_lines", 6, "Many credit lines", "bureau",
            lambda v, t, m, peer_name: f"{int(v)} external credit lines; peers ({peer_name}) median {_fmt(m)}, 99th percentile {_fmt(t)}")

    # Own-history signals (same account, last 3 months vs first 9 months)
    jump = d["rep_dpd_last3"] - d["rep_dpd_first9"]
    m4 = ((jump >= config.ANOMALY_OWN_DPD_JUMP_DAYS) & (d["rep_dpd_last3"] >= config.ANOMALY_OWN_DPD_LAST3_MIN)).fillna(False)
    sig.append(("Sudden deterioration vs own history", "repayment", "own history", m4, jump, pd.Series(config.ANOMALY_OWN_DPD_JUMP_DAYS, index=d.index),
                d["rep_dpd_first9"],
                lambda v, t, m, peer_name: f"Mean days past due rose from {_fmt(m)} (months 1-9) by {_fmt(v)} days in months 10-12",
                (jump / 60).clip(0, 1)))
    drop = d["rep_paid_ratio_first9"] - d["rep_paid_ratio_last3"]
    m5 = (drop >= config.ANOMALY_OWN_PAID_DROP).fillna(False)
    sig.append(("Payments collapsed vs own history", "repayment", "own history", m5, drop, pd.Series(config.ANOMALY_OWN_PAID_DROP, index=d.index),
                d["rep_paid_ratio_first9"],
                None,
                drop.clip(0, 1)))

    n = len(d)
    n_sig = np.zeros(n, dtype=int)
    sev_sum = np.zeros(n)
    own_flag = np.zeros(n, dtype=bool)
    sig_lists: list[list[dict]] = [[] for _ in range(n)]
    for name, kind, basis, mask, val, thr, med, text, sev in sig:
        idx = np.flatnonzero(mask.to_numpy())
        for i in idx:
            if name == "Payments collapsed vs own history":
                txt = f"Payment ratio fell {val.iloc[i] * 100:.0f} points: {med.iloc[i] * 100:.0f}% of dues paid in months 1-9 vs {(med.iloc[i] - val.iloc[i]) * 100:.0f}% in months 10-12"
            else:
                txt = text(float(val.iloc[i]), float(thr.iloc[i]), float(med.iloc[i]), peer.iloc[i])
            sig_lists[i].append({"signal": name, "kind": kind, "basis": basis, "value": float(val.iloc[i]),
                                 "threshold": float(thr.iloc[i]), "reference": float(med.iloc[i]), "text": txt})
        n_sig[idx] += 1
        sev_sum[idx] += sev.to_numpy()[idx]
        if basis == "own history":
            own_flag[idx] = True

    flagged = (n_sig >= config.ANOMALY_MIN_SIGNALS) | (config.ANOMALY_STRONG_SINGLE_SIGNAL & own_flag)
    out = pd.DataFrame({
        "account_id": d["account_id"], "peer_group": peer, "n_signals": n_sig,
        "anomaly_score": np.where(n_sig > 0, n_sig + sev_sum / np.maximum(n_sig, 1), 0.0),
        "flagged": flagged,
        "signal_kinds": [",".join(sorted({s["kind"] for s in sl})) for sl in sig_lists],
        "signals_json": [json.dumps(sl) for sl in sig_lists],
        "reasons": [" | ".join(s["text"] for s in sl) for sl in sig_lists],
    })
    return out
